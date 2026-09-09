from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.scraper.exceptions import ExtractionError, SourceUnavailableError
from app.scraper.sources.base import (
    ExtractedJob,
    JobReference,
    JobSource,
    RawJobPayload,
)


class TheMuseSource(JobSource):
    """Adapter for The Muse public jobs API."""

    name = "the_muse"
    API_URL = "https://www.themuse.com/api/public/jobs"

    def __init__(
        self,
        api_key: str | None = None,
        category: str | None = None,
        level: str | None = None,
        company_filter: str | None = None,
        descending: bool = True,
        max_pages: int = 25,
        timeout: int = 20,
        session: requests.Session | None = None,
    ):
        load_dotenv()
        self.api_key = os.getenv("THE_MUSE_API_KEY") if api_key is None else api_key
        self.category = self._string(category)
        self.level = self._string(level)
        self.company_filter = self._string(company_filter)
        self.descending = descending
        self.max_pages = max(1, max_pages)
        self.timeout = timeout
        self.session = session or self._build_session()
        self._payloads: dict[str, RawJobPayload] = {}

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "JobApplicationAgent/1.0",
            }
        )
        retry = Retry(
            total=4,
            connect=3,
            read=3,
            status=4,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def discover_jobs(
        self,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobReference]:
        if limit < 1:
            return []

        discovered: list[JobReference] = []
        seen: set[str] = set()
        page_number = 0

        while len(discovered) < limit and page_number < self.max_pages:
            payload = self._request_page(page_number, location=location)
            records = payload.get("results")
            if not isinstance(records, list):
                raise ExtractionError("The Muse response has no results list")
            if not records:
                break

            for record in records:
                if not isinstance(record, Mapping):
                    continue
                if not self._matches(record, query=query, location=location):
                    continue
                source_url = self._source_url(record)
                if not source_url or source_url in seen:
                    continue

                raw = RawJobPayload(
                    source=self.name,
                    source_url=source_url,
                    data=dict(record),
                )
                self._payloads[source_url] = raw
                seen.add(source_url)
                discovered.append(
                    JobReference(
                        source=self.name,
                        url=source_url,
                        external_id=self._string(record.get("id")),
                        title=self._string(record.get("name")),
                        company=self._company(record),
                        location=self._location(record),
                    )
                )
                if len(discovered) >= limit:
                    break

            if self._is_last_page(payload, page_number):
                break
            page_number += 1

        return discovered

    def _request_page(
        self,
        page_number: int,
        location: str | None,
    ) -> dict[str, Any]:
        params: dict[str, object] = {
            "page": page_number,
            "descending": str(self.descending).lower(),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.category:
            params["category"] = self.category
        if self.level:
            params["level"] = self.level
        if self.company_filter:
            params["company"] = self.company_filter
        if location:
            params["location"] = location

        try:
            response = self.session.get(
                self.API_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise SourceUnavailableError(
                f"The Muse API request failed on page {page_number} ({detail})"
            ) from exc
        except ValueError as exc:
            raise SourceUnavailableError(
                f"The Muse API returned invalid JSON on page {page_number}"
            ) from exc

        if not isinstance(payload, dict):
            raise ExtractionError("The Muse API returned a non-object response")
        return payload

    def fetch_job(self, reference: JobReference | str) -> RawJobPayload:
        source_url = reference.url if isinstance(reference, JobReference) else reference
        try:
            return self._payloads[source_url]
        except KeyError as exc:
            raise ExtractionError(
                "The Muse job payload is not cached; call discover_jobs first"
            ) from exc

    def fetch(self, url: str) -> RawJobPayload:
        return self.fetch_job(url)

    def extract(self, raw: RawJobPayload) -> ExtractedJob:
        if not isinstance(raw, RawJobPayload) or not isinstance(raw.data, Mapping):
            raise ExtractionError("Expected a The Muse JSON payload")

        record = raw.data
        raw_description = self._string(record.get("contents"))
        company = record.get("company")
        metadata = {
            "categories": self._names(record.get("categories")),
            "levels": self._names(record.get("levels")),
            "tags": record.get("tags"),
            "job_type": record.get("type"),
            "short_name": record.get("short_name"),
            "model_type": record.get("model_type"),
            "company_id": company.get("id") if isinstance(company, Mapping) else None,
            "company_short_name": company.get("short_name") if isinstance(company, Mapping) else None,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return ExtractedJob(
            title=self._string(record.get("name")),
            company=self._company(record),
            location=self._location(record),
            description=raw_description,
            raw_description=raw_description,
            date_posted=self._string(record.get("publication_date")),
            source=self.name,
            source_url=raw.source_url,
            external_id=self._string(record.get("id")),
            metadata=metadata,
        )

    @staticmethod
    def _source_url(record: Mapping[str, object]) -> str | None:
        refs = record.get("refs")
        if isinstance(refs, Mapping):
            return TheMuseSource._string(refs.get("landing_page"))
        return None

    @staticmethod
    def _company(record: Mapping[str, object]) -> str | None:
        company = record.get("company")
        if isinstance(company, Mapping):
            return TheMuseSource._string(company.get("name"))
        return None

    @staticmethod
    def _location(record: Mapping[str, object]) -> str | None:
        locations = TheMuseSource._names(record.get("locations"))
        return "; ".join(locations) or None

    @staticmethod
    def _names(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        names = []
        for item in value:
            if isinstance(item, Mapping):
                name = TheMuseSource._string(item.get("name"))
                if name:
                    names.append(name)
        return names

    @staticmethod
    def _matches(
        record: Mapping[str, object],
        query: str | None,
        location: str | None,
    ) -> bool:
        if query:
            searchable = " ".join(
                [
                    str(record.get("name") or ""),
                    " ".join(TheMuseSource._names(record.get("categories"))),
                ]
            )
            if query.casefold().strip() not in searchable.casefold():
                return False
        if location:
            searchable_locations = " ".join(
                TheMuseSource._names(record.get("locations"))
            ).casefold()
            if location.casefold().strip() not in searchable_locations:
                return False
        return True

    @staticmethod
    def _is_last_page(payload: Mapping[str, object], page_number: int) -> bool:
        page_count = payload.get("page_count")
        try:
            return page_number + 1 >= int(page_count) if page_count is not None else False
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None
