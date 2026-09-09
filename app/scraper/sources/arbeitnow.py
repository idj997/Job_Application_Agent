from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.scraper.exceptions import ExtractionError, SourceUnavailableError
from app.scraper.sources.base import (
    ExtractedJob,
    JobReference,
    JobSource,
    RawJobPayload,
)


class ArbeitnowSource(JobSource):
    """Adapter for Arbeitnow's public UK job-board API."""

    name = "arbeitnow"
    API_URL = "https://www.arbeitnow.co.uk/api/job-board-api"
    BASE_URL = "https://www.arbeitnow.co.uk"

    def __init__(
        self,
        visa_sponsorship: bool | None = None,
        timeout: int = 20,
        session: requests.Session | None = None,
    ):
        self.visa_sponsorship = visa_sponsorship
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
        page_number = 1

        while len(discovered) < limit:
            response_payload = self._request_page(page_number)
            records = response_payload.get("data")
            if not isinstance(records, list):
                raise ExtractionError("Arbeitnow response has no data list")

            for record in records:
                if not isinstance(record, Mapping):
                    continue
                if not self._matches(record, query=query, location=location):
                    continue

                source_url = self._source_url(record)
                if not source_url or source_url in seen:
                    continue

                external_id = self._external_id(record)
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
                        external_id=external_id,
                        title=self._string(record.get("title")),
                        company=self._string(record.get("company_name")),
                        location=self._string(record.get("location")),
                    )
                )
                if len(discovered) >= limit:
                    break

            if not self._has_next_page(response_payload, page_number):
                break
            page_number += 1

        return discovered

    def _request_page(self, page_number: int) -> dict[str, Any]:
        params: dict[str, object] = {"page": page_number}
        if self.visa_sponsorship is not None:
            params["visa_sponsorship"] = str(self.visa_sponsorship).lower()

        try:
            response = self.session.get(
                self.API_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise SourceUnavailableError(
                f"Arbeitnow API request failed on page {page_number}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ExtractionError("Arbeitnow API returned a non-object response")
        return payload

    def fetch_job(self, reference: JobReference | str) -> RawJobPayload:
        source_url = reference.url if isinstance(reference, JobReference) else reference
        try:
            return self._payloads[source_url]
        except KeyError as exc:
            raise ExtractionError(
                "Arbeitnow job payload is not cached; call discover_jobs first"
            ) from exc

    def fetch(self, url: str) -> RawJobPayload:
        return self.fetch_job(url)

    def extract(self, raw: RawJobPayload) -> ExtractedJob:
        if not isinstance(raw, RawJobPayload) or not isinstance(raw.data, Mapping):
            raise ExtractionError("Expected an Arbeitnow JSON payload")

        record = raw.data
        raw_description = self._string(record.get("description"))
        job_types = record.get("job_types")

        metadata = {
            "remote": record.get("remote"),
            "visa_sponsorship": record.get("visa_sponsorship"),
            "tags": record.get("tags"),
            "employment_type": job_types,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return ExtractedJob(
            title=self._string(record.get("title")),
            company=self._string(record.get("company_name")),
            location=self._string(record.get("location")),
            description=raw_description,
            raw_description=raw_description,
            date_posted=self._date_posted(record.get("created_at")),
            salary=self._string(record.get("salary")),
            source=self.name,
            source_url=raw.source_url,
            external_id=self._external_id(record),
            metadata=metadata,
        )

    @classmethod
    def _source_url(cls, record: Mapping[str, object]) -> str | None:
        value = cls._string(record.get("url"))
        return urljoin(cls.BASE_URL, value) if value else None

    @staticmethod
    def _external_id(record: Mapping[str, object]) -> str | None:
        for key in ("id", "slug"):
            value = ArbeitnowSource._string(record.get(key))
            if value:
                return value
        return None

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None

    @staticmethod
    def _date_posted(value: object) -> str | None:
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value, timezone.utc).isoformat()
            except (OSError, OverflowError, ValueError):
                return str(value)
        return ArbeitnowSource._string(value)

    @staticmethod
    def _matches(
        record: Mapping[str, object],
        query: str | None,
        location: str | None,
    ) -> bool:
        if query:
            tags = record.get("tags")
            tag_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags or "")
            searchable = " ".join(
                str(record.get(field) or "")
                for field in ("title", "company_name", "description")
            )
            haystack = f"{searchable} {tag_text}".casefold()
            if query.casefold().strip() not in haystack:
                return False

        if location:
            requested_location = location.casefold().strip()
            uk_aliases = {"united kingdom", "uk", "gb", "great britain"}
            record_location = str(record.get("location") or "").casefold()
            if (
                requested_location not in uk_aliases
                and requested_location not in record_location
            ):
                return False

        return True

    @staticmethod
    def _has_next_page(payload: Mapping[str, object], page_number: int) -> bool:
        links = payload.get("links")
        if isinstance(links, Mapping) and "next" in links:
            return bool(links.get("next"))

        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            current = meta.get("current_page", page_number)
            last = meta.get("last_page")
            try:
                return int(current) < int(last) if last is not None else False
            except (TypeError, ValueError):
                return False

        return False
