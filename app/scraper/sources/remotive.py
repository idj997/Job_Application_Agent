from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


class RemotiveSource(JobSource):
    """Adapter for Remotive's public remote-jobs API."""

    name = "remotive"
    API_URL = "https://remotive.com/api/remote-jobs"

    def __init__(
        self,
        category: str | None = None,
        company_name: str | None = None,
        timeout: int = 20,
        session: requests.Session | None = None,
    ):
        self.category = self._string(category)
        self.company_name = self._string(company_name)
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

        payload = self._request_jobs(
            query=query,
            limit=None if query or location else limit,
        )
        records = payload.get("jobs")
        if not isinstance(records, list):
            raise ExtractionError("Remotive response has no jobs list")

        discovered: list[JobReference] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if not self._matches(record, query=query, location=location):
                continue
            source_url = self._string(record.get("url"))
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
                    title=self._string(record.get("title")),
                    company=self._string(record.get("company_name")),
                    location=self._string(record.get("candidate_required_location")),
                )
            )
            if len(discovered) >= limit:
                break
        return discovered

    def _request_jobs(
        self,
        query: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        params: dict[str, object] = {}
        if query:
            params["search"] = query
        if self.category:
            params["category"] = self.category
        if self.company_name:
            params["company_name"] = self.company_name
        if limit is not None:
            params["limit"] = limit

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
                f"Remotive API request failed ({detail})"
            ) from exc
        except ValueError as exc:
            raise SourceUnavailableError(
                "Remotive API returned invalid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise ExtractionError("Remotive API returned a non-object response")
        return payload

    def fetch_job(self, reference: JobReference | str) -> RawJobPayload:
        source_url = reference.url if isinstance(reference, JobReference) else reference
        try:
            return self._payloads[source_url]
        except KeyError as exc:
            raise ExtractionError(
                "Remotive job payload is not cached; call discover_jobs first"
            ) from exc

    def fetch(self, url: str) -> RawJobPayload:
        return self.fetch_job(url)

    def extract(self, raw: RawJobPayload) -> ExtractedJob:
        if not isinstance(raw, RawJobPayload) or not isinstance(raw.data, Mapping):
            raise ExtractionError("Expected a Remotive JSON payload")

        record = raw.data
        raw_description = self._string(record.get("description"))
        metadata = {
            "remote": True,
            "category": record.get("category"),
            "employment_type": record.get("job_type"),
            "company_logo": record.get("company_logo"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return ExtractedJob(
            title=self._string(record.get("title")),
            company=self._string(record.get("company_name")),
            location=self._string(record.get("candidate_required_location")),
            description=raw_description,
            raw_description=raw_description,
            date_posted=self._string(record.get("publication_date")),
            salary=self._string(record.get("salary")),
            source=self.name,
            source_url=raw.source_url,
            external_id=self._string(record.get("id")),
            metadata=metadata,
        )

    @staticmethod
    def _matches(
        record: Mapping[str, object],
        query: str | None,
        location: str | None,
    ) -> bool:
        if query:
            searchable = " ".join(
                [
                    str(record.get("title") or ""),
                    str(record.get("category") or ""),
                ]
            )
            if query.casefold().strip() not in searchable.casefold():
                return False
        if location:
            required_location = str(record.get("candidate_required_location") or "")
            requested = location.casefold().strip()
            available = required_location.casefold()
            if not (
                requested in available
                or "worldwide" in available
                or "anywhere" in available
            ):
                return False
        return True

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None
