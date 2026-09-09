from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.scraper.exceptions import (
    ExtractionError,
    SourceConfigurationError,
    SourceUnavailableError,
)
from app.scraper.sources.base import (
    ExtractedJob,
    JobReference,
    JobSource,
    RawJobPayload,
)


class GreenhouseSource(JobSource):
    """Adapter for a public employer-specific Greenhouse job board."""

    name = "greenhouse"
    API_ROOT = "https://boards-api.greenhouse.io/v1/boards"

    def __init__(
        self,
        board_token: str | None = None,
        company: str | None = None,
        timeout: int = 20,
        session: requests.Session | None = None,
    ):
        load_dotenv()
        configured_token = (
            os.getenv("GREENHOUSE_BOARD_TOKEN")
            if board_token is None
            else board_token
        )
        configured_company = (
            os.getenv("GREENHOUSE_COMPANY") if company is None else company
        )
        self.board_token = self._normalize_board_token(configured_token)
        self.company = self._string(configured_company)
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
        self._validate_configuration()
        if limit < 1:
            return []

        if not self.company:
            self.company = self._request_board_name()

        payload = self._request_json(
            path="jobs",
            params={"content": "true"},
        )
        records = payload.get("jobs")
        if not isinstance(records, list):
            raise ExtractionError("Greenhouse response has no jobs list")

        discovered: list[JobReference] = []
        seen: set[str] = set()
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
                    title=self._string(record.get("title")),
                    company=self.company,
                    location=self._location(record),
                )
            )
            if len(discovered) >= limit:
                break

        return discovered

    def _request_board_name(self) -> str | None:
        payload = self._request_json()
        return self._string(payload.get("name"))

    def _request_json(
        self,
        path: str | None = None,
        params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.API_ROOT}/{self.board_token}"
        if path:
            url = f"{url}/{path}"

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise SourceUnavailableError(
                f"Greenhouse board request failed ({detail})"
            ) from exc
        except ValueError as exc:
            raise SourceUnavailableError(
                "Greenhouse board returned invalid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise ExtractionError("Greenhouse API returned a non-object response")
        return payload

    def fetch_job(self, reference: JobReference | str) -> RawJobPayload:
        source_url = reference.url if isinstance(reference, JobReference) else reference
        try:
            return self._payloads[source_url]
        except KeyError as exc:
            raise ExtractionError(
                "Greenhouse job payload is not cached; call discover_jobs first"
            ) from exc

    def fetch(self, url: str) -> RawJobPayload:
        return self.fetch_job(url)

    def extract(self, raw: RawJobPayload) -> ExtractedJob:
        if not isinstance(raw, RawJobPayload) or not isinstance(raw.data, Mapping):
            raise ExtractionError("Expected a Greenhouse JSON payload")

        record = raw.data
        raw_description = self._string(record.get("content"))
        metadata = {
            "updated_at": record.get("updated_at"),
            "internal_job_id": record.get("internal_job_id"),
            "requisition_id": record.get("requisition_id"),
            "language": record.get("language"),
            "departments": record.get("departments"),
            "offices": record.get("offices"),
            "greenhouse_metadata": record.get("metadata"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return ExtractedJob(
            title=self._string(record.get("title")),
            company=self.company,
            location=self._location(record),
            description=raw_description,
            raw_description=raw_description,
            source=self.name,
            source_url=raw.source_url,
            external_id=self._string(record.get("id")),
            metadata=metadata,
        )

    def _validate_configuration(self) -> None:
        if not self.board_token:
            raise SourceConfigurationError(
                "Greenhouse requires a board token via --board-token or "
                "GREENHOUSE_BOARD_TOKEN"
            )

    def _source_url(self, record: Mapping[str, object]) -> str | None:
        source_url = self._string(record.get("absolute_url"))
        if source_url:
            return source_url
        external_id = self._string(record.get("id"))
        if external_id and self.board_token:
            return (
                f"https://job-boards.greenhouse.io/"
                f"{self.board_token}/jobs/{external_id}"
            )
        return None

    @staticmethod
    def _location(record: Mapping[str, object]) -> str | None:
        location = record.get("location")
        if isinstance(location, Mapping):
            return GreenhouseSource._string(location.get("name"))
        return None

    @staticmethod
    def _matches(
        record: Mapping[str, object],
        query: str | None,
        location: str | None,
    ) -> bool:
        if query:
            departments = record.get("departments")
            department_text = ""
            if isinstance(departments, list):
                department_text = " ".join(
                    str(department.get("name") or "")
                    for department in departments
                    if isinstance(department, Mapping)
                )
            searchable = " ".join(
                [
                    str(record.get("title") or ""),
                    department_text,
                ]
            )
            if query.casefold().strip() not in searchable.casefold():
                return False

        if location:
            record_location = GreenhouseSource._location(record) or ""
            if location.casefold().strip() not in record_location.casefold():
                return False

        return True

    @staticmethod
    def _normalize_board_token(value: object) -> str | None:
        token = GreenhouseSource._string(value)
        if not token:
            return None
        if "://" in token:
            path_parts = [part for part in urlsplit(token).path.split("/") if part]
            return path_parts[-1] if path_parts else None
        return token.strip("/") or None

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None
