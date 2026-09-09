from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

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


class AdzunaSource(JobSource):
    """Adapter for the Adzuna UK job-search API."""

    name = "adzuna"
    API_ROOT = "https://api.adzuna.com/v1/api/jobs"
    MAX_RESULTS_PER_PAGE = 50

    def __init__(
        self,
        app_id: str | None = None,
        app_key: str | None = None,
        country: str = "gb",
        salary_min: int | float | None = None,
        salary_max: int | float | None = None,
        full_time: bool | None = None,
        permanent: bool | None = None,
        timeout: int = 20,
        session: requests.Session | None = None,
    ):
        load_dotenv()
        self.app_id = os.getenv("ADZUNA_APP_ID") if app_id is None else app_id
        self.app_key = os.getenv("ADZUNA_APP_KEY") if app_key is None else app_key
        self.country = country.casefold()
        self.salary_min = self._salary_filter(salary_min, "salary_min")
        self.salary_max = self._salary_filter(salary_max, "salary_max")
        self.full_time = full_time
        self.permanent = permanent
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
        self._validate_credentials()
        if limit < 1:
            return []

        page_size = min(limit, self.MAX_RESULTS_PER_PAGE)
        page_number = 1
        discovered: list[JobReference] = []
        seen: set[str] = set()

        while len(discovered) < limit:
            response_payload = self._request_page(
                page_number=page_number,
                page_size=page_size,
                query=query,
                location=self._api_location(location),
            )
            records = response_payload.get("results")
            if not isinstance(records, list):
                raise ExtractionError("Adzuna response has no results list")
            if not records:
                break

            for record in records:
                if not isinstance(record, Mapping):
                    continue
                source_url = self._string(record.get("redirect_url"))
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
                        company=self._company(record),
                        location=self._location(record),
                    )
                )
                if len(discovered) >= limit:
                    break

            if self._is_last_page(
                payload=response_payload,
                page_number=page_number,
                page_size=page_size,
                result_count=len(records),
            ):
                break
            page_number += 1

        return discovered

    def _request_page(
        self,
        page_number: int,
        page_size: int,
        query: str | None,
        location: str | None,
    ) -> dict[str, Any]:
        params: dict[str, object] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": page_size,
            "content-type": "application/json",
        }
        if query:
            params["what"] = query
        if location:
            params["where"] = location
        if self.salary_min is not None:
            params["salary_min"] = self.salary_min
        if self.salary_max is not None:
            params["salary_max"] = self.salary_max
        if self.full_time:
            params["full_time"] = "1"
        if self.permanent:
            params["permanent"] = "1"

        url = f"{self.API_ROOT}/{self.country}/search/{page_number}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise SourceUnavailableError(
                f"Adzuna API request failed on page {page_number} ({detail})"
            ) from exc
        except ValueError as exc:
            raise SourceUnavailableError(
                f"Adzuna API returned invalid JSON on page {page_number}"
            ) from exc

        if not isinstance(payload, dict):
            raise ExtractionError("Adzuna API returned a non-object response")
        return payload

    def fetch_job(self, reference: JobReference | str) -> RawJobPayload:
        source_url = reference.url if isinstance(reference, JobReference) else reference
        try:
            return self._payloads[source_url]
        except KeyError as exc:
            raise ExtractionError(
                "Adzuna job payload is not cached; call discover_jobs first"
            ) from exc

    def fetch(self, url: str) -> RawJobPayload:
        return self.fetch_job(url)

    def extract(self, raw: RawJobPayload) -> ExtractedJob:
        if not isinstance(raw, RawJobPayload) or not isinstance(raw.data, Mapping):
            raise ExtractionError("Expected an Adzuna JSON payload")

        record = raw.data
        raw_description = self._string(record.get("description"))
        category = record.get("category")
        location = record.get("location")

        metadata = {
            "description_type": "summary",
            "contract_time": record.get("contract_time"),
            "contract_type": record.get("contract_type"),
            "category": dict(category) if isinstance(category, Mapping) else category,
            "location_area": location.get("area") if isinstance(location, Mapping) else None,
            "salary_min": record.get("salary_min"),
            "salary_max": record.get("salary_max"),
            "salary_is_predicted": record.get("salary_is_predicted"),
            "latitude": record.get("latitude"),
            "longitude": record.get("longitude"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return ExtractedJob(
            title=self._string(record.get("title")),
            company=self._company(record),
            location=self._location(record),
            description=raw_description,
            raw_description=raw_description,
            date_posted=self._string(record.get("created")),
            salary=self._salary(record),
            source=self.name,
            source_url=raw.source_url,
            external_id=self._string(record.get("id")),
            metadata=metadata,
        )

    def _validate_credentials(self) -> None:
        if not self.app_id or not self.app_key:
            raise SourceConfigurationError(
                "Adzuna requires ADZUNA_APP_ID and ADZUNA_APP_KEY"
            )

    def _api_location(self, location: str | None) -> str | None:
        if not location or not location.strip():
            return None
        normalized = location.strip().casefold().replace(".", "")
        if self.country == "gb" and normalized in {
            "uk",
            "united kingdom",
            "great britain",
        }:
            return None
        return location.strip()

    @staticmethod
    def _salary_filter(
        value: int | float | None,
        name: str,
    ) -> int | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SourceConfigurationError(f"Adzuna {name} must be an integer") from exc
        if not number.is_integer():
            raise SourceConfigurationError(f"Adzuna {name} must be an integer")
        return int(number)

    @staticmethod
    def _company(record: Mapping[str, object]) -> str | None:
        company = record.get("company")
        if isinstance(company, Mapping):
            return AdzunaSource._string(company.get("display_name"))
        return None

    @staticmethod
    def _location(record: Mapping[str, object]) -> str | None:
        location = record.get("location")
        if not isinstance(location, Mapping):
            return None
        display_name = AdzunaSource._string(location.get("display_name"))
        if display_name:
            return display_name
        area = location.get("area")
        if isinstance(area, list):
            values = [str(value).strip() for value in area if str(value).strip()]
            return ", ".join(reversed(values)) or None
        return None

    def _salary(self, record: Mapping[str, object]) -> str | None:
        minimum = record.get("salary_min")
        maximum = record.get("salary_max")
        if minimum is None and maximum is None:
            return None

        currency = "£" if self.country == "gb" else ""
        minimum_text = self._format_number(minimum) if minimum is not None else None
        maximum_text = self._format_number(maximum) if maximum is not None else None
        if minimum_text and maximum_text and minimum_text != maximum_text:
            return f"{currency}{minimum_text} - {currency}{maximum_text}"
        value = minimum_text or maximum_text
        return f"{currency}{value}" if value else None

    @staticmethod
    def _format_number(value: object) -> str | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return AdzunaSource._string(value)
        return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None

    @staticmethod
    def _is_last_page(
        payload: Mapping[str, object],
        page_number: int,
        page_size: int,
        result_count: int,
    ) -> bool:
        if result_count < page_size:
            return True
        count = payload.get("count")
        try:
            return page_number * page_size >= int(count) if count is not None else False
        except (TypeError, ValueError):
            return False
