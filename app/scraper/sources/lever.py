from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
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


class LeverSource(JobSource):
    """Adapter for an employer's public Lever Postings API site."""

    name = "lever"
    API_ROOTS = {
        "global": "https://api.lever.co/v0/postings",
        "eu": "https://api.eu.lever.co/v0/postings",
    }
    JOB_ROOTS = {
        "global": "https://jobs.lever.co",
        "eu": "https://jobs.eu.lever.co",
    }
    MAX_RESULTS_PER_PAGE = 100

    def __init__(
        self,
        company_slug: str | None = None,
        company: str | None = None,
        region: str = "global",
        commitment: str | None = None,
        team: str | None = None,
        department: str | None = None,
        timeout: int = 20,
        session: requests.Session | None = None,
    ):
        load_dotenv()
        configured_slug = (
            os.getenv("LEVER_COMPANY_SLUG")
            if company_slug is None
            else company_slug
        )
        configured_company = os.getenv("LEVER_COMPANY") if company is None else company
        self.company_slug = self._normalize_company_slug(configured_slug)
        self.company = self._string(configured_company)
        self.region = region.casefold().strip()
        self.commitment = self._string(commitment)
        self.team = self._string(team)
        self.department = self._string(department)
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

        discovered: list[JobReference] = []
        seen: set[str] = set()
        skip = 0

        while len(discovered) < limit:
            records = self._request_page(
                skip=skip,
                page_size=self.MAX_RESULTS_PER_PAGE,
            )
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
                        title=self._string(record.get("text")),
                        company=self.company,
                        location=self._location(record),
                    )
                )
                if len(discovered) >= limit:
                    break

            if len(records) < self.MAX_RESULTS_PER_PAGE:
                break
            skip += len(records)

        return discovered

    def _request_page(self, skip: int, page_size: int) -> list[dict[str, Any]]:
        params: dict[str, object] = {
            "mode": "json",
            "skip": skip,
            "limit": page_size,
        }
        if self.commitment:
            params["commitment"] = self.commitment
        if self.team:
            params["team"] = self.team
        if self.department:
            params["department"] = self.department

        url = f"{self.API_ROOTS[self.region]}/{self.company_slug}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise SourceUnavailableError(
                f"Lever postings request failed ({detail})"
            ) from exc
        except ValueError as exc:
            raise SourceUnavailableError(
                "Lever postings API returned invalid JSON"
            ) from exc

        if not isinstance(payload, list):
            raise ExtractionError("Lever postings API returned a non-list response")
        return [dict(record) for record in payload if isinstance(record, Mapping)]

    def fetch_job(self, reference: JobReference | str) -> RawJobPayload:
        source_url = reference.url if isinstance(reference, JobReference) else reference
        try:
            return self._payloads[source_url]
        except KeyError as exc:
            raise ExtractionError(
                "Lever job payload is not cached; call discover_jobs first"
            ) from exc

    def fetch(self, url: str) -> RawJobPayload:
        return self.fetch_job(url)

    def extract(self, raw: RawJobPayload) -> ExtractedJob:
        if not isinstance(raw, RawJobPayload) or not isinstance(raw.data, Mapping):
            raise ExtractionError("Expected a Lever JSON payload")

        record = raw.data
        raw_description = self._description(record)
        categories = record.get("categories")
        salary_range = record.get("salaryRange")
        metadata = {
            "company_slug": self.company_slug,
            "region": self.region,
            "commitment": self._category(categories, "commitment"),
            "team": self._category(categories, "team"),
            "department": self._category(categories, "department"),
            "all_locations": self._category_value(categories, "allLocations"),
            "workplace_type": record.get("workplaceType"),
            "country": record.get("country"),
            "apply_url": record.get("applyUrl"),
            "salary_range": dict(salary_range) if isinstance(salary_range, Mapping) else salary_range,
            "salary_description": record.get("salaryDescriptionPlain"),
            "created_at": record.get("createdAt"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return ExtractedJob(
            title=self._string(record.get("text")),
            company=self.company,
            location=self._location(record),
            description=raw_description,
            raw_description=raw_description,
            date_posted=self._created_at(record.get("createdAt")),
            salary=self._salary(record),
            source=self.name,
            source_url=raw.source_url,
            external_id=self._string(record.get("id")),
            metadata=metadata,
        )

    def _validate_configuration(self) -> None:
        if not self.company_slug:
            raise SourceConfigurationError(
                "Lever requires a company slug via --company-slug or "
                "LEVER_COMPANY_SLUG"
            )
        if self.region not in self.API_ROOTS:
            raise SourceConfigurationError(
                "Lever region must be 'global' or 'eu'"
            )

    def _source_url(self, record: Mapping[str, object]) -> str | None:
        source_url = self._string(record.get("hostedUrl"))
        if source_url:
            return source_url
        external_id = self._string(record.get("id"))
        if external_id and self.company_slug:
            return f"{self.JOB_ROOTS[self.region]}/{self.company_slug}/{external_id}"
        return None

    @staticmethod
    def _description(record: Mapping[str, object]) -> str | None:
        parts: list[str] = []
        description = LeverSource._string(
            record.get("description") or record.get("descriptionPlain")
        )
        if description:
            parts.append(description)

        lists = record.get("lists")
        if isinstance(lists, list):
            for item in lists:
                if not isinstance(item, Mapping):
                    continue
                heading = LeverSource._string(item.get("text"))
                content = LeverSource._string(item.get("content"))
                if heading:
                    parts.append(f"<h2>{heading}</h2>")
                if content:
                    parts.append(f"<ul>{content}</ul>")

        additional = LeverSource._string(
            record.get("additional") or record.get("additionalPlain")
        )
        if additional:
            parts.append(additional)
        return "\n".join(parts) or None

    @staticmethod
    def _location(record: Mapping[str, object]) -> str | None:
        return LeverSource._category(record.get("categories"), "location")

    @staticmethod
    def _category(categories: object, key: str) -> str | None:
        value = LeverSource._category_value(categories, key)
        return LeverSource._string(value)

    @staticmethod
    def _category_value(categories: object, key: str) -> object:
        if isinstance(categories, Mapping):
            return categories.get(key)
        return None

    @staticmethod
    def _matches(
        record: Mapping[str, object],
        query: str | None,
        location: str | None,
    ) -> bool:
        categories = record.get("categories")
        if query:
            searchable = " ".join(
                str(value or "")
                for value in (
                    record.get("text"),
                    LeverSource._category_value(categories, "team"),
                    LeverSource._category_value(categories, "department"),
                )
            )
            if query.casefold().strip() not in searchable.casefold():
                return False

        if location:
            location_values = [LeverSource._category(categories, "location") or ""]
            all_locations = LeverSource._category_value(categories, "allLocations")
            if isinstance(all_locations, list):
                location_values.extend(str(value) for value in all_locations)
            searchable_locations = " ".join(location_values).casefold()
            if location.casefold().strip() not in searchable_locations:
                return False
        return True

    def _salary(self, record: Mapping[str, object]) -> str | None:
        salary_range = record.get("salaryRange")
        if isinstance(salary_range, Mapping):
            minimum = salary_range.get("min")
            maximum = salary_range.get("max")
            currency = self._currency(salary_range.get("currency"))
            interval = self._string(salary_range.get("interval"))
            minimum_text = self._format_number(minimum)
            maximum_text = self._format_number(maximum)
            if minimum_text and maximum_text and minimum_text != maximum_text:
                result = f"{currency}{minimum_text} - {currency}{maximum_text}"
            else:
                value = minimum_text or maximum_text
                result = f"{currency}{value}" if value else ""
            interval_label = self._interval_label(interval)
            if result and interval_label:
                result = f"{result} {interval_label}"
            if result:
                return result
        return self._string(
            record.get("salaryDescriptionPlain") or record.get("salaryDescription")
        )

    @staticmethod
    def _currency(value: object) -> str:
        currency = LeverSource._string(value) or ""
        return {
            "GBP": "£",
            "USD": "$",
            "EUR": "€",
        }.get(currency.upper(), f"{currency} " if currency else "")

    @staticmethod
    def _interval_label(value: str | None) -> str | None:
        if not value:
            return None
        return {
            "per-year-salary": "per year",
            "per-month-salary": "per month",
            "semi-month-salary": "per half-month",
            "bi-month-salary": "per two months",
            "bi-week-salary": "per two weeks",
            "per-week-salary": "per week",
            "per-day-wage": "per day",
            "per-hour-wage": "per hour",
            "one-time": "one time",
        }.get(value, value.replace("_", " ").replace("-", " "))

    @staticmethod
    def _format_number(value: object) -> str | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return LeverSource._string(value)
        return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"

    @staticmethod
    def _created_at(value: object) -> str | None:
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
            except (OSError, OverflowError, ValueError):
                return str(value)
        return LeverSource._string(value)

    @staticmethod
    def _normalize_company_slug(value: object) -> str | None:
        slug = LeverSource._string(value)
        if not slug:
            return None
        if "://" in slug:
            parts = [part for part in urlsplit(slug).path.split("/") if part]
            if "postings" in parts:
                index = parts.index("postings")
                return parts[index + 1] if index + 1 < len(parts) else None
            return parts[0] if parts else None
        return slug.strip("/") or None

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None
