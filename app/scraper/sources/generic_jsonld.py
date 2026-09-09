from __future__ import annotations

import json

from bs4 import BeautifulSoup

from app.scraper.crawler import JobCrawler
from app.scraper.extractor import HtmlJobExtractor
from app.scraper.sources.base import (
    ExtractedJob,
    JobSource,
    RawJobPage,
)


class GenericJsonLdSource(JobSource):
    name = "generic_jsonld"

    def __init__(self, crawler: JobCrawler | None = None):
        self.crawler = crawler or JobCrawler()

    def fetch(self, url: str) -> RawJobPage:
        return self.crawler.fetch(
            url=url,
            source=self.name,
        )

    def extract(self, page: RawJobPage) -> ExtractedJob:
        soup = BeautifulSoup(page.html, "html.parser")

        scripts = soup.find_all(
            "script",
            attrs={"type": "application/ld+json"},
        )

        for script in scripts:
            if not script.string:
                continue

            try:
                payload = json.loads(script.string)
            except json.JSONDecodeError:
                continue

            job_posting = self._find_job_posting(payload)

            if job_posting is None:
                continue

            return self._to_job(
                job_posting=job_posting,
                page=page,
            )

        fallback = HtmlJobExtractor()

        return fallback.extract(page)

    def _find_job_posting(self, payload):
        if isinstance(payload, dict):
            if payload.get("@type") == "JobPosting":
                return payload

            graph = payload.get("@graph")

            if isinstance(graph, list):
                for item in graph:
                    if (
                        isinstance(item, dict)
                        and item.get("@type") == "JobPosting"
                    ):
                        return item

        if isinstance(payload, list):
            for item in payload:
                result = self._find_job_posting(item)

                if result:
                    return result

        return None

    def _to_job(
        self,
        job_posting: dict,
        page: RawJobPage,
    ) -> ExtractedJob:

        return ExtractedJob(
            title=job_posting.get("title"),
            company=self._extract_company(job_posting),
            location=self._extract_location(job_posting),
            description=job_posting.get("description"),
            date_posted=job_posting.get("datePosted"),
            source=self.name,
            source_url=page.url,
        )

    @staticmethod
    def _extract_company(job_posting: dict) -> str | None:
        organization = job_posting.get("hiringOrganization")

        if isinstance(organization, dict):
            return organization.get("name")

        return None

    @staticmethod
    def _extract_location(job_posting: dict) -> str | None:
        location = job_posting.get("jobLocation")

        if isinstance(location, list):
            locations = [
                GenericJsonLdSource._parse_location(item)
                for item in location
            ]

            locations = [
                location
                for location in locations
                if location
            ]

            return ", ".join(locations) or None

        if isinstance(location, dict):
            return GenericJsonLdSource._parse_location(location)

        return None

    @staticmethod
    def _parse_location(location: dict) -> str | None:
        address = location.get("address")

        if not isinstance(address, dict):
            return None

        parts = [
            address.get("addressLocality"),
            address.get("addressRegion"),
            address.get("addressCountry"),
        ]

        parts = [
            str(part)
            for part in parts
            if part
        ]

        return ", ".join(parts) or None