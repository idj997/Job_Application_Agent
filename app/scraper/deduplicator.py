from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.scraper.sources.base import ExtractedJob


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "msclkid",
    "ref",
    "source",
}


class JobDeduplicator:
    @staticmethod
    def canonicalize_url(url: str) -> str:
        parts = urlsplit(url)

        query_params = parse_qsl(
            parts.query,
            keep_blank_values=True,
        )

        filtered_params = [
            (key, value)
            for key, value in query_params
            if key.lower() not in TRACKING_PARAMS
        ]

        filtered_params.sort()

        canonical_query = urlencode(filtered_params)

        path = parts.path.rstrip("/")

        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                path,
                canonical_query,
                "",
            )
        )

    @staticmethod
    def url_fingerprint(url: str) -> str:
        canonical_url = JobDeduplicator.canonicalize_url(url)

        return hashlib.sha256(
            canonical_url.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def content_fingerprint(job: ExtractedJob) -> str:
        fields = [
            job.title or "",
            job.company or "",
            job.location or "",
            job.description or "",
        ]

        normalized = " ".join(fields)

        normalized = normalized.lower()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def is_same_url(
        url_a: str,
        url_b: str,
    ) -> bool:
        return (
            JobDeduplicator.canonicalize_url(url_a)
            == JobDeduplicator.canonicalize_url(url_b)
        )

    @staticmethod
    def is_exact_duplicate(
        job_a: ExtractedJob,
        job_b: ExtractedJob,
    ) -> bool:
        return (
            JobDeduplicator.content_fingerprint(job_a)
            == JobDeduplicator.content_fingerprint(job_b)
        )