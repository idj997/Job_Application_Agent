from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
from app.scraper.deduplicator import JobDeduplicator
from app.scraper.sources.base import ExtractedJob, RawJobPage, RawJobPayload


class JobStorage:
    def __init__(
        self,
        raw_dir: str = "data/raw_jobs",
        processed_dir: str = "data/processed_jobs",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_id(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def save_raw_page(self, page: RawJobPage) -> Path:
        job_id = JobDeduplicator.url_fingerprint(
            page.url
        )[:16]

        path = self.raw_dir / f"{job_id}.html"

        path.write_text(
            page.html,
            encoding="utf-8",
        )

        return path

    def save_raw_payload(self, payload: RawJobPayload) -> Path:
        job_id = JobDeduplicator.url_fingerprint(
            payload.source_url
        )[:16]
        path = self.raw_dir / f"{job_id}.json"
        path.write_text(
            json.dumps(payload.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def save_raw(self, raw: RawJobPage | RawJobPayload) -> Path:
        if isinstance(raw, RawJobPage):
            return self.save_raw_page(raw)
        if isinstance(raw, RawJobPayload):
            return self.save_raw_payload(raw)
        raise TypeError(f"Unsupported raw job type: {type(raw).__name__}")

    def duplicate_kind(self, job: ExtractedJob) -> str | None:
        if not job.source_url:
            return None

        canonical_url = JobDeduplicator.canonicalize_url(job.source_url)
        job_id = JobDeduplicator.url_fingerprint(canonical_url)[:16]
        if (self.processed_dir / f"{job_id}.json").exists():
            return "url"

        content_hash = JobDeduplicator.content_fingerprint(job)
        for path in self.processed_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("content_hash") == content_hash:
                return "content"

        return None

    def save_job(self, job: ExtractedJob) -> Path:
        if not job.source_url:
            raise ValueError("Job must have source_url before saving.")

        canonical_url = JobDeduplicator.canonicalize_url(
            job.source_url
        )

        job_id = JobDeduplicator.url_fingerprint(
            canonical_url
        )[:16]

        payload = asdict(job)

        payload["id"] = job_id
        payload["canonical_url"] = canonical_url
        payload["content_hash"] = (
            JobDeduplicator.content_fingerprint(job)
        )

        payload["scraped_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        path = self.processed_dir / f"{job_id}.json"

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return path
