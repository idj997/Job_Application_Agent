from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.scraper.cleaner import JobCleaner
from app.scraper.exceptions import InvalidJobError, ScraperError
from app.scraper.quality import JobQualityValidator
from app.scraper.sources.base import JobReference, JobSource
from app.scraper.storage import JobStorage


logger = logging.getLogger(__name__)


@dataclass
class CollectionSummary:
    source: str
    discovered: int = 0
    saved: int = 0
    duplicates: int = 0
    failed: int = 0
    rejected: int = 0
    source_error: str | None = None


@dataclass
class MultiSourceCollectionSummary:
    summaries: list[CollectionSummary] = field(default_factory=list)

    @property
    def discovered(self) -> int:
        return sum(summary.discovered for summary in self.summaries)

    @property
    def saved(self) -> int:
        return sum(summary.saved for summary in self.summaries)

    @property
    def duplicates(self) -> int:
        return sum(summary.duplicates for summary in self.summaries)

    @property
    def failed(self) -> int:
        return sum(summary.failed for summary in self.summaries)

    @property
    def rejected(self) -> int:
        return sum(summary.rejected for summary in self.summaries)


class JobCollectionService:
    def __init__(
        self,
        storage: JobStorage,
        cleaner: JobCleaner,
        validator: JobQualityValidator | None = None,
    ):
        self.storage = storage
        self.cleaner = cleaner
        self.validator = validator or JobQualityValidator()

    def collect(
        self,
        source: JobSource,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
        collection_metadata: dict[str, object] | None = None,
    ) -> CollectionSummary:
        summary = CollectionSummary(source=source.name)

        try:
            references = source.discover_jobs(
                query=query,
                location=location,
                limit=limit,
            )
        except ScraperError as exc:
            summary.failed = 1
            summary.source_error = str(exc)
            logger.error("Could not start collection from %s: %s", source.name, exc)
            return summary
        except Exception as exc:
            summary.failed = 1
            summary.source_error = str(exc)
            logger.exception("Could not start collection from %s", source.name)
            return summary

        summary.discovered = len(references)

        for item in references:
            reference = self._reference(source, item)
            try:
                fetch_job = getattr(source, "fetch_job", None)
                raw = fetch_job(reference) if callable(fetch_job) else source.fetch(reference.url)
                job = source.extract(raw)
                job = self.cleaner.clean(job)
                if collection_metadata:
                    job.metadata = {
                        **(job.metadata or {}),
                        "collection": {
                            **collection_metadata,
                        },
                    }
                self.validator.validate(job)

                if self.storage.duplicate_kind(job):
                    summary.duplicates += 1
                    continue

                self.storage.save_raw(raw)
                self.storage.save_job(job)
                summary.saved += 1
            except InvalidJobError as exc:
                summary.rejected += 1
                logger.warning("Rejected job %s: %s", reference.url, exc)
            except Exception:
                summary.failed += 1
                logger.exception("Failed to collect job %s", reference.url)

        return summary

    def collect_many(
        self,
        sources: Iterable[JobSource],
        query: str | None = None,
        location: str | None = None,
        limit_per_source: int = 50,
    ) -> MultiSourceCollectionSummary:
        summaries = [
            self.collect(
                source=source,
                query=query,
                location=location,
                limit=limit_per_source,
            )
            for source in sources
        ]
        return MultiSourceCollectionSummary(summaries=summaries)

    @staticmethod
    def _reference(source: JobSource, item: JobReference | str) -> JobReference:
        if isinstance(item, JobReference):
            return item
        return JobReference(source=source.name, url=item)
