from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.scraper.role_catalog import RoleFamily
from app.scraper.sources.base import JobSource
from app.services.job_collection_service import CollectionSummary, JobCollectionService


@dataclass
class RoleQuerySummary:
    role_family: str
    query: str
    collection: CollectionSummary


@dataclass
class RoleCorpusSummary:
    queries: list[RoleQuerySummary] = field(default_factory=list)

    @property
    def discovered(self) -> int:
        return sum(item.collection.discovered for item in self.queries)

    @property
    def saved(self) -> int:
        return sum(item.collection.saved for item in self.queries)

    @property
    def duplicates(self) -> int:
        return sum(item.collection.duplicates for item in self.queries)

    @property
    def rejected(self) -> int:
        return sum(item.collection.rejected for item in self.queries)

    @property
    def failed(self) -> int:
        return sum(item.collection.failed for item in self.queries)


class RoleCorpusCollectionService:
    def __init__(self, collection_service: JobCollectionService):
        self.collection_service = collection_service

    def collect(
        self,
        sources: Iterable[JobSource],
        role_families: Iterable[RoleFamily],
        *,
        location: str | None = None,
        limit_per_query: int = 5,
    ) -> RoleCorpusSummary:
        if limit_per_query < 1:
            raise ValueError("limit_per_query must be at least 1")

        result = RoleCorpusSummary()
        families = list(role_families)
        for source in sources:
            source_failed = False
            for family in families:
                for query in family.queries:
                    if source_failed:
                        break
                    summary = self.collection_service.collect(
                        source=source,
                        query=query,
                        location=location,
                        limit=limit_per_query,
                        collection_metadata={
                            "role_family": family.name,
                            "role_family_name": family.display_name,
                            "query": query,
                        },
                    )
                    result.queries.append(
                        RoleQuerySummary(
                            role_family=family.name,
                            query=query,
                            collection=summary,
                        )
                    )
                    # A discovery-level error is usually persistent (credentials,
                    # configuration, or an unavailable API). Avoid hammering it for
                    # every remaining role query while allowing other sources to run.
                    source_failed = summary.source_error is not None
                if source_failed:
                    break
        return result
