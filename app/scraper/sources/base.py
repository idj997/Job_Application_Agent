from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class JobReference:
    source: str
    url: str

    external_id: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None


@dataclass
class RawJobPage:
    url: str
    html: str
    source: str
    status_code: int = 200


@dataclass
class RawJobPayload:
    source: str
    source_url: str
    data: object
    content_type: str = "application/json"


@dataclass
class ExtractedJob:
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    date_posted: Optional[str] = None
    salary: Optional[str] = None

    source: Optional[str] = None
    source_url: Optional[str] = None

    external_id: Optional[str] = None
    raw_description: Optional[str] = None
    metadata: Optional[dict] = None


class JobSource(ABC):
    """
    Base interface for all job sources.

    A source is responsible for discovering (where supported), fetching,
    and normalizing jobs. HTML-only sources can retain their existing
    ``RawJobPage`` fetch contract.
    """

    name: str

    def discover_jobs(
        self,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobReference]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support job discovery"
        )

    @abstractmethod
    def fetch(self, url: str) -> RawJobPage | RawJobPayload:
        raise NotImplementedError

    @abstractmethod
    def extract(self, raw: RawJobPage | RawJobPayload) -> ExtractedJob:
        raise NotImplementedError
