from __future__ import annotations

from app.scraper.sources.adzuna import AdzunaSource
from app.scraper.sources.arbeitnow import ArbeitnowSource
from app.scraper.sources.base import JobSource
from app.scraper.sources.dwp import DwpSource
from app.scraper.sources.generic_jsonld import GenericJsonLdSource
from app.scraper.sources.greenhouse import GreenhouseSource
from app.scraper.sources.lever import LeverSource
from app.scraper.sources.remotive import RemotiveSource
from app.scraper.sources.the_muse import TheMuseSource


SOURCE_REGISTRY: dict[str, type[JobSource]] = {
    "adzuna": AdzunaSource,
    "arbeitnow": ArbeitnowSource,
    "dwp": DwpSource,
    "generic_jsonld": GenericJsonLdSource,
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "remotive": RemotiveSource,
    "the_muse": TheMuseSource,
}


def build_source(name: str, **kwargs: object) -> JobSource:
    try:
        source_class = SOURCE_REGISTRY[name.casefold()]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCE_REGISTRY))
        raise ValueError(
            f"Unknown job source {name!r}. Available sources: {available}"
        ) from exc
    return source_class(**kwargs)
