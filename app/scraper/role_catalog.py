from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLE_CATALOG = Path("config/job_role_queries.json")


class RoleCatalogError(ValueError):
    """Raised when a role-query catalog is missing or malformed."""


@dataclass(frozen=True)
class RoleFamily:
    name: str
    display_name: str
    queries: tuple[str, ...]


class RoleCatalog:
    def __init__(self, families: Iterable[RoleFamily]):
        self._families = {family.name: family for family in families}
        if not self._families:
            raise RoleCatalogError("role catalog must contain at least one family")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._families)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ROLE_CATALOG) -> RoleCatalog:
        catalog_path = Path(path)
        if not catalog_path.exists():
            raise RoleCatalogError(f"role catalog does not exist: {catalog_path}")
        try:
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RoleCatalogError(
                f"role catalog contains invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise RoleCatalogError("role catalog root must be an object")
        raw_families = payload.get("role_families")
        if not isinstance(raw_families, dict):
            raise RoleCatalogError("role catalog must contain role_families")

        families = [
            cls._parse_family(name, value)
            for name, value in raw_families.items()
        ]
        return cls(families)

    @staticmethod
    def _parse_family(name: Any, payload: Any) -> RoleFamily:
        if not isinstance(name, str) or not name.strip():
            raise RoleCatalogError("role family names must be non-empty strings")
        if not isinstance(payload, dict):
            raise RoleCatalogError(f"role family {name!r} must be an object")
        display_name = payload.get("display_name")
        queries = payload.get("queries")
        if not isinstance(display_name, str) or not display_name.strip():
            raise RoleCatalogError(
                f"role family {name!r} requires a display_name"
            )
        if not isinstance(queries, list) or not queries:
            raise RoleCatalogError(
                f"role family {name!r} requires at least one query"
            )
        cleaned_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                raise RoleCatalogError(
                    f"role family {name!r} contains an invalid query"
                )
            cleaned = query.strip()
            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                cleaned_queries.append(cleaned)
        return RoleFamily(
            name=name.strip(),
            display_name=display_name.strip(),
            queries=tuple(cleaned_queries),
        )

    def select(self, names: Iterable[str] | None = None) -> list[RoleFamily]:
        if names is None:
            return list(self._families.values())

        selected: list[RoleFamily] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            try:
                family = self._families[name]
            except KeyError as exc:
                available = ", ".join(self.names)
                raise RoleCatalogError(
                    f"unknown role family {name!r}; available: {available}"
                ) from exc
            selected.append(family)
            seen.add(name)
        if not selected:
            raise RoleCatalogError("select at least one role family")
        return selected

    def match_title(self, title: str | None) -> str | None:
        if not title or not title.strip():
            return None
        normalized_title = self._normalized_title(title)
        matches: list[tuple[int, str]] = []
        for family in self._families.values():
            for query in family.queries:
                normalized_query = self._normalized_title(query)
                if normalized_query in normalized_title:
                    matches.append((len(normalized_query), family.name))
        if not matches:
            return None
        return max(matches)[1]

    @staticmethod
    def _normalized_title(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9+]+", " ", value.casefold())
        normalized = re.sub(r"\bengineering\b", "engineer", normalized)
        normalized = re.sub(r"\bscientist\b", "science", normalized)
        return re.sub(r"\s+", " ", normalized).strip()
