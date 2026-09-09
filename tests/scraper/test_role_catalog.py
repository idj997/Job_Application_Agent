import json

import pytest

from app.scraper.role_catalog import RoleCatalog, RoleCatalogError


def test_default_catalog_covers_diverse_software_and_hardware_roles():
    catalog = RoleCatalog.load()

    assert {
        "data_engineering",
        "data_science",
        "ai_engineering",
        "machine_learning",
        "generative_ai",
        "cpu_engineering",
        "gpu_engineering",
    }.issubset(catalog.names)
    assert all(family.queries for family in catalog.select())
    assert catalog.match_title("Senior Data Engineering Manager") == "data_engineering"
    assert catalog.match_title("Senior Data Scientist") == "data_science"
    assert catalog.match_title("Staff GPU Compiler Engineer") == "gpu_engineering"


def test_catalog_rejects_unknown_roles_and_invalid_query_lists(tmp_path):
    catalog = RoleCatalog.load()
    with pytest.raises(RoleCatalogError, match="unknown role family"):
        catalog.select(["quantum_alchemy"])

    path = tmp_path / "roles.json"
    path.write_text(
        json.dumps(
            {
                "role_families": {
                    "broken": {"display_name": "Broken", "queries": []}
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RoleCatalogError, match="at least one query"):
        RoleCatalog.load(path)
