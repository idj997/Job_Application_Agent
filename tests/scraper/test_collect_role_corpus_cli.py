import pytest

from scripts.collect_role_corpus import main


def test_dry_run_reports_bounded_collection_plan(capsys):
    main(
        [
            "--sources",
            "adzuna",
            "--roles",
            "data_engineering",
            "gpu_engineering",
            "--limit-per-query",
            "3",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "Role families:       2" in output
    assert "Distinct queries:    8" in output
    assert "Query/source pairs:  8" in output
    assert "GPU Engineering" in output


def test_safety_cap_blocks_an_accidentally_large_run():
    with pytest.raises(SystemExit):
        main(
            [
                "--sources",
                "adzuna",
                "arbeitnow",
                "--all-roles",
                "--max-query-source-pairs",
                "10",
                "--dry-run",
            ]
        )

