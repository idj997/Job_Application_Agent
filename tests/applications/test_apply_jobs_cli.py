"""Exercise CLI orchestration with real local ledgers and no external calls."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.applications import browser, jobs
from app.applications.tracking import ApplicationLedger, job_key
from app.services.job_collection_service import CollectionSummary
from scripts import apply_jobs


MASTER_CV = "Alex Example\nalex@example.test\nBuilt Python and SQL data pipelines."


def job_record(number=1):
    return {
        "title": f"Data Engineer {number}", "company": "Example Ltd", "source": "adzuna",
        "source_url": f"https://employer.example/jobs/{number}",
        "description": "Build Python data pipelines and maintain SQL reporting.",
        "metadata": {"description_type": "summary"},
    }


def write_job(directory, number=1, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"job-{number}.json"
    record = job_record(number)
    record.update(changes)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path):
    cv = tmp_path / "master.txt"
    cv.write_text(MASTER_CV, encoding="utf-8")
    return SimpleNamespace(root=tmp_path, cv=cv, output=tmp_path / "applications", jobs=tmp_path / "jobs")


@pytest.fixture(autouse=True)
def external_calls_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This CLI test must not contact external services")

    instances = []

    class OfflineProvider:
        def __init__(self, model=None):
            self.model = model or "offline-test-model"
            self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            instances.append(self)

        generate_structured = forbidden

    monkeypatch.setattr("app.providers.openai.OpenAIProvider", OfflineProvider)
    monkeypatch.setattr(jobs, "_fetch_public_page", forbidden)
    monkeypatch.setattr(browser, "assist_application", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    return instances


def prepare_command(workspace, *options):
    return ["--output", str(workspace.output), "prepare", "--cv", str(workspace.cv),
            "--jobs-dir", str(workspace.jobs), "--no-enrich", *options]


def test_dry_run_has_no_provider_browser_network_or_artifacts(workspace, monkeypatch, capsys):
    write_job(workspace.jobs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not construct a provider or ledger")
    monkeypatch.setattr("app.providers.openai.OpenAIProvider", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)

    assert apply_jobs.main(prepare_command(workspace, "--dry-run")) == 0

    output = capsys.readouterr().out
    assert "1 selected" in output
    assert "No network requests, model calls, or application writes" in output
    assert not workspace.output.exists()


def test_profile_cv_path_is_relative_to_json_profile(workspace):
    config = workspace.root / "config"
    config.mkdir()
    profile_path = config / "candidate.json"
    profile_path.write_text(json.dumps({"cv_path": "../master.txt", "name": "Alex Example"}))

    profile, cv_path = apply_jobs.load_profile(profile_path, None)

    assert profile.name == "Alex Example"
    assert cv_path == workspace.cv.resolve()
    assert apply_jobs.main(["--output", str(workspace.output), "prepare", "--profile", str(profile_path), "--dry-run"]) == 0
    assert not workspace.output.exists()


def test_explicit_cv_overrides_profile_cv_path(workspace):
    profile_path = workspace.root / "candidate.json"
    profile_path.write_text(json.dumps({"cv_path": "missing-master.pdf"}))
    assert apply_jobs.load_profile(profile_path, workspace.cv)[1] == workspace.cv.resolve()


@pytest.mark.parametrize("targets", [[], ["--job", "one.json", "--job", "two.json"], ["--job", "one.json", "--url", "https://example.com/job"], ["--fetch"]])
def test_description_file_requires_exactly_one_explicit_target(workspace, targets, capsys):
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(prepare_command(workspace, *targets, "--description-file", "description.txt", "--dry-run"))
    assert error.value.code == 2
    assert "exactly one" in capsys.readouterr().err
    assert not workspace.output.exists()


def test_description_file_is_passed_only_to_selected_job(workspace, monkeypatch):
    path = write_job(workspace.jobs)
    description = workspace.root / "description.txt"
    description.write_text("Full description fixture")
    seen = []
    real_prepare = jobs.prepare_job
    def capture(record, **kwargs):
        seen.append((record, kwargs))
        return real_prepare(record, fetch_full=False)
    monkeypatch.setattr(jobs, "prepare_job", capture)

    assert apply_jobs.main(prepare_command(workspace, "--job", str(path), "--description-file", str(description))) == 0

    assert len(seen) == 1
    assert seen[0][1] == {"full_description_file": description, "fetch_full": False}


def test_explicit_url_excludes_unselected_stored_jobs(workspace, monkeypatch):
    write_job(workspace.jobs, 1)
    fetched = []
    def fetch(url):
        fetched.append(url)
        return job_record(2)
    monkeypatch.setattr(jobs, "fetch_job_url", fetch)

    assert apply_jobs.main(prepare_command(workspace, "--url", "https://employer.example/jobs/2")) == 0

    records = ApplicationLedger(workspace.output / "applications.db").list()
    assert [record["title"] for record in records] == ["Data Engineer 2"]
    assert fetched == ["https://employer.example/jobs/2"]


def test_dry_run_url_count_matches_real_url_selection(workspace, capsys):
    write_job(workspace.jobs, 1)
    write_job(workspace.jobs, 2)
    assert apply_jobs.main(prepare_command(workspace, "--url", "https://employer.example/jobs/3", "--dry-run")) == 0
    assert "Jobs: 1 selected" in capsys.readouterr().out


def test_limit_is_applied_before_fetching_direct_urls(workspace, monkeypatch):
    fetched = []
    def fetch(url):
        fetched.append(url)
        return job_record(len(fetched))
    monkeypatch.setattr(jobs, "fetch_job_url", fetch)
    assert apply_jobs.main(prepare_command(
        workspace, "--url", "https://employer.example/jobs/1", "--url", "https://employer.example/jobs/2", "--limit", "1",
    )) == 0
    assert fetched == ["https://employer.example/jobs/1"]


def test_explicit_job_file_excludes_other_stored_jobs(workspace):
    selected = write_job(workspace.jobs, 1)
    write_job(workspace.jobs, 2)
    assert apply_jobs.main(prepare_command(workspace, "--job", str(selected))) == 0
    records = ApplicationLedger(workspace.output / "applications.db").list()
    assert [record["title"] for record in records] == ["Data Engineer 1"]


def test_fetch_reuses_existing_collection_service(workspace, monkeypatch, capsys):
    calls = []
    sources = []
    def source(name, **kwargs):
        sources.append((name, kwargs))
        return SimpleNamespace(name=name)
    class Collector:
        def __init__(self, storage, cleaner):
            self.storage = storage

        def collect(self, source, **kwargs):
            calls.append((source.name, kwargs))
            write_job(self.storage.processed_dir, len(calls))
            return CollectionSummary(source=source.name, saved=1)
    monkeypatch.setattr("scripts.scrape_jobs.build_configured_source", source)
    monkeypatch.setattr("app.services.job_collection_service.JobCollectionService", Collector)

    assert apply_jobs.main(prepare_command(workspace, "--fetch", "--sources", "lever", "greenhouse", "--query", "Data Engineer", "--limit", "2", "--company-slug", "example", "--board-token", "example")) == 0

    assert [item[0] for item in sources] == ["lever", "greenhouse"]
    assert calls == [(name, {"query": "Data Engineer", "location": "United Kingdom", "limit": 2}) for name in ["lever", "greenhouse"]]
    assert len(ApplicationLedger(workspace.output / "applications.db").list()) == 2
    assert "lever: 1 saved" in capsys.readouterr().out


def saved_record(workspace, *, status="ready", key="abc123" + "0" * 58):
    workspace.output.mkdir(parents=True, exist_ok=True)
    pdf = workspace.output / "cv.pdf"
    pdf.write_bytes(b"offline PDF fixture")
    record = {
        "job_key": key, "status": status, "title": "Data Engineer", "company": "Example Ltd",
        "application_url": "https://jobs.lever.co/example/123",
        "decision": {"score": 90}, "cv_score": 85, "reasons": ["Matches the CV"],
        "artifacts": {"pdf": str(pdf)},
    }
    ledger = ApplicationLedger(workspace.output / "applications.db")
    ledger.save(key, record)
    return ledger, key


def test_list_and_show_read_saved_scores_and_details(workspace, capsys):
    ledger, key = saved_record(workspace)
    assert apply_jobs.main(["--output", str(workspace.output), "list"]) == 0
    listing = capsys.readouterr().out
    assert "READY" in listing and "Match: 90" in listing and "CV quality: 85" in listing
    assert apply_jobs.main(["--output", str(workspace.output), "show", key[:6]]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["job_key"] == key
    assert shown["artifacts"] == ledger.get(key)["artifacts"]


def test_show_requires_unique_record_prefix(workspace, capsys):
    saved_record(workspace, key="abc123" + "0" * 58)
    saved_record(workspace, key="abc123" + "1" * 58)
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["--output", str(workspace.output), "show", "abc123"])
    assert error.value.code == 2
    assert "uniquely match" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["review", "skipped", "failed", "opened"])
def test_only_ready_applications_can_open(workspace, status, capsys):
    saved_record(workspace, status=status)
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["--output", str(workspace.output), "open", "abc123"])
    assert error.value.code == 2
    assert "Only a ready" in capsys.readouterr().err


def test_open_persists_attempt_before_calling_browser(workspace, monkeypatch):
    ledger, key = saved_record(workspace)
    def assist(record, **kwargs):
        persisted = ApplicationLedger(ledger.path).get(key)
        assert persisted["submission_attempted"] is True
        assert persisted["status"] == "opened"
        assert kwargs["submit"] is False
        return {"status": "manual", "detail": "Applicant closed the window", "confirmation": None}
    monkeypatch.setattr(browser, "assist_application", assist)
    assert apply_jobs.main(["--output", str(workspace.output), "open", key[:6]]) == 0
    assert ledger.get(key)["status"] == "opened"


def test_browser_failure_keeps_persisted_attempt_for_resolution(workspace, monkeypatch):
    ledger, key = saved_record(workspace)
    def failed(*args, **kwargs):
        raise RuntimeError("Browser stopped unexpectedly")
    monkeypatch.setattr(browser, "assist_application", failed)
    with pytest.raises(SystemExit):
        apply_jobs.main(["--output", str(workspace.output), "open", key[:6]])
    assert ledger.get(key)["submission_attempted"] is True
    assert ledger.get(key)["status"] == "opened"


def test_confirmed_browser_submission_is_saved(workspace, monkeypatch):
    ledger, key = saved_record(workspace)
    def submitted(record, **kwargs):
        assert kwargs["submit"] is True
        return {"status": "submitted", "detail": "Application received", "confirmation": "Receipt A123"}
    monkeypatch.setattr(browser, "assist_application", submitted)
    assert apply_jobs.main(["--output", str(workspace.output), "open", key[:6], "--submit"]) == 0
    assert ledger.get(key)["status"] == "submitted"
    assert ledger.get(key)["confirmation"] == "Receipt A123"


def test_stale_ready_record_cannot_open_browser_twice(workspace, monkeypatch):
    ledger, key = saved_record(workspace)
    stale = ledger.get(key)
    monkeypatch.setattr(apply_jobs, "_record", lambda ledger, prefix: deepcopy(stale))
    opened = []
    def assist(record, **kwargs):
        opened.append(record)
        return {"status": "manual", "detail": "Closed without confirmation", "confirmation": None}
    monkeypatch.setattr(browser, "assist_application", assist)
    assert apply_jobs.main(["--output", str(workspace.output), "open", key[:6]]) == 0
    try:
        apply_jobs.main(["--output", str(workspace.output), "open", key[:6]])
    except SystemExit as error:
        assert error.code == 2
    assert len(opened) == 1


def test_mark_submitted_records_manual_confirmation_and_blocks_reset(workspace):
    ledger, key = saved_record(workspace)
    assert apply_jobs.main(["--output", str(workspace.output), "mark-submitted", key[:6], "--confirmation", "Email receipt confirmed"]) == 0
    assert ledger.get(key)["confirmation"] == "Email receipt confirmed"
    with pytest.raises(SystemExit):
        apply_jobs.main(["--output", str(workspace.output), "resolve-not-submitted", key[:6], "--note", "Checked website"])
    assert ledger.get(key)["status"] == "submitted"


def test_resolve_not_submitted_retains_checked_history(workspace):
    ledger, key = saved_record(workspace, status="opened")
    assert apply_jobs.main(["--output", str(workspace.output), "resolve-not-submitted", key[:6], "--note", "Employer portal confirms draft only"]) == 0
    assert ledger.get(key)["status"] == "ready"
    assert ledger.get(key)["resolution_history"][-1]["note"] == "Employer portal confirms draft only"


def test_cached_summary_uses_no_api_calls_and_keeps_saved_timestamp(workspace, capsys, external_calls_forbidden):
    write_job(workspace.jobs)
    command = prepare_command(workspace)
    assert apply_jobs.main(command) == 0
    ledger = ApplicationLedger(workspace.output / "applications.db")
    before = ledger.list()[0]
    capsys.readouterr()

    assert apply_jobs.main(command) == 0

    assert "saved result" in capsys.readouterr().out
    assert ledger.get(before["job_key"])["updated_at"] == before["updated_at"]
    assert all(provider.usage["requests"] == 0 for provider in external_calls_forbidden)
    assert not list(workspace.output.rglob("*.pdf"))


def test_submitted_record_stays_cached_even_with_retry(workspace, capsys):
    path = write_job(workspace.jobs)
    record = json.loads(path.read_text())
    ledger, key = saved_record(workspace, key=job_key(record))
    ledger.mark_submitted(key, "Previously verified receipt")

    assert apply_jobs.main(prepare_command(workspace, "--retry")) == 0

    assert ledger.get(key)["status"] == "submitted"
    assert "saved result" in capsys.readouterr().out


def test_invalid_job_does_not_abort_remaining_batch(workspace, capsys):
    invalid = write_job(workspace.jobs, 1, title=None)
    valid = write_job(workspace.jobs, 2)
    apply_jobs.main(prepare_command(workspace, "--job", str(invalid), "--job", str(valid)))
    records = ApplicationLedger(workspace.output / "applications.db").list()
    assert [record["title"] for record in records] == ["Data Engineer 2"]


def test_no_jobs_does_not_create_application_ledger(workspace, capsys):
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(prepare_command(workspace))
    assert error.value.code == 2
    assert "No valid jobs" in capsys.readouterr().err
    assert not workspace.output.exists()
