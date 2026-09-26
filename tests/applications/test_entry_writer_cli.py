"""Entry-isolated writing is explicit, input-bound and correctly budgeted offline."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.applications.cv_evidence import build_cv_sources
from app.applications.cv_selection import prepared_job_fingerprint, source_catalog_fingerprint
from app.applications.jobs import prepare_job
from app.applications.tracking import job_key
from scripts import apply_jobs


MASTER_CV = "\n".join([
    "Alex Example", "alex@example.test", "Python engineer building reliable data systems.",
    "Example Ltd | Software Engineer | 2022–2025", "Built Python and SQL data pipelines.",
])
PRIVATE_CANARY = "PRIVATE_ENTRY_PLAN_CONTENT_MUST_NOT_APPEAR"


def forbidden(*args, **kwargs):
    raise AssertionError("No network, provider construction, model loading or unexpected write is permitted")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("app.applications.jobs._fetch_public_page", forbidden)
    monkeypatch.setattr(apply_jobs, "build_provider", forbidden)
    monkeypatch.setattr(apply_jobs, "build_local_role_provider", forbidden)
    monkeypatch.setenv("APPLICATION_PROVIDER", "ollama")
    monkeypatch.setenv("APPLICATION_MATCHER", "classifier")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OPENAI_MODEL", "offline-openai-writer")
    for name in ("CV_CRITIC_MODEL", "CV_CRITIC_NUM_CTX", "CV_CRITIC_NUM_PREDICT",
                 "CV_CRITIC_TIMEOUT_SECONDS", "OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT",
                 "OLLAMA_TIMEOUT_SECONDS", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace(tmp_path):
    cv = tmp_path / "master.txt"
    cv.write_text(MASTER_CV, encoding="utf-8")
    job_payload = {
        "title": "Data Engineer", "company": "Example Ltd", "source": "manual",
        "source_url": "https://employer.example/jobs/one",
        "description": "Python is required. " + "Maintain useful systems and support the reporting team. " * 8,
        "metadata": {"description_type": "full"},
    }
    job = tmp_path / "job.json"
    job.write_text(json.dumps(job_payload), encoding="utf-8")
    prepared_job = prepare_job(job_payload, fetch_full=False)
    payload = {
        "source_catalog_sha256": source_catalog_fingerprint(build_cv_sources(MASTER_CV)),
        "job_sha256": prepared_job_fingerprint(prepared_job),
        "sections": [
            {"heading": "Contact", "slots": [
                {"slot_id": "contact", "evidence_ids": ["S0001", "S0002"],
                 "mode": "fixed", "style": "paragraph", "max_words": 10},
            ]},
            {"heading": "Professional Summary", "slots": [
                {"slot_id": "summary", "evidence_ids": ["S0003"],
                 "mode": "rewrite", "style": "paragraph", "max_words": 40},
            ]},
            {"heading": "Experience", "slots": [
                {"slot_id": "role", "evidence_ids": ["S0004"],
                 "mode": "fixed", "style": "paragraph", "max_words": 15},
                {"slot_id": "achievement", "evidence_ids": ["S0005"],
                 "mode": "rewrite", "style": "bullet", "max_words": 25},
            ]},
        ],
        "roles": [{"heading_slot_id": "role", "achievement_slot_ids": ["achievement"], "primary": True}],
        "project_heading_slot_ids": [],
    }
    plan = tmp_path / "selection.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    return SimpleNamespace(cv=cv, job=job, job_payload=job_payload, prepared_job=prepared_job,
                           plan=plan, payload=payload, output=tmp_path / "applications")


def command(workspace, *extra, entry_by_entry=True):
    options = ["--entry-by-entry"] if entry_by_entry else []
    return ["--output", str(workspace.output), "prepare", "--cv", str(workspace.cv),
            "--job", str(workspace.job), "--limit", "1", "--no-enrich",
            "--selection-plan", str(workspace.plan), "--max-revisions", "0", *options, *extra]


@pytest.mark.parametrize("options", [
    [], ["--max-revisions", "0"], ["--selection-plan", "not-read.json"],
    ["--selection-plan", "not-read.json", "--max-revisions", "1"],
    ["--selection-plan", "not-read.json", "--max-revisions", "2"],
])
@pytest.mark.parametrize("dry_run", [False, True])
def test_entry_flag_requires_plan_and_zero_revisions_before_loading(tmp_path, monkeypatch, capsys, options, dry_run):
    monkeypatch.setattr(apply_jobs, "load_profile", forbidden)
    monkeypatch.setattr(apply_jobs, "DeferredGenerationProvider", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["--output", str(tmp_path / "untouched"), "prepare", "--entry-by-entry",
                         *options, *(["--dry-run"] if dry_run else [])])
    assert error.value.code == 2
    assert "--entry-by-entry requires --selection-plan and --max-revisions 0" in capsys.readouterr().err
    assert not (tmp_path / "untouched").exists()


@pytest.mark.parametrize("options", [
    [], ["--job", "one.json", "--no-enrich"],
    ["--job", "one.json", "--limit", "1"],
    ["--job", "one.json", "--job", "two.json", "--limit", "1", "--no-enrich"],
    ["--job", "one.json", "--limit", "2", "--no-enrich"],
    ["--job", "one.json", "--limit", "1", "--no-enrich", "--url", "https://example.test/job"],
    ["--job", "one.json", "--limit", "1", "--no-enrich", "--fetch"],
    ["--job", "one.json", "--limit", "1", "--no-enrich", "--description-file", "description.txt"],
    ["--job", "one.json", "--limit", "1", "--no-enrich", "--match-only"],
])
def test_entry_mode_preserves_selection_plan_scope_guards(monkeypatch, options):
    monkeypatch.setattr(apply_jobs, "load_profile", forbidden)
    monkeypatch.setattr(apply_jobs, "DeferredGenerationProvider", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["prepare", "--entry-by-entry", "--selection-plan", "not-read.json",
                         "--max-revisions", "0", *options, "--dry-run"])
    assert error.value.code == 2


@pytest.mark.parametrize("rewrite_slots", [0, 1, 2])
@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("matcher", ["classifier", "llm"])
@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_dry_run_counts_slot_retries_and_full_audit_separately(
    workspace, monkeypatch, capsys, rewrite_slots, critic, matcher, provider,
):
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    payload = deepcopy(workspace.payload)
    if rewrite_slots < 2:
        payload["sections"][1]["slots"][0]["mode"] = "fixed"
    if rewrite_slots == 0:
        payload["sections"][2]["slots"][1]["mode"] = "fixed"
    workspace.plan.write_text(json.dumps(payload), encoding="utf-8")
    options = ["--critic-model", "ministral-3:8b"] if critic else ["--no-critic"]
    assert apply_jobs.main(command(workspace, "--dry-run", "--provider", provider,
                                   "--matcher", matcher, *options)) == 0
    output = capsys.readouterr().out
    writer_calls = 2 * rewrite_slots
    matching_calls = int(matcher == "llm")
    name = "Ollama (local)" if provider == "ollama" else "OpenAI"
    assert f"Maximum logical writer calls: {writer_calls}; matching calls: {matching_calls}" in output
    if critic:
        assert f"Maximum logical writer provider calls ({name}): {writer_calls + matching_calls};" in output
        assert "Maximum logical critic calls: 1;" in output
        assert "Maximum logical writer self-audit calls:" not in output
    else:
        assert f"Maximum logical {name} calls: {writer_calls + matching_calls + 1};" in output
        assert "Maximum logical writer self-audit calls: 1;" in output
        assert "Maximum logical critic calls:" not in output
    assert "writer sees only the source lines assigned to each rewrite slot" in output
    assert "full source catalog is retained for audit" in output
    assert "at most 2 attempts per rewrite slot" in output
    assert "one local validation retry" in output
    assert "no critic-driven revisions" in output
    assert "No network requests, model calls, or application writes" in output
    assert "Alex Example" not in output and "Built Python" not in output
    assert not workspace.output.exists()


@pytest.mark.parametrize("problem", ["changed_cv", "changed_job", "unknown_source", "bad_json"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_entry_mode_validates_source_job_and_plan_before_providers(
    workspace, monkeypatch, capsys, problem, dry_run,
):
    monkeypatch.setattr(apply_jobs, "DeferredGenerationProvider", forbidden)
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    if problem == "changed_cv":
        workspace.cv.write_text(MASTER_CV + "\n" + PRIVATE_CANARY, encoding="utf-8")
    elif problem == "changed_job":
        payload = deepcopy(workspace.job_payload)
        payload["description"] += PRIVATE_CANARY
        workspace.job.write_text(json.dumps(payload), encoding="utf-8")
    elif problem == "unknown_source":
        payload = deepcopy(workspace.payload)
        payload["sections"][1]["slots"][0]["evidence_ids"] = ["S9999"]
        workspace.plan.write_text(json.dumps(payload), encoding="utf-8")
    else:
        workspace.plan.write_text("{" + PRIVATE_CANARY, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(command(workspace, *(["--dry-run"] if dry_run else [])))
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "selection" in output.err.lower()
    assert PRIVATE_CANARY not in output.out + output.err
    assert not workspace.output.exists()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("critic", [False, True])
def test_workflow_only_receives_entry_mode_when_explicitly_enabled(workspace, monkeypatch, enabled, critic):
    captured = {}
    matcher = object()
    monkeypatch.setattr(apply_jobs, "build_matcher", lambda name: matcher)

    class Ledger:
        def __init__(self, path):
            captured["ledger_path"] = path

        def get(self, key):
            return None

    class Workflow:
        def __init__(self, provider, ledger, output, **options):
            captured["options"] = options

        def prepare(self, job, profile, cv_text, **options):
            captured["job"] = job
            captured["cv_text"] = cv_text
            return {"job_key": job_key(job), "status": "review", "reasons": [], "artifacts": {}}

    monkeypatch.setattr(apply_jobs, "ApplicationLedger", Ledger)
    monkeypatch.setattr("app.applications.workflow.ApplicationWorkflow", Workflow)
    monkeypatch.setattr(apply_jobs, "write_dashboard", lambda *args: None)
    options = ["--critic-model", "ministral-3:8b"] if critic else ["--no-critic"]
    assert apply_jobs.main(command(workspace, *options, entry_by_entry=enabled)) == 0
    assert captured["options"]["matcher"] is matcher
    expected_plan = deepcopy(workspace.payload)
    for section in expected_plan["sections"]:
        for slot in section["slots"]:
            slot["required_spans"] = []
    assert captured["options"]["selection_plan"].model_dump() == expected_plan
    assert captured["options"]["max_revisions"] == 0
    assert ("critic_provider" in captured["options"]) is critic
    if enabled:
        assert captured["options"]["entry_by_entry"] is True
    else:
        assert "entry_by_entry" not in captured["options"]
    assert captured["job"] == workspace.prepared_job
    assert captured["cv_text"] == MASTER_CV
    assert not workspace.output.exists()
