"""Reviewed selection plans are validated before models, network or ledger writes."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.applications.cv_evidence import build_cv_sources
from app.applications.jobs import prepare_job
from app.applications.tracking import job_key
from scripts import apply_jobs


MASTER_CV = "\n".join([
    "Alex Example", "alex@example.test", "Python engineer building reliable data systems.",
    "Example Ltd | Software Engineer | 2022–2025", "Built Python and SQL data pipelines.",
])
PRIVATE_CANARY = "PRIVATE_PLAN_CONTENT_MUST_NOT_APPEAR"


def forbidden(*args, **kwargs):
    raise AssertionError("No network, model construction/inference or unexpected write is permitted")


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
    for name in ("CV_CRITIC_MODEL", "CV_CRITIC_NUM_CTX", "CV_CRITIC_NUM_PREDICT",
                 "CV_CRITIC_TIMEOUT_SECONDS", "OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT",
                 "OLLAMA_TIMEOUT_SECONDS", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace(tmp_path):
    from app.applications.cv_selection import prepared_job_fingerprint, source_catalog_fingerprint

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


def command(workspace, *extra, with_plan=True):
    options = ["--selection-plan", str(workspace.plan)] if with_plan else []
    return ["--output", str(workspace.output), "prepare", "--cv", str(workspace.cv),
            "--job", str(workspace.job), "--limit", "1", "--no-enrich", *options, *extra]


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
@pytest.mark.parametrize("dry_run", [False, True])
def test_selection_scope_guards_precede_cv_models_and_writes(tmp_path, monkeypatch, options, dry_run):
    monkeypatch.setattr(apply_jobs, "load_profile", forbidden)
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    extra = ["--dry-run"] if dry_run else []
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["--output", str(tmp_path / "untouched"), "prepare",
                         "--selection-plan", "not-read.json", *options, *extra])
    assert error.value.code == 2
    assert not (tmp_path / "untouched").exists()


@pytest.mark.parametrize("problem", ["missing", "malformed", "schema", "oversized", "encoding"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_plan_file_fails_safely_before_models_and_writes(workspace, monkeypatch, capsys, problem, dry_run):
    monkeypatch.setattr(apply_jobs, "DeferredGenerationProvider", forbidden)
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    if problem == "missing":
        workspace.plan.unlink()
    elif problem == "malformed":
        workspace.plan.write_text("{" + PRIVATE_CANARY, encoding="utf-8")
    elif problem == "schema":
        workspace.plan.write_text(json.dumps({"sections": PRIVATE_CANARY}), encoding="utf-8")
    elif problem == "oversized":
        workspace.plan.write_text(PRIVATE_CANARY + " " * apply_jobs.MAX_SELECTION_PLAN_BYTES, encoding="utf-8")
    else:
        workspace.plan.write_bytes(b"\xff" + PRIVATE_CANARY.encode())
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(command(workspace, *(["--dry-run"] if dry_run else [])))
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "Could not load --selection-plan" in output.err
    assert PRIVATE_CANARY not in output.err + output.out
    assert not workspace.output.exists()


@pytest.mark.parametrize("problem", ["changed_cv", "changed_job", "unknown_id", "word_budget", "invalid_job"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_compile_failures_precede_models_and_ledger_even_on_dry_run(workspace, monkeypatch, capsys, problem, dry_run):
    monkeypatch.setattr(apply_jobs, "DeferredGenerationProvider", forbidden)
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    if problem == "changed_cv":
        workspace.cv.write_text(MASTER_CV + "\n" + PRIVATE_CANARY, encoding="utf-8")
    elif problem == "changed_job":
        payload = deepcopy(workspace.job_payload)
        payload["description"] += PRIVATE_CANARY
        workspace.job.write_text(json.dumps(payload), encoding="utf-8")
    elif problem == "invalid_job":
        workspace.job.write_text(json.dumps({"title": PRIVATE_CANARY}), encoding="utf-8")
    else:
        payload = deepcopy(workspace.payload)
        slot = payload["sections"][1]["slots"][0]
        if problem == "unknown_id":
            slot["evidence_ids"] = ["S9999"]
        else:
            slot["max_words"] = 71
        workspace.plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(command(workspace, *(["--dry-run"] if dry_run else [])))
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "Selection plan validation failed" in output.err
    assert PRIVATE_CANARY not in output.err + output.out
    assert not workspace.output.exists()


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("max_revisions", [0, 1, 2])
def test_dry_run_prints_bounded_summary_and_unchanged_call_budgets(
    workspace, monkeypatch, capsys, critic, max_revisions,
):
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    options = ["--critic-model", "ministral-3:8b"] if critic else ["--no-critic"]
    assert apply_jobs.main(command(workspace, "--dry-run", "--max-revisions", str(max_revisions), *options)) == 0
    output = capsys.readouterr().out
    # Fixed entries consume their exact source-joined length, not their cap.
    assert "Reviewed selection plan: 4 slots (2 fixed, 2 rewrite); 3 non-contact entries; maximum 76 words." in output
    assert "source/job fingerprints and bounds validated" in output
    assert "No network requests, model calls, or application writes" in output
    assert "Alex Example" not in output and "Built Python" not in output
    if critic:
        assert f"Maximum logical writer calls: {max_revisions + 1}; matching calls: 0" in output
        assert f"Maximum logical critic calls: {max_revisions + 1};" in output
    else:
        assert f"Maximum logical Ollama (local) calls: {2 * (max_revisions + 1)};" in output
    assert not workspace.output.exists()


@pytest.mark.parametrize("with_plan", [False, True])
def test_workflow_receives_validated_plan_and_exact_job_only_when_opted_in(workspace, monkeypatch, with_plan):
    from app.applications.cv_selection import SelectionPlan

    captured = {}
    matcher = object()
    monkeypatch.setattr(apply_jobs, "build_matcher", lambda name: matcher)
    if not with_plan:
        monkeypatch.setattr(apply_jobs, "_load_selection_plan", forbidden)

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
    assert apply_jobs.main(command(workspace, "--no-critic", with_plan=with_plan)) == 0
    assert captured["options"]["matcher"] is matcher
    if with_plan:
        assert isinstance(captured["options"]["selection_plan"], SelectionPlan)
        assert captured["options"]["selection_plan"].model_dump() == SelectionPlan.model_validate(workspace.payload).model_dump()
    else:
        assert "selection_plan" not in captured["options"]
    assert captured["job"] == workspace.prepared_job
    assert captured["cv_text"] == MASTER_CV
    assert not workspace.output.exists()
