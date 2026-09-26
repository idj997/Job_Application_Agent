"""Opt-in critic orchestration stays local, lazy and independent of matching."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.applications.cv_evidence import ReferencedCV
from app.applications.models import MatchAssessment, RequirementMatch
from app.applications.tracking import ApplicationLedger
from scripts import apply_jobs


MASTER_CV = "Alex Example\nalex@example.test\nBuilt Python and SQL data pipelines."


def forbidden(*args, **kwargs):
    raise AssertionError("This path must not construct a client, contact a server or load weights")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
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
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "title": "Data Engineer", "company": "Example Ltd",
        "source_url": "https://employer.example/jobs/one", "source": "manual",
        "description": "Python is required. " + "Maintain useful systems and support the reporting team. " * 8,
        "metadata": {"description_type": "full"},
    }), encoding="utf-8")
    return SimpleNamespace(cv=cv, job=job, output=tmp_path / "applications")


def command(workspace, *extra):
    return ["--output", str(workspace.output), "prepare", "--cv", str(workspace.cv),
            "--job", str(workspace.job), "--limit", "1", "--no-enrich", *extra]


def install_matcher(monkeypatch, outcome="apply"):
    class Matcher:
        name = "classifier"
        fingerprint = {"version": "offline-critic-cli"}
        diagnostics = {}
        calls = 0

        def assess(self, *args):
            self.calls += 1
            return MatchAssessment(
                description_complete=True,
                requirements=[RequirementMatch(
                    requirement="Python", importance="IMPORTANT" if outcome == "skip" else "CORE",
                    status={"apply": "MET", "review": "UNKNOWN", "skip": "MISSING"}[outcome],
                    job_evidence="Python is required.",
                    cv_evidence=["Built Python and SQL data pipelines."] if outcome == "apply" else [],
                    explanation="Offline classifier fixture.",
                )], constraint_checks=[], recommended_verdict="APPLY", rationale="Offline match.", uncertainties=[],
            )
    matcher = Matcher()
    monkeypatch.setattr(apply_jobs, "build_matcher", lambda name: matcher)
    return matcher


def test_critic_is_disabled_by_default_and_explicit_flag_overrides_environment(monkeypatch):
    assert apply_jobs.selected_critic_model(None) is None
    monkeypatch.setenv("CV_CRITIC_MODEL", " gemma3:4b ")
    assert apply_jobs.selected_critic_model(None) == "gemma3:4b"
    assert apply_jobs.selected_critic_model(" another-model:4b ") == "another-model:4b"
    assert apply_jobs.selected_critic_model(None, disabled=True) is None
    with pytest.raises(ValueError, match="--no-critic"):
        apply_jobs.selected_critic_model(" ")


@pytest.mark.parametrize("command_name", ["prepare", "doctor"])
def test_critic_flag_and_disable_flag_are_mutually_exclusive(command_name):
    with pytest.raises(SystemExit) as error:
        apply_jobs.main([command_name, "--critic-model", "gemma3:4b", "--no-critic"])
    assert error.value.code == 2


@pytest.mark.parametrize("selection", ["flag", "environment"])
@pytest.mark.parametrize("outcome,match_only,expected", [
    ("apply", True, "matched"), ("review", False, "review"), ("skip", False, "skipped"),
])
def test_classifier_outcomes_do_not_construct_writer_or_critic(
    workspace, monkeypatch, capsys, selection, outcome, match_only, expected,
):
    matcher = install_matcher(monkeypatch, outcome)
    options = ["--critic-model", "missing-local-critic:4b"] if selection == "flag" else []
    if selection == "environment":
        monkeypatch.setenv("CV_CRITIC_MODEL", "missing-local-critic:4b")
    if match_only:
        options.append("--match-only")
    assert apply_jobs.main(command(workspace, *options)) == 0
    assert matcher.calls == 1
    record = ApplicationLedger(workspace.output / "applications.db").list()[0]
    assert record["status"] == expected
    assert not record["artifacts"]
    output = capsys.readouterr().out
    assert "missing-local-critic:4b" in output
    assert 'Ollama critic usage: {"requests": 0' in output


@pytest.mark.parametrize("provider", ["ollama", "openai"])
@pytest.mark.parametrize("matcher", ["classifier", "llm"])
@pytest.mark.parametrize("max_revisions", [0, 1, 2])
@pytest.mark.parametrize("match_only", [False, True])
def test_dry_run_separates_writer_matching_and_critic_calls_without_loading(
    workspace, monkeypatch, capsys, provider, matcher, max_revisions, match_only,
):
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    options = ["--provider", provider, "--matcher", matcher, "--critic-model", "gemma3:4b",
               "--max-revisions", str(max_revisions), "--dry-run"]
    if match_only:
        options.append("--match-only")
    assert apply_jobs.main(command(workspace, *options)) == 0
    output = capsys.readouterr().out
    drafts = 0 if match_only else max_revisions + 1
    matching = 1 if matcher == "llm" else 0
    assert f"Maximum logical writer calls: {drafts}; matching calls: {matching}" in output
    assert f"Maximum logical critic calls: {drafts}; replaces writer self-audits" in output
    assert "No network requests, model calls, or application writes" in output
    assert not workspace.output.exists()


def test_no_critic_restores_legacy_dry_run_budgets(workspace, monkeypatch, capsys):
    monkeypatch.setenv("CV_CRITIC_MODEL", "gemma3:4b")
    assert apply_jobs.main(command(workspace, "--no-critic", "--dry-run", "--max-revisions", "1")) == 0
    output = capsys.readouterr().out
    assert "Maximum logical Ollama (local) calls: 4;" in output
    assert "CV critic:" not in output
    assert "Maximum logical critic calls:" not in output


@pytest.mark.parametrize("selection", ["flag", "environment"])
def test_same_writer_critic_model_is_rejected_before_loading(workspace, monkeypatch, capsys, selection):
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    options = ["--critic-model", "qwen3:8b"] if selection == "flag" else []
    if selection == "environment":
        monkeypatch.setenv("CV_CRITIC_MODEL", "qwen3:8b")
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(command(workspace, *options, "--dry-run"))
    assert error.value.code == 2
    assert "critic model must differ from the writer model" in capsys.readouterr().err
    assert not workspace.output.exists()


def test_same_model_environment_can_be_disabled(workspace, monkeypatch):
    monkeypatch.setenv("CV_CRITIC_MODEL", "qwen3:8b")
    assert apply_jobs.main(command(workspace, "--no-critic", "--dry-run")) == 0


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_doctor_checks_critic_dependency_and_limits_without_clients(monkeypatch, capsys, provider):
    checked = []
    monkeypatch.setattr(apply_jobs.importlib.util, "find_spec", lambda name: checked.append(name) or object())
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setenv("CV_CRITIC_NUM_CTX", "16384")
    monkeypatch.setenv("CV_CRITIC_NUM_PREDICT", "1024")
    monkeypatch.setenv("CV_CRITIC_TIMEOUT_SECONDS", "1200")
    assert apply_jobs.main(["doctor", "--provider", provider, "--critic-model", "gemma3:4b"]) == 0
    output = capsys.readouterr().out
    assert "ollama" in checked
    assert "CV critic: Ollama (local); model: gemma3:4b" in output
    assert "CV_CRITIC_NUM_CTX: 16384" in output
    assert "CV_CRITIC_NUM_PREDICT: 1024" in output
    assert "CV_CRITIC_TIMEOUT_SECONDS: 1200" in output
    assert "does not contact Ollama or verify model files" in output
    assert "not validated ATS scores" in output


def test_doctor_no_critic_does_not_require_ollama_for_openai(monkeypatch, capsys):
    checked = []
    monkeypatch.setattr(apply_jobs.importlib.util, "find_spec", lambda name: checked.append(name) or object())
    monkeypatch.setenv("CV_CRITIC_MODEL", "gemma3:4b")
    assert apply_jobs.main(["doctor", "--provider", "openai", "--no-critic"]) == 0
    assert "ollama" not in checked
    assert "CV critic:" not in capsys.readouterr().out


def test_doctor_reports_missing_critic_sdk_without_constructing_it(monkeypatch):
    monkeypatch.setattr(apply_jobs.importlib.util, "find_spec", lambda name: None if name == "ollama" else object())
    assert apply_jobs.main(["doctor", "--provider", "openai", "--critic-model", "gemma3:4b"]) == 1


def test_doctor_rejects_same_model_override_without_network(monkeypatch, capsys):
    monkeypatch.setattr(apply_jobs.importlib.util, "find_spec", forbidden)
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["doctor", "--provider", "ollama", "--model", "same:4b", "--critic-model", "same:4b"])
    assert error.value.code == 2
    assert "critic model must differ" in capsys.readouterr().err


def test_deferred_local_role_loads_once_and_has_stable_fingerprint(monkeypatch):
    calls = []
    fake = SimpleNamespace(model="gemma3:4b", model_digest="local-digest", usage={"requests": 2},
                           budget_diagnostics=[{"context_tokens": 12288, "reported_prompt_tokens": 100}],
                           generate=lambda prompt: prompt,
                           generate_structured=lambda **kwargs: kwargs["prompt"])
    monkeypatch.setattr(apply_jobs, "build_local_role_provider", lambda model, role: calls.append((model, role)) or fake)
    provider = apply_jobs.DeferredGenerationProvider("ollama", "gemma3:4b", role="critic")
    fingerprint = provider.fingerprint
    assert not calls
    assert provider.model_digest is None
    assert provider.budget_diagnostics == []
    assert provider.usage["requests"] == 0
    assert fingerprint["role"] == "critic" and fingerprint["num_ctx"] == 12288
    assert fingerprint["num_predict"] == 4096 and fingerprint["keep_alive"] == 0
    assert provider.generate("first") == "first"
    assert provider.generate_structured("second", dict) == "second"
    assert calls == [("gemma3:4b", "critic")]
    assert provider.fingerprint == fingerprint
    assert provider.model_digest == "local-digest"
    assert provider.budget_diagnostics == fake.budget_diagnostics
    assert provider.usage == {"requests": 2}
    fingerprint["role"] = "changed"
    assert provider.fingerprint["role"] == "critic"


@pytest.mark.parametrize("role,prefix,model", [
    ("critic", "CV_CRITIC", "gemma3:4b"), ("writer", "OLLAMA", "qwen3:8b"),
])
@pytest.mark.parametrize("setting,value", [
    ("NUM_CTX", "16384"), ("NUM_PREDICT", "1024"), ("TIMEOUT_SECONDS", "1200"),
    ("NUM_CTX", "invalid"), ("BASE_URL", "http://localhost:11435"),
])
def test_deferred_role_rejects_changed_configuration_before_construction(
    monkeypatch, role, prefix, model, setting, value,
):
    from app.providers.base import ModelProviderError

    provider = apply_jobs.DeferredGenerationProvider("ollama", model, role=role)
    fingerprint = provider.fingerprint
    name = "OLLAMA_BASE_URL" if setting == "BASE_URL" else prefix + "_" + setting
    monkeypatch.setenv(name, value)
    with pytest.raises(ModelProviderError, match="configuration changed.*no model request was sent"):
        provider.generate("Complete sources")
    assert provider._provider is None
    assert provider.fingerprint == fingerprint
    assert provider.usage["requests"] == 0
    assert provider.budget_diagnostics == []


def test_deferred_role_accepts_equivalent_normalized_configuration(monkeypatch):
    fake = SimpleNamespace(model="gemma3:4b", generate=lambda prompt: prompt)
    monkeypatch.setattr(apply_jobs, "build_local_role_provider", lambda model, role: fake)
    provider = apply_jobs.DeferredGenerationProvider("ollama", "gemma3:4b", role="critic")
    monkeypatch.setenv("CV_CRITIC_NUM_CTX", "012288")
    monkeypatch.setenv("CV_CRITIC_TIMEOUT_SECONDS", "900.0")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    assert provider.generate("Complete sources") == "Complete sources"


def test_legacy_deferred_provider_keeps_existing_factory(monkeypatch):
    calls = []
    fake = SimpleNamespace(model="qwen3:8b", usage={"requests": 1}, generate=lambda prompt: prompt)
    monkeypatch.setattr(apply_jobs, "local_role_config", forbidden)
    monkeypatch.setattr(apply_jobs, "build_provider", lambda name, model: calls.append((name, model)) or fake)
    provider = apply_jobs.DeferredGenerationProvider("ollama", None)
    assert provider.fingerprint is None and not calls
    assert provider.generate("test") == "test"
    assert calls == [("ollama", None)]


@pytest.mark.parametrize("writer_backend,writer_model,critic_model", [
    ("ollama", "offline-writer", "gemma3:4b"),
    ("openai", "offline-writer", "gemma3:4b"),
    ("ollama", "ministral-3:8b", "qwen3:8b"),
    ("ollama", "ministral-3:14b", "qwen3:8b"),
    ("ollama", "ministral-3:8b", "qwen3:14b"),
    ("ollama", "ministral-3:14b", "qwen3:14b"),
    ("ollama", "qwen3:8b", "ministral-3:14b"),
])
def test_classifier_apply_wires_separate_critic_and_reports_usage(
    workspace, monkeypatch, capsys, writer_backend, writer_model, critic_model,
):
    from app.applications.cv_critique import CVCritique

    matcher = install_matcher(monkeypatch)
    calls, constructed = [], []

    class Generator:
        name = "ollama"

        def __init__(self, model, role):
            self.model, self.role = model, role
            self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        def generate_structured(self, *, prompt, schema, system_prompt):
            self.usage["requests"] += 1
            calls.append((self.role, schema))
            assert matcher.calls == 1
            if self.role == "critic":
                assert schema is CVCritique
                return schema.model_validate({
                    "evaluation": {"requirement_coverage": 90, "clarity": 90, "factual_consistency": True,
                                   "unsupported_claims": [], "missing_critical_information": [], "improvements": []},
                    "revision_needed": False, "suggestions": [],
                })
            assert schema is ReferencedCV
            def entry(text, source_id, style="paragraph"):
                return {"text": text, "evidence_ids": [source_id], "style": style}
            return schema.model_validate({"sections": [
                {"heading": "Contact", "entries": [entry("Alex Example", "S0001"), entry("alex@example.test", "S0002")]},
                {"heading": "Experience", "entries": [entry("Built Python and SQL data pipelines.", "S0003", "bullet")]},
            ]})

    def local(model, role):
        constructed.append(("ollama", model, role))
        assert matcher.calls == 1
        return Generator(model, role)

    def generic(name, model):
        assert name == "openai"
        constructed.append((name, model, "writer"))
        assert matcher.calls == 1
        result = Generator(model, "writer")
        result.name = name
        return result

    monkeypatch.setattr(apply_jobs, "build_local_role_provider", local)
    monkeypatch.setattr(apply_jobs, "build_provider", generic)
    assert apply_jobs.main(command(workspace, "--provider", writer_backend, "--model", writer_model,
                                  "--critic-model", critic_model, "--max-revisions", "0")) == 0
    assert calls == [("writer", ReferencedCV), ("critic", CVCritique)]
    assert constructed == [(writer_backend, writer_model, "writer"), ("ollama", critic_model, "critic")]
    record = ApplicationLedger(workspace.output / "applications.db").list()[0]
    assert record["decision"]["verdict"] == "APPLY"
    assert record["model"] == writer_model and record["critic_model"] == critic_model
    assert Path(record["artifacts"]["pdf"]).is_file()
    output = capsys.readouterr().out
    assert 'Ollama critic usage: {"requests": 1' in output


@pytest.mark.parametrize("writer_model,critic_model", [
    ("ministral-3:8b", "qwen3:8b"), ("ministral-3:14b", "qwen3:8b"),
    ("ministral-3:8b", "qwen3:14b"), ("ministral-3:14b", "qwen3:14b"),
])
def test_reversed_roles_dry_run_keeps_independent_budgets_and_no_model_calls(
    workspace, monkeypatch, capsys, writer_model, critic_model,
):
    monkeypatch.setattr(apply_jobs, "build_matcher", forbidden)
    monkeypatch.setattr(apply_jobs, "ApplicationLedger", forbidden)
    assert apply_jobs.main(command(workspace, "--provider", "ollama", "--matcher", "classifier",
                                   "--model", writer_model, "--critic-model", critic_model,
                                   "--max-revisions", "0", "--dry-run")) == 0
    output = capsys.readouterr().out
    assert "Maximum logical writer calls: 1; matching calls: 0" in output
    assert "Maximum logical critic calls: 1;" in output
    assert "CV critic: Ollama (local); model: " + critic_model in output
    assert "No network requests, model calls, or application writes" in output
    assert not workspace.output.exists()


@pytest.mark.parametrize("size", ["8b", "14b"])
def test_reversed_roles_do_not_construct_models_for_classifier_review(workspace, monkeypatch, size):
    matcher = install_matcher(monkeypatch, "review")
    writer_model, critic_model = "ministral-3:" + size, "qwen3:" + size
    assert apply_jobs.main(command(workspace, "--provider", "ollama", "--matcher", "classifier",
                                   "--model", writer_model, "--critic-model", critic_model,
                                   "--max-revisions", "0")) == 0
    record = ApplicationLedger(workspace.output / "applications.db").list()[0]
    assert matcher.calls == 1
    assert record["status"] == "review" and record["decision"]["verdict"] == "REVIEW"
    assert record["model"] == writer_model and record["critic_model"] == critic_model
    assert record["artifacts"] == {}
