"""Exercise the real SDK and document pipeline with HTTP confined to memory."""

import json
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from app.applications.documents import read_cv
from app.applications.models import CandidateProfile, MatchAssessment, Preferences
from app.applications.tracking import ApplicationLedger, job_key
from app.applications.workflow import ApplicationWorkflow
from app.providers.openai import OpenAIProvider


def _assert_objects_are_strict(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
        for child in schema.values():
            _assert_objects_are_strict(child)
    elif isinstance(schema, list):
        for child in schema:
            _assert_objects_are_strict(child)


@pytest.mark.parametrize("backend,matcher_mode", [("openai", "llm"), ("ollama", "llm"), ("ollama", "classifier")])
@pytest.mark.parametrize("reference_problem", [None, "legacy_quotes", "unknown_id"])
def test_real_sdk_strict_parsing_documents_ledger_and_cached_rerun(tmp_path, backend, matcher_mode, reference_problem):
    master_cv = """Jérôme Müller
jerome@example.test | London
Data Engineer | Example Ltd | 2022–present
Built Python pipelines that cut report preparation by 20%.
BSc Computer Science, Example University, 2021.
"""
    job = {
        "title": "Data Engineer", "company": "Hiring Ltd",
        "source": "greenhouse", "external_id": "sdk-integration",
        "source_url": "https://boards.greenhouse.io/hiring/jobs/sdk-integration",
        "metadata": {"description_type": "full"},
        "description": (
            "Python experience is required. This role is based in London. "
            "The Data Engineer will maintain reporting pipelines for our growing team. "
            "You will discuss reporting needs with colleagues, document the existing "
            "systems and identify practical improvements to their day-to-day use. "
            "We offer a supportive environment and give new colleagues time to learn "
            "the systems, understand the work and contribute to our shared projects."
        ),
    }
    source_path = tmp_path / "master_cv.md"
    source_path.write_text(master_cv, encoding="utf-8")
    profile = CandidateProfile(
        name="Jérôme Müller", email="jerome@example.test", cv_path=str(source_path),
        preferences=Preferences(locations=["London"]),
    )

    def cv_entry(text, source_id, style="paragraph"):
        return {"text": text, "style": style, "evidence_ids": [source_id]}

    queued_outputs = [
        {
            "description_complete": True,
            "requirements": [{
                "requirement": "Python", "importance": "CORE", "status": "MET",
                "job_evidence": "Python experience is required.",
                "cv_evidence": ["Built Python pipelines that cut report preparation by 20%."],
                "explanation": "The candidate has documented Python pipeline experience.",
            }],
            "constraint_checks": [{
                "constraint_id": "location", "status": "PASS",
                "job_evidence": "This role is based in London.",
                "explanation": "The advertised location matches the preference.",
            }],
            "recommended_verdict": "APPLY", "rationale": "Experience and location fit.",
            "uncertainties": [],
        },
        {
            "sections": [
                {"heading": "Contact", "entries": [
                    cv_entry("Jérôme Müller", "S0001"), cv_entry("jerome@example.test | London", "S0002"),
                ]},
                {"heading": "Experience", "entries": [
                    cv_entry("Data Engineer | Example Ltd | 2022–present", "S0003"),
                    cv_entry("Built Python pipelines that cut report preparation by 20%.", "S0004", "bullet"),
                ]},
                {"heading": "Education", "entries": [
                    cv_entry("BSc Computer Science, Example University, 2021.", "S0005"),
                ]},
            ],
        },
        {
            "requirement_coverage": 88, "clarity": 92, "factual_consistency": True,
            "unsupported_claims": [], "missing_critical_information": [], "improvements": [],
        },
    ]
    if reference_problem == "legacy_quotes":
        # A formerly accepted model-written quote must not slip through the
        # actual SDK/schema boundary after migrating to source references.
        for section in queued_outputs[1]["sections"]:
            for entry in section["entries"]:
                entry.pop("evidence_ids")
                entry["evidence_quotes"] = [entry["text"]]
    elif reference_problem == "unknown_id":
        queued_outputs[1]["sections"][1]["entries"][1]["evidence_ids"] = ["S9999"]
    requests = []

    class OfflineMatcher:
        name = "classifier"
        fingerprint = {"version": "offline-integration"}
        diagnostics = {"note": "Test-double predictions, not model accuracy evidence."}

        def assess(self, actual_job, actual_profile, actual_cv):
            assert actual_job == job
            assert actual_profile == profile
            assert actual_cv == master_cv.strip()
            return MatchAssessment.model_validate(queued_outputs.pop(0))

    def respond(request):
        body = json.loads(request.content)
        requests.append((request, body))
        output = queued_outputs.pop(0)
        if backend == "ollama":
            return httpx.Response(200, json={
                "model": "offline-model", "created_at": "2026-09-13T12:00:00Z",
                "message": {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
                "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 50,
            })
        return httpx.Response(200, json={
            "id": f"resp_offline_{len(requests)}", "object": "response",
            "created_at": 1_700_000_000, "status": "completed", "model": "offline-model",
            "error": None, "incomplete_details": None, "instructions": None,
            "metadata": {}, "parallel_tool_calls": True, "tools": [],
            "tool_choice": "auto", "temperature": 1.0, "top_p": 1.0,
            "output": [{
                "id": f"msg_offline_{len(requests)}", "type": "message",
                "role": "assistant", "status": "completed",
                "content": [{
                    "type": "output_text", "text": json.dumps(output, ensure_ascii=False),
                    "annotations": [],
                }],
            }],
            "usage": {
                "input_tokens": 100, "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 150,
            },
        })

    # MockTransport handles every request for either SDK. No external or local
    # model server is contacted, even with real credentials in the environment.
    with ExitStack() as stack:
        if backend == "openai":
            client = stack.enter_context(OpenAI(
                api_key="offline-test", base_url="https://openai.invalid/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            ))
            provider = OpenAIProvider(model="offline-model", client=client, max_output_tokens=4000)
        else:
            from ollama import Client
            from app.providers.ollama_provider import OllamaProvider
            client = Client(host="http://127.0.0.1:11434", transport=httpx.MockTransport(respond), trust_env=False)
            stack.enter_context(client._client)
            provider = OllamaProvider(model="offline-model", client=client, num_ctx=32768, num_predict=4000)
        ledger = ApplicationLedger(tmp_path / "applications.sqlite3")
        workflow = ApplicationWorkflow(
            provider, ledger, tmp_path / "output", max_revisions=0,
            matcher=OfflineMatcher() if matcher_mode == "classifier" else None,
        )

        record = workflow.prepare(job, profile, read_cv(source_path))

        if reference_problem:
            assert record["status"] == "failed"
            assert record["decision"]["verdict"] == "APPLY"
            assert record["reasons"]
            assert "cv_score" not in record
            assert len(requests) == (1 if matcher_mode == "classifier" else 2)
            # The audit remains untouched, and no employer-facing documents
            # can be produced from a rejected response or unresolved ID.
            assert len(queued_outputs) == 1 and "factual_consistency" in queued_outputs[0]
            assert not {"markdown", "docx", "pdf", "draft_json"} & record["artifacts"].keys()
            assert "referenced_cv" not in record["artifacts"]
            if reference_problem == "unknown_id":
                failed = json.loads(Path(record["artifacts"]["failed_referenced_cv"]).read_text())
                assert failed["sections"][1]["entries"][1]["evidence_ids"] == ["S9999"]
            else:
                assert "failed_referenced_cv" not in record["artifacts"]
            assert Path(record["artifacts"]["source_catalog"]).is_file()
            assert ledger.get(job_key(job))["status"] == "failed"
            with pytest.raises(ValueError, match="not ready"):
                ledger.claim_open(record["job_key"])
            return

        assert record["status"] == "ready", record["reasons"]
        assert record["provider"] == backend
        assert record["matcher"] == matcher_mode
        assert record["decision"]["verdict"] == "APPLY"
        assert record["decision"]["score"] == 100
        assert record["cv_score"] == 89
        assert not queued_outputs
        request_count = 2 if matcher_mode == "classifier" else 3
        assert len(requests) == request_count
        assert provider.usage == {
            "requests": request_count, "input_tokens": 100 * request_count,
            "output_tokens": 50 * request_count, "total_tokens": 150 * request_count,
        }
        schemas = ["ReferencedCV", "CVEvaluation"] if matcher_mode == "classifier" else ["MatchAssessment", "ReferencedCV", "CVEvaluation"]
        source_catalog = [
            {"source_id": f"S{index:04d}", "text": text}
            for index, text in enumerate(master_cv.splitlines(), start=1)
        ]
        for (request, body), schema_name in zip(requests, schemas):
            assert request.method == "POST"
            messages = body["messages" if backend == "ollama" else "input"]
            payload = json.loads(messages[1]["content"])
            if schema_name == "ReferencedCV":
                assert payload["master_cv_sources"] == source_catalog
                assert "master_cv" not in payload
                assert payload["tailoring_policy"]
                schema = body["format"] if backend == "ollama" else body["text"]["format"]["schema"]
                assert '"evidence_ids"' in json.dumps(schema)
                assert '"evidence_quotes"' not in json.dumps(schema)
            elif schema_name == "CVEvaluation":
                assert payload["master_cv"] == master_cv.strip()
                assert "master_cv_sources" not in payload
                assert payload["proposed_cv"] == record["cv_attempts"][0]["cv"]
                assert payload["proposed_cv"]["sections"][1]["entries"][1] == {
                    "text": "Built Python pipelines that cut report preparation by 20%.",
                    "style": "bullet",
                    "evidence_quotes": ["Built Python pipelines that cut report preparation by 20%."],
                }
                assert "evidence_ids" not in json.dumps(payload["proposed_cv"])
            if backend == "ollama":
                assert request.url.host == "127.0.0.1"
                assert request.url.path == "/api/chat"
                assert body["model"] == "offline-model"
                assert body["stream"] is False and body["think"] is False
                assert body["options"]["num_ctx"] == 32768
                assert body["options"]["num_predict"] == 4000
                assert [message["role"] for message in body["messages"]] == ["system", "user"]
                assert body["format"]["title"] == schema_name
                # Ollama enforces a JSON schema, validated again with Pydantic.
                assert body["format"]["additionalProperties"] is False
                continue
            assert request.url.host == "openai.invalid"
            assert request.url.path == "/v1/responses"
            assert body["store"] is False
            assert body["max_output_tokens"] == 4000
            assert body["model"] == "offline-model"
            assert [message["role"] for message in body["input"]] == ["system", "user"]
            format_ = body["text"]["format"]
            assert format_["type"] == "json_schema"
            assert format_["name"] == schema_name
            assert format_["strict"] is True
            _assert_objects_are_strict(format_["schema"])

        for extension in ("markdown", "docx", "pdf"):
            artifact = Path(record["artifacts"][extension])
            assert artifact.is_file() and artifact.is_absolute()
            text = " ".join(read_cv(artifact).split())
            assert "Jérôme Müller" in text
            assert "jerome@example.test" in text
            assert "Data Engineer | Example Ltd | 2022–present" in text
            assert "cut report preparation by 20%" in text
            assert "BSc Computer Science, Example University, 2021." in text
            assert "S0001" not in text and "evidence_ids" not in text
        assert json.loads(Path(record["artifacts"]["source_catalog"]).read_text()) == source_catalog
        attempt = record["cv_attempts"][0]
        assert attempt["referenced_cv"]["sections"][1]["entries"][1]["evidence_ids"] == ["S0004"]
        assert attempt["cv"]["sections"][1]["entries"][1]["evidence_quotes"] == [
            "Built Python pipelines that cut report preparation by 20%."
        ]
        assert ledger.get(job_key(job))["artifacts"] == record["artifacts"]

        repeated = workflow.prepare(job, profile, read_cv(source_path))

        assert repeated["cached"] is True
        assert repeated["status"] == "ready"
        assert repeated["artifacts"] == record["artifacts"]
        assert len(requests) == request_count
