"""Exercise the real SDK and document pipeline with HTTP confined to memory."""

import json
from pathlib import Path

import httpx
from openai import OpenAI

from app.applications.documents import read_cv
from app.applications.models import CandidateProfile, Preferences
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


def test_real_sdk_strict_parsing_documents_ledger_and_cached_rerun(tmp_path):
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

    def cv_entry(text, style="paragraph"):
        return {"text": text, "style": style, "evidence_quotes": [text]}

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
                    cv_entry("Jérôme Müller"), cv_entry("jerome@example.test | London"),
                ]},
                {"heading": "Experience", "entries": [
                    cv_entry("Data Engineer | Example Ltd | 2022–present"),
                    cv_entry("Built Python pipelines that cut report preparation by 20%.", "bullet"),
                ]},
                {"heading": "Education", "entries": [
                    cv_entry("BSc Computer Science, Example University, 2021."),
                ]},
            ],
        },
        {
            "requirement_coverage": 88, "clarity": 92, "factual_consistency": True,
            "unsupported_claims": [], "missing_critical_information": [], "improvements": [],
        },
    ]
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request, body))
        output = queued_outputs.pop(0)
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

    # MockTransport handles every HTTP request; the .invalid endpoint cannot be
    # contacted even if a developer has a real API key in their environment.
    with OpenAI(
        api_key="offline-test", base_url="https://openai.invalid/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        provider = OpenAIProvider(model="offline-model", client=client, max_output_tokens=4000)
        ledger = ApplicationLedger(tmp_path / "applications.sqlite3")
        workflow = ApplicationWorkflow(provider, ledger, tmp_path / "output", max_revisions=0)

        record = workflow.prepare(job, profile, read_cv(source_path))

        assert record["status"] == "ready", record["reasons"]
        assert record["decision"]["verdict"] == "APPLY"
        assert record["decision"]["score"] == 100
        assert record["cv_score"] == 89
        assert not queued_outputs
        assert len(requests) == 3
        assert provider.usage == {
            "requests": 3, "input_tokens": 300, "output_tokens": 150, "total_tokens": 450,
        }
        for (request, body), schema_name in zip(requests, ["MatchAssessment", "TailoredCV", "CVEvaluation"]):
            assert request.method == "POST"
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
        assert ledger.get(job_key(job))["artifacts"] == record["artifacts"]

        repeated = workflow.prepare(job, profile, read_cv(source_path))

        assert repeated["cached"] is True
        assert repeated["status"] == "ready"
        assert repeated["artifacts"] == record["artifacts"]
        assert len(requests) == 3
