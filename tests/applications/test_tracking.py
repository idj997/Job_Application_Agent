import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from app.applications.tracking import ApplicationLedger, job_key


def test_ledger_persists_artifacts_and_updates_across_instances(tmp_path):
    path = tmp_path / "state" / "applications.db"
    ledger = ApplicationLedger(path)
    ledger.save("vacancy", {"status": "ready", "title": "Engineer", "artifacts": {"pdf": "/tmp/cv.pdf"}})
    original = ledger.get("vacancy")
    reopened = ApplicationLedger(path)
    assert reopened.get("vacancy") == original
    reopened.save("vacancy", {**original, "status": "opened"})
    updated = ledger.get("vacancy")
    assert updated["created_at"] == original["created_at"]
    assert updated["status"] == "opened"
    assert ledger.list() == [updated]


def test_submitted_application_cannot_be_reset_by_preparation(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "ready"})
    ledger.mark_submitted("vacancy", "Employer receipt ABC123")
    receipt = ledger.get("vacancy")
    ledger.save("vacancy", {"status": "ready", "title": "retry"})
    assert ledger.get("vacancy") == receipt
    assert receipt["confirmation"] == "Employer receipt ABC123"
    assert receipt["submitted_at"]


def test_uncertain_submission_cannot_be_reset_by_preparation(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "submission_unknown", "detail": "Click timed out"})
    ledger.save("vacancy", {"status": "ready"})
    assert ledger.get("vacancy")["status"] == "submission_unknown"


def test_attempt_marker_survives_a_restart_and_can_be_finalized(tmp_path):
    path = tmp_path / "applications.db"
    ledger = ApplicationLedger(path)
    ledger.save("vacancy", {"status": "opened", "submission_attempted": True})
    reopened = ApplicationLedger(path)
    reopened.save("vacancy", {"status": "ready"})
    assert reopened.get("vacancy")["submission_attempted"] is True
    reopened.mark_submitted("vacancy", "Employer receipt")
    assert reopened.get("vacancy")["status"] == "submitted"


@pytest.mark.parametrize("status", ["review", "skipped", "failed"])
def test_only_ready_or_opened_can_be_confirmed(tmp_path, status):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": status})
    with pytest.raises(ValueError, match="ready, opened, or uncertain"):
        ledger.mark_submitted("vacancy", "Employer receipt")


def test_confirmation_cannot_be_empty_or_forged_via_save(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "opened"})
    with pytest.raises(ValueError, match="nonempty"):
        ledger.mark_submitted("vacancy", "  ")
    with pytest.raises(ValueError, match="mark_submitted"):
        ledger.save("vacancy", {"status": "submitted"})
    assert ledger.get("vacancy")["status"] == "opened"


def test_tracking_links_and_lever_apply_page_share_identity():
    base = {"source_url": "https://jobs.lever.co/example/123"}
    tracked = {"application_url": "https://jobs.lever.co/example/123/apply/?utm_source=feed&lever-source=feed#form"}
    assert job_key(base) == job_key(tracked)


def test_requisition_identity_survives_different_sources():
    first = {"company": "Example Ltd", "metadata": {"requisition_id": "R001"}, "source": "board-a"}
    second = {"company": "example ltd", "requisition_id": "r001", "source": "board-b"}
    assert job_key(first) == job_key(second)
    assert job_key(first) != job_key({**second, "company": "Another employer"})


def test_nontracking_query_parameters_identify_distinct_vacancies():
    assert job_key({"source_url": "https://example.com/jobs?id=1"}) != job_key({"source_url": "https://example.com/jobs?id=2"})


def test_job_without_stable_identity_is_rejected():
    with pytest.raises(ValueError, match="valid source URL"):
        job_key({"title": "Engineer"})


def test_explicit_receipt_resolves_an_uncertain_submission(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "submission_unknown"})
    ledger.mark_submitted("vacancy", "Employer emailed application receipt ABC123")
    assert ledger.get("vacancy")["status"] == "submitted"


def test_checked_nonsubmission_clears_attempt_and_preserves_history(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "opened", "submission_attempted": True})
    ledger.resolve_not_submitted("vacancy", "Checked employer account: no application exists")
    record = ledger.get("vacancy")
    assert record["status"] == "ready"
    assert not record.get("submission_attempted")
    assert record["resolution_history"][0]["previous_status"] == "opened"
    assert record["resolution_history"][0]["note"] == "Checked employer account: no application exists"
    ledger.save("vacancy", {**record, "status": "submission_unknown"})
    ledger.resolve_not_submitted("vacancy", "Employer support confirmed no submission was received")
    assert len(ledger.get("vacancy")["resolution_history"]) == 2
    ledger.save("vacancy", {"status": "ready", "title": "Freshly regenerated application"})
    assert len(ledger.get("vacancy")["resolution_history"]) == 2


def test_resolution_requires_note_and_cannot_reset_confirmed_submission(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "opened", "submission_attempted": True})
    with pytest.raises(ValueError, match="nonempty note"):
        ledger.resolve_not_submitted("vacancy", " ")
    assert ledger.get("vacancy")["submission_attempted"] is True
    ledger.mark_submitted("vacancy", "Employer receipt")
    with pytest.raises(ValueError, match="cannot be reset"):
        ledger.resolve_not_submitted("vacancy", "Try again")
    assert ledger.get("vacancy")["status"] == "submitted"


def test_claim_open_returns_original_and_persists_attempt(tmp_path):
    ledger = ApplicationLedger(tmp_path / "applications.db")
    ledger.save("vacancy", {"status": "ready", "title": "Engineer"})
    ready = ledger.get("vacancy")
    assert ledger.claim_open("vacancy") == ready
    assert ledger.get("vacancy")["status"] == "opened"
    assert ledger.get("vacancy")["submission_attempted"] is True
    with pytest.raises(ValueError, match="already opened"):
        ledger.claim_open("vacancy")


def test_two_parallel_open_claims_cannot_both_succeed(tmp_path):
    path = tmp_path / "applications.db"
    ledger = ApplicationLedger(path)
    ledger.save("vacancy", {"status": "ready"})
    start = Barrier(2)

    def claim():
        connection = ApplicationLedger(path)
        start.wait(timeout=5)
        try:
            return connection.claim_open("vacancy")["status"]
        except ValueError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: claim(), range(2))) == ["blocked", "ready"]
    assert ledger.get("vacancy")["submission_attempted"] is True
