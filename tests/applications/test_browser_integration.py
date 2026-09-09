"""Real DOM coverage using an offline Chromium page, never a live job site."""

from pathlib import Path

import pytest

from app.applications.browser import configure_browser_runtime, prepare_page
from app.applications.tracking import ApplicationLedger


LEVER_FORM = """
<!doctype html>
<html><body>
  <h1>Apply for this job</h1>
  <form id="application">
    <label>Full name <input name="name" required></label>
    <label>Email <input name="email" type="email" required></label>
    <label>Phone <input name="phone"></label>
    <label>Resume <input type="file" name="resume" required></label>
    CUSTOM_QUESTION
    <button id="btn-submit" type="submit">Submit application</button>
  </form>
  <script>
    window.submitCount = 0;
    document.querySelector('form').addEventListener('submit', event => {
      event.preventDefault();
      window.submitCount += 1;
      document.querySelector('h1').textContent = 'Application submitted!';
    });
  </script>
</body></html>
"""


class EmployerPage:
    """Keep the real page offline while supplying the expected employer URL."""

    url = "https://jobs.lever.co/example/123/apply"

    def __init__(self, page):
        self.page = page

    def __getattr__(self, name):
        return getattr(self.page, name)


@pytest.fixture(scope="module")
def chromium_browser():
    configure_browser_runtime()
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except sync_api.Error as exc:
            pytest.skip(f"Local Chromium unavailable: {str(exc).splitlines()[0]}")
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def employer_page(chromium_browser):
    context = chromium_browser.new_context(offline=True)
    # Defense in depth: any accidental request introduced in a fixture is blocked.
    context.route("**/*", lambda route: route.abort())
    page = context.new_page()
    page.set_content(LEVER_FORM.replace("CUSTOM_QUESTION", ""))
    try:
        yield EmployerPage(page)
    finally:
        context.close()


@pytest.fixture
def application_record(tmp_path):
    cv = tmp_path / "candidate.pdf"
    cv.write_bytes(b"%PDF-1.4\n%Local fixture; no upload destination\n")
    return {
        "status": "ready",
        "application_url": EmployerPage.url,
        "profile": {"name": "Test Applicant", "email": "applicant@example.test", "phone": "01234567890"},
        "artifacts": {"pdf": str(cv)},
    }


def test_real_lever_form_fill_upload_and_receipt(employer_page, application_record):
    result = prepare_page(employer_page, application_record, submit=True)
    assert result["status"] == "submitted"
    assert "Application submitted!" in result["confirmation"]
    assert employer_page.locator('input[name="name"]').input_value() == "Test Applicant"
    assert employer_page.locator('input[name="email"]').input_value() == "applicant@example.test"
    assert employer_page.locator('input[name="phone"]').input_value() == "01234567890"
    assert employer_page.locator('input[name="resume"]').evaluate("input => input.files[0].name") == Path(application_record["artifacts"]["pdf"]).name
    assert employer_page.evaluate("window.submitCount") == 1


def test_real_form_without_submit_flag_stays_open(employer_page, application_record):
    result = prepare_page(employer_page, application_record)
    assert result["status"] == "opened"
    assert employer_page.evaluate("window.submitCount") == 0


@pytest.mark.parametrize("custom_question", [
    '<label>Why this company?<textarea name="motivation" required></textarea></label>',
    '<label><input type="checkbox" name="consent" required>Confirm this statement</label>',
    '<div role="combobox" aria-label="Work authorization"></div>',
])
def test_real_custom_question_blocks_submission(employer_page, application_record, custom_question):
    employer_page.set_content(LEVER_FORM.replace("CUSTOM_QUESTION", custom_question))
    result = prepare_page(employer_page, application_record, submit=True)
    assert result["status"] == "opened"
    assert "custom application question" in result["detail"]
    assert employer_page.evaluate("window.submitCount") == 0


def test_real_form_requires_name_even_if_html_does_not(employer_page, application_record):
    employer_page.set_content(LEVER_FORM.replace("CUSTOM_QUESTION", "").replace('name="name" required', 'name="name"'))
    application_record["profile"].pop("name")
    result = prepare_page(employer_page, application_record, submit=True)
    assert result["status"] == "opened"
    assert "name or email is incomplete" in result["detail"]
    assert employer_page.evaluate("window.submitCount") == 0


def test_attempt_is_durable_before_real_form_submit(employer_page, application_record, tmp_path):
    path = tmp_path / "applications.db"
    ledger = ApplicationLedger(path)
    ledger.save("vacancy", application_record)

    def record_attempt():
        assert employer_page.evaluate("window.submitCount") == 0
        ledger.save("vacancy", {**application_record, "status": "opened", "submission_attempted": True})
        assert ApplicationLedger(path).get("vacancy")["submission_attempted"] is True

    result = prepare_page(employer_page, application_record, submit=True, before_submit=record_attempt)
    assert result["status"] == "submitted"
    ledger.mark_submitted("vacancy", result["confirmation"])
    assert ApplicationLedger(path).get("vacancy")["status"] == "submitted"


def test_persistence_failure_blocks_real_form_submit(employer_page, application_record):
    def storage_failure():
        raise OSError("No space left on device")

    result = prepare_page(employer_page, application_record, submit=True, before_submit=storage_failure)
    assert result["status"] == "opened"
    assert "Could not save" in result["detail"]
    assert employer_page.evaluate("window.submitCount") == 0
