from copy import deepcopy

import pytest

from app.applications.browser import (
    FIELD_SELECTORS, RESUME_SELECTORS, SUBMIT_SELECTORS,
    application_site, guard_public_requests, prepare_page, validate_application_url,
)


class FakeLocator:
    def __init__(self, page, selector, count=1):
        self.page = page
        self.selector = selector
        self.matches = count

    def count(self):
        return self.matches

    def is_visible(self):
        return True

    def is_enabled(self):
        return True

    def input_value(self):
        return self.page.values.get(self.selector, "")

    def fill(self, value):
        self.page.values[self.selector] = value

    def set_input_files(self, path):
        self.page.uploads.append(path)

    def evaluate(self, script, argument=None):
        return self.page.blockers if argument is not None else "Submit application"

    def click(self, **kwargs):
        self.page.clicks += 1
        if self.page.click_error:
            raise TimeoutError("The response timed out after the click")

    def nth(self, index):
        return self

    def inner_text(self):
        return self.page.receipt


class FakePage:
    def __init__(self, url="https://jobs.lever.co/example/123/apply"):
        self.url = url
        self.values = {}
        self.uploads = []
        self.clicks = 0
        self.blockers = []
        self.receipt = "Application submitted!"
        self.click_error = False
        self.captcha = False

    def locator(self, selector):
        if selector.startswith("[data-sitekey]"):
            return FakeLocator(self, selector, int(self.captcha))
        if selector.startswith("#application_confirmation"):
            return FakeLocator(self, selector, int(bool(self.receipt and self.clicks)))
        known = set(FIELD_SELECTORS["lever"].values()) | {RESUME_SELECTORS["lever"], SUBMIT_SELECTORS["lever"]}
        return FakeLocator(self, selector, int(selector in known))

    def wait_for_timeout(self, timeout):
        pass


@pytest.fixture
def record(tmp_path):
    pdf = tmp_path / "cv.pdf"
    pdf.write_bytes(b"%PDF-test")
    return {
        "status": "ready", "application_url": "https://jobs.lever.co/example/123/apply",
        "profile": {"name": "Applicant Name", "email": "candidate@example.test", "phone": "01234567890"},
        "artifacts": {"pdf": str(pdf)},
    }


def test_default_fills_exact_fields_and_cv_without_submitting(record):
    page = FakePage()
    outcome = prepare_page(page, record)
    assert outcome["status"] == "opened"
    assert page.clicks == 0
    assert page.values == {'input[name="name"]': "Applicant Name", 'input[name="email"]': "candidate@example.test", 'input[name="phone"]': "01234567890"}
    assert page.uploads == [record["artifacts"]["pdf"]]


def test_explicit_submit_requires_real_receipt(record):
    page = FakePage()
    outcome = prepare_page(page, record, submit=True)
    assert page.clicks == 1
    assert outcome["status"] == "submitted"
    assert "Application submitted!" in outcome["confirmation"]


@pytest.mark.parametrize("reason", ["A custom application question needs the applicant", "A required field is incomplete or invalid"])
def test_unanswered_fields_prevent_submit(record, reason):
    page = FakePage()
    page.blockers = [reason]
    outcome = prepare_page(page, record, submit=True)
    assert outcome["status"] == "opened"
    assert page.clicks == 0


def test_captcha_prevents_submit(record):
    page = FakePage()
    page.captcha = True
    outcome = prepare_page(page, record, submit=True)
    assert outcome["status"] == "opened"
    assert "verification" in outcome["detail"]
    assert page.clicks == 0


@pytest.mark.parametrize("click_error", [False, True])
def test_ambiguous_submission_is_never_retried(record, click_error):
    page = FakePage()
    page.receipt = ""
    page.click_error = click_error
    outcome = prepare_page(page, record, submit=True)
    assert outcome["status"] == "submission_unknown"
    assert page.clicks == 1
    outcome = prepare_page(page, {**record, "status": outcome["status"]}, submit=True)
    assert page.clicks == 1
    assert outcome["status"] == "submission_unknown"


def test_redirect_to_another_vacancy_does_not_receive_personal_data(record):
    page = FakePage("https://jobs.lever.co/another-employer/999/apply")
    outcome = prepare_page(page, record, submit=True)
    assert outcome["status"] == "opened"
    assert page.values == {}
    assert page.uploads == []
    assert page.clicks == 0


def test_existing_field_conflict_is_preserved_and_prevents_submit(record):
    page = FakePage()
    page.values['input[name="email"]'] = "other@example.test"
    outcome = prepare_page(page, record, submit=True)
    assert page.values['input[name="email"]'] == "other@example.test"
    assert outcome["status"] == "opened"
    assert page.clicks == 0


def test_missing_cv_prevents_submit(record):
    page = FakePage()
    record = deepcopy(record)
    record["artifacts"] = {}
    assert prepare_page(page, record, submit=True)["status"] == "opened"
    assert page.clicks == 0


def test_attempt_is_saved_before_click_and_storage_failure_prevents_submit(record):
    page = FakePage()
    saved = []

    def save_attempt():
        assert page.clicks == 0
        saved.append(True)

    assert prepare_page(page, record, submit=True, before_submit=save_attempt)["status"] == "submitted"
    assert saved == [True]
    page = FakePage()

    def failing_storage():
        raise OSError("Disk full")

    assert prepare_page(page, record, submit=True, before_submit=failing_storage)["status"] == "opened"
    assert page.clicks == 0


@pytest.mark.parametrize("state", [{"status": "review"}, {"status": "skipped"}, {"status": "opened", "submission_attempted": True}])
def test_nonready_or_attempted_application_is_not_filled_or_submitted(record, state):
    page = FakePage()
    outcome = prepare_page(page, {**record, **state}, submit=True)
    assert outcome["status"] != "submitted"
    assert page.values == {}
    assert page.uploads == []
    assert page.clicks == 0


@pytest.mark.parametrize("url", [
    "file:///tmp/cv.pdf", "javascript:alert(1)", "https://user:password@example.com/jobs",
    "https://example.com:invalid/jobs", "https://jobs.lever.co:8443/example/123",
    "http://localhost/jobs", "http://127.0.0.1/jobs", "http://[::1]/jobs",
    "http://192.168.1.1/jobs", "https://example.com/\nmalformed", "https://example.com\\@localhost/jobs",
])
def test_invalid_application_url_is_rejected(url):
    with pytest.raises(ValueError):
        validate_application_url(url)


def test_host_allowlist_does_not_match_lookalike_or_unencrypted_urls():
    assert application_site("https://jobs.lever.co/example/123") == "lever"
    assert application_site("https://jobs.lever.co.example.com/example/123") is None
    assert application_site("http://jobs.lever.co/example/123") is None


def test_browser_request_guard_blocks_private_redirect_without_network(monkeypatch):
    from types import SimpleNamespace
    from app.applications import jobs

    def destination(url):
        if "internal.example" in url:
            raise ValueError("Hostname resolved to a private address")

    monkeypatch.setattr(jobs, "_public_destination", destination)
    routes = {}
    context = SimpleNamespace(route=lambda pattern, handler: routes.update(handler=handler))
    guard_public_requests(context)
    calls = []
    route = SimpleNamespace(
        request=SimpleNamespace(url="https://internal.example/redirected-application"),
        abort=lambda reason: calls.append("blocked"), continue_=lambda: calls.append("continued"),
    )
    routes["handler"](route)
    assert calls == ["blocked"]
    route.request.url = "https://jobs.lever.co/example/123/apply"
    routes["handler"](route)
    assert calls == ["blocked", "continued"]
