"""Visible, conservative assistance for employer application forms.

The browser stays open until the applicant closes it. Automatic submission is
opt-in, limited to recognized simple forms, and never retried after a click.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit


SUPPORTED_HOSTS = {
    "boards.greenhouse.io": "greenhouse",
    "job-boards.greenhouse.io": "greenhouse",
    "boards.eu.greenhouse.io": "greenhouse",
    "job-boards.eu.greenhouse.io": "greenhouse",
    "jobs.lever.co": "lever",
    "jobs.eu.lever.co": "lever",
}
FIELD_SELECTORS = {
    "greenhouse": {
        "first_name": "input#first_name",
        "last_name": "input#last_name",
        "email": "input#email",
        "phone": "input#phone",
    },
    "lever": {
        "name": 'input[name="name"]',
        "email": 'input[name="email"]',
        "phone": 'input[name="phone"]',
    },
}
RESUME_SELECTORS = {
    "greenhouse": 'input[type="file"]#resume',
    "lever": 'input[type="file"][name="resume"]',
}
SUBMIT_SELECTORS = {
    "greenhouse": 'input#submit_app[type="submit"], button#submit_app[type="submit"]',
    "lever": 'button#btn-submit[type="submit"]',
}
_RECEIPT_TEXT = re.compile(
    r"(?:your application (?:has been|was) (?:successfully )?(?:submitted|received)|"
    r"application submitted|thank you for applying|thanks for applying)", re.IGNORECASE
)


def configure_browser_runtime() -> None:
    """Use the project's installed browsers without overriding an explicit setting."""
    local_browsers = Path(__file__).resolve().parents[2] / ".venv" / "playwright-browsers"
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ and local_browsers.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(local_browsers)


def validate_application_url(url: str) -> str:
    from app.applications.jobs import _parse_url
    _parse_url(url)
    return url


def guard_public_requests(context) -> None:
    """Reject private destinations, including redirects, before browser requests."""
    from app.applications.jobs import _public_destination

    def check_request(route):
        try:
            # Check each request afresh. Chromium still performs its own DNS lookup;
            # this guard is not an IP-pinned network proxy.
            _public_destination(route.request.url)
        except ValueError:
            route.abort("blockedbyclient")
        else:
            route.continue_()

    context.route("**/*", check_request)


def application_site(url: str) -> str | None:
    validate_application_url(url)
    parts = urlsplit(url)
    if parts.scheme.casefold() != "https" or parts.port not in {None, 443}:
        return None
    return SUPPORTED_HOSTS.get(parts.hostname.casefold())


def _result(status: str, detail: str, confirmation: str | None = None) -> dict:
    return {"status": status, "detail": detail, "confirmation": confirmation}


def fill_known_fields(page, record: dict, site: str) -> list[str]:
    """Fill exact known controls only; return issues requiring applicant attention."""
    issues = []
    profile = record.get("profile") or {}
    for name, selector in FIELD_SELECTORS[site].items():
        value = profile.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        field = page.locator(selector)
        if field.count() != 1 or not field.is_visible() or not field.is_enabled():
            issues.append(f"Could not identify the {name} field")
            continue
        existing = field.input_value().strip()
        if existing and existing != value.strip():
            issues.append(f"Existing {name} differs from the saved profile")
            continue
        if not existing:
            field.fill(value.strip())

    pdf = (record.get("artifacts") or {}).get("pdf")
    resume = page.locator(RESUME_SELECTORS[site])
    if not pdf or not Path(pdf).is_file():
        issues.append("The prepared CV PDF is missing")
    elif resume.count() != 1 or not resume.is_enabled():
        issues.append("Could not identify the CV upload field")
    else:
        resume.set_input_files(str(Path(pdf).resolve()))
    return issues


def submission_blockers(page, site: str) -> list[str]:
    """Inspect the actual form, including custom questions and validation state."""
    captcha = page.locator(
        '[data-sitekey], .g-recaptcha, .h-captcha, '
        'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], '
        'iframe[src*="challenges.cloudflare.com"]'
    )
    if captcha.count():
        return ["The form includes verification that needs the applicant"]
    button = page.locator(SUBMIT_SELECTORS[site])
    if button.count() != 1 or not button.is_visible() or not button.is_enabled():
        return ["A supported submit control was not found"]
    text = button.evaluate("element => (element.value || element.textContent || '').trim()")
    if not re.fullmatch(r"submit application", text, flags=re.IGNORECASE):
        return ["The submit control has an unrecognized label"]
    selectors = list(FIELD_SELECTORS[site].values()) + [RESUME_SELECTORS[site]]
    core_fields = [selector for name, selector in FIELD_SELECTORS[site].items() if name != "phone"]
    return button.evaluate(
        """(button, configuration) => {
            const {selectors, coreFields} = configuration;
            const form = button.form || button.closest('form');
            if (!form) return ['The application form could not be identified'];
            const controls = [...form.elements];
            const blockers = [];
            for (const selector of coreFields) {
                const fields = form.querySelectorAll(selector);
                if (fields.length !== 1 || !fields[0].value.trim())
                    blockers.push('The applicant name or email is incomplete');
            }
            if (form.querySelector('[role="combobox"], [role="checkbox"], [contenteditable="true"]'))
                blockers.push('A custom application question needs the applicant');
            for (const element of controls) {
                if (element.disabled || ['hidden', 'submit', 'button', 'reset'].includes(element.type))
                    continue;
                if (!selectors.some(selector => element.matches(selector))) {
                    blockers.push('A custom application question needs the applicant');
                    continue;
                }
                if (element.type === 'file' && (!element.files || !element.files.length))
                    blockers.push('The CV upload is incomplete');
                if (element.required && !element.checkValidity())
                    blockers.push('A required field is incomplete or invalid');
            }
            if (!form.checkValidity()) blockers.push('The form has unresolved validation errors');
            return [...new Set(blockers)];
        }""",
        {"selectors": selectors, "coreFields": core_fields},
    )


def submission_confirmation(page, site: str) -> str | None:
    """Accept a visible receipt on the expected ATS, never a URL change alone."""
    try:
        current_site = application_site(page.url)
    except ValueError:
        return None
    if current_site != site:
        return None
    receipts = page.locator(
        '#application_confirmation, .application-confirmation, '
        '[data-qa="application-confirmation"], h1, h2, [role="status"]'
    )
    for index in range(receipts.count()):
        receipt = receipts.nth(index)
        if receipt.is_visible():
            text = receipt.inner_text().strip()
            if _RECEIPT_TEXT.search(text):
                return f"{page.url}\n{text[:1000]}"
    return None


def prepare_page(page, record: dict, *, submit: bool = False,
                 before_submit: Callable[[], None] | None = None) -> dict:
    """Prepare an already loaded page. This helper also supports fake-page tests."""
    if record.get("status") in {"submitted", "submission_unknown"}:
        return _result(record["status"], "This application requires no retry; inspect its saved outcome")
    if record.get("submission_attempted"):
        return _result("submission_unknown", "An earlier submission attempt requires verification; retry blocked")
    if record.get("status") not in {"ready", "opened"}:
        return _result(record.get("status") or "review", "This application has not passed preparation checks")
    requested_url = validate_application_url(str(record.get("application_url") or ""))
    site = application_site(requested_url)
    if site is None or application_site(page.url) != site:
        return _result("opened", "This employer form is open for manual completion")
    # A redirect to a different employer or vacancy must not receive a CV.
    requested = urlsplit(requested_url)
    current = urlsplit(page.url)
    requested_path = requested.path.rstrip("/").removesuffix("/apply")
    current_path = current.path.rstrip("/").removesuffix("/apply")
    if requested.hostname != current.hostname or requested_path != current_path:
        return _result("opened", "The destination changed; check the employer and vacancy manually")
    issues = fill_known_fields(page, record, site)
    if not submit:
        detail = "Known profile fields and CV prepared. Review and submit in the browser"
        return _result("opened", detail + (": " + "; ".join(issues) if issues else ""))
    blockers = issues + submission_blockers(page, site)
    if blockers:
        return _result("opened", "Manual completion needed: " + "; ".join(blockers))
    if before_submit:
        try:
            before_submit()
        except Exception:
            return _result("opened", "Could not save the submission attempt; complete the application after checking local storage")
    # If click raises, its effect is unknown. Never issue a second click.
    try:
        page.locator(SUBMIT_SELECTORS[site]).click(timeout=10000)
        for _ in range(20):
            confirmation = submission_confirmation(page, site)
            if confirmation:
                return _result("submitted", "The employer displayed a submission receipt", confirmation)
            page.wait_for_timeout(500)
    except Exception as exc:
        return _result("submission_unknown", f"Submission outcome needs verification ({type(exc).__name__})")
    return _result("submission_unknown", "Submit was attempted, but no receipt was detected. Verify before retrying")


def assist_application(record: dict, *, submit: bool = False, keep_open: bool = True,
                       before_submit: Callable[[], None] | None = None) -> dict:
    """Launch headed Chromium; the default waits until the applicant closes it.

    Playwright and its Chromium runtime are optional until this function is used.
    ``keep_open=False`` is for automated tests, not the interactive CLI workflow.
    """
    if record.get("status") in {"submitted", "submission_unknown"}:
        return _result(record["status"], "Submission is already recorded or needs verification; retry blocked")
    if record.get("submission_attempted"):
        return _result("submission_unknown", "An earlier submission attempt requires verification; retry blocked")
    if record.get("status") not in {"ready", "opened"}:
        return _result(record.get("status") or "review", "This application has not passed preparation checks")
    url = validate_application_url(str(record.get("application_url") or ""))
    configure_browser_runtime()
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Install Playwright and its browser first: pip install playwright; "
            "python -m playwright install chromium"
        ) from exc

    outcome = _result("opened", "The application browser was opened")
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=False)
        except PlaywrightError:
            return _result(
                "opened",
                "Chromium could not start. Install its runtime with python -m playwright install chromium "
                "and run from a graphical desktop. The saved application remains available",
            )
        context = browser.new_context(accept_downloads=False, service_workers="block")
        guard_public_requests(context)
        page = context.new_page()
        attempted = {"value": False}
        # Observe native form submissions, including a user's manual submit.
        page.expose_function("recordApplicationSubmit", lambda: attempted.update(value=True))
        page.add_init_script(
            "document.addEventListener('submit', () => { "
            "if (window.recordApplicationSubmit) window.recordApplicationSubmit(); "
            "}, true);"
        )
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            # Lever job URLs often initially show the description rather than the form.
            if application_site(url) == "lever" and not urlsplit(url).path.rstrip("/").endswith("/apply"):
                parts = urlsplit(url)
                url = parts._replace(path=parts.path.rstrip("/") + "/apply", fragment="").geturl()
                record = {**record, "application_url": url}
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
            outcome = prepare_page(page, record, submit=submit, before_submit=before_submit)
            site = application_site(url)
            if keep_open:
                while browser.is_connected() and context.pages:
                    for current_page in list(context.pages):
                        if (not current_page.is_closed() and site
                                and (attempted["value"] or outcome["status"] in {"submitted", "submission_unknown"})):
                            confirmation = submission_confirmation(current_page, site)
                            if confirmation:
                                outcome = _result("submitted", "The employer displayed a submission receipt", confirmation)
                    context.pages[0].wait_for_timeout(500)
        except Exception as exc:
            if not (page.is_closed() or not browser.is_connected()):
                outcome = _result(
                    "submission_unknown" if attempted["value"] else "opened",
                    f"Browser assistance stopped ({type(exc).__name__}); inspect the employer's outcome",
                )
        finally:
            if attempted["value"] and outcome["status"] not in {"submitted", "submission_unknown"}:
                outcome = _result("submission_unknown", "A form was submitted without a detected receipt. Verify before retrying")
            if browser.is_connected():
                browser.close()
    return outcome
