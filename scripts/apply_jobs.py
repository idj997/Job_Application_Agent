"""OpenAI application workflow: prepare packages, inspect them, and apply."""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import sys
from pathlib import Path

from app.applications.documents import read_cv
from app.applications.models import CandidateProfile
from app.applications.tracking import ApplicationLedger


DEFAULT_OUTPUT = Path("data/applications")


def load_profile(path: Path | None, cv_path: Path | None) -> tuple[CandidateProfile, Path]:
    if path:
        try:
            profile = CandidateProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load candidate profile {path}: use config/candidate_profile.example.json as the template.") from exc
    else:
        profile = CandidateProfile()
    if cv_path is None:
        if not profile.cv_path:
            raise ValueError("Provide --cv /path/to/master.pdf (TXT, Markdown and DOCX also work), or set cv_path in --profile.")
        cv_path = Path(profile.cv_path).expanduser()
        if not cv_path.is_absolute() and path:
            cv_path = path.parent / cv_path
    return profile, cv_path.expanduser().resolve()


def _read_jobs(paths: list[Path]) -> list[dict]:
    from app.applications.jobs import load_jobs
    jobs = []
    for path in paths:
        try:
            jobs.extend(load_jobs(path.parent, [path]))
        except (OSError, ValueError) as exc:
            print(f"Skipping invalid job file {path}: {type(exc).__name__}", file=sys.stderr)
    return jobs


def write_dashboard(ledger: ApplicationLedger, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in ledger.list():
        artifacts = record.get("artifacts", {})
        links = []
        for label in ("pdf", "docx", "assessment"):
            if artifacts.get(label):
                links.append(f'<a href="{html.escape(Path(artifacts[label]).resolve().as_uri(), quote=True)}">{label.upper()}</a>')
        reasons = " ".join(record.get("reasons", []))
        decision = record.get("decision", {})
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            record["job_key"][:12], record.get("company") or "", record.get("title") or "",
            record["status"], decision.get("score", "—"), record.get("cv_score", "—"),
        )) + f'<td>{" · ".join(links)}</td><td>{html.escape(reasons)}</td></tr>')
    page = output / "applications.html"
    page.write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Your applications</title><style>
body{font:15px system-ui,sans-serif;margin:40px;color:#182230;background:#f7f9fb}
h1{font-size:28px}table{width:100%;border-collapse:collapse;background:white}
th,td{text-align:left;padding:12px;border-bottom:1px solid #e2e8ef;vertical-align:top}
th{font-size:12px;text-transform:uppercase;color:#526275}a{color:#125baa}p{color:#526275}
</style><h1>Your applications</h1>
<p>Review the CV and assessment for each role. Ready means prepared; submitted means a confirmation has been recorded.</p>
<table><thead><tr><th>ID</th><th>Company</th><th>Role</th><th>Status</th><th>Match</th><th>CV quality</th><th>Documents</th><th>Notes</th></tr></thead><tbody>
""" + "\n".join(rows) + "</tbody></table></html>", encoding="utf-8")
    return page.resolve()


def _record(ledger: ApplicationLedger, prefix: str) -> dict:
    matches = [record for record in ledger.list() if record["job_key"].startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("Application ID must uniquely match an entry from the list command.")
    return matches[0]


def _print_record(record: dict) -> None:
    decision = record.get("decision", {})
    cached = " (saved result)" if record.get("cached") else ""
    print(f"{record['job_key'][:12]}  {record['status'].upper():<18} {record.get('company') or ''} — {record.get('title') or ''}{cached}")
    print(f"  Match: {decision.get('score', '—')}  CV quality: {record.get('cv_score', '—')}")
    for reason in record.get("reasons", []):
        print("  " + reason)
    if record.get("artifacts", {}).get("pdf"):
        print("  CV: " + record["artifacts"]["pdf"])


def prepare(args: argparse.Namespace) -> int:
    from app.applications.jobs import fetch_job_url, prepare_job
    from app.applications.workflow import ApplicationWorkflow
    from app.providers.openai import OpenAIProvider

    if args.limit < 1 or args.limit > 100:
        raise ValueError("--limit must be between 1 and 100; start with a small batch.")
    if not 0 <= args.match_threshold <= 100 or not 0 <= args.cv_threshold <= 100:
        raise ValueError("Score thresholds must be between 0 and 100.")
    if args.fetch and (args.job or args.url):
        raise ValueError("Use --fetch for discovery, or --job/--url for specific jobs.")
    if args.description_file and (len(args.job or []) + len(args.url or []) != 1 or args.fetch):
        raise ValueError("--description-file must accompany exactly one --job or --url.")
    profile, cv_path = load_profile(args.profile, args.cv)
    cv_text = read_cv(cv_path)
    if args.dry_run:
        paths = args.job or ([] if args.url else sorted(args.jobs_dir.glob("*.json")))
        print(f"CV loaded: {cv_path.name} ({len(cv_text):,} characters)")
        print(f"Jobs: {'discovery planned' if args.fetch else str(min(args.limit, len(paths) + len(args.url or []))) + ' selected'}")
        print(f"Maximum logical OpenAI calls: {args.limit * (1 + 2 * (args.max_revisions + 1))}; SDK may retry transient failures.")
        print("Constraints: " + json.dumps(profile.preferences.constraints(), ensure_ascii=False))
        print("Dry run complete. No network requests, model calls, or application writes.")
        return 0
    provider = OpenAIProvider(model=args.model)
    failures = 0
    if args.fetch:
        from app.scraper.cleaner import JobCleaner
        from app.scraper.storage import JobStorage
        from app.services.job_collection_service import JobCollectionService
        from scripts.scrape_jobs import build_configured_source

        query = args.query or next(iter(profile.preferences.target_roles), None)
        if not query:
            raise ValueError("Set --query or preferences.target_roles when fetching jobs.")
        storage = JobStorage(raw_dir=str(args.output / "raw_jobs"), processed_dir=str(args.jobs_dir))
        collector = JobCollectionService(storage, JobCleaner())
        for source_name in args.sources:
            try:
                source = build_configured_source(source_name, board_token=args.board_token, company_slug=args.company_slug)
            except Exception as exc:
                print(f"{source_name}: configuration failed ({type(exc).__name__}); check source credentials and options.", file=sys.stderr)
                failures += 1
                continue
            summary = collector.collect(source, query=query, location=args.location, limit=args.limit)
            print(f"{source_name}: {summary.saved} saved, {summary.duplicates} duplicates, {summary.failed} failed")
            if summary.source_error:
                print("  " + summary.source_error)
    paths = args.job or ([] if args.url else sorted(args.jobs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True))
    jobs = _read_jobs(paths)
    for url in (args.url or [])[:max(0, args.limit - len(jobs))]:
        try:
            jobs.append(fetch_job_url(url))
        except ValueError as exc:
            print(f"Could not load a job URL: {exc}", file=sys.stderr)
            failures += 1
    if not jobs:
        raise ValueError("No valid jobs found. Supply --job, --url, or --fetch --query, or run the existing scrape_jobs collector.")
    ledger = ApplicationLedger(args.output / "applications.db")
    workflow = ApplicationWorkflow(provider, ledger, args.output, match_threshold=args.match_threshold,
                                   cv_threshold=args.cv_threshold, max_revisions=args.max_revisions)
    for index, job in enumerate(jobs[:args.limit], start=1):
        print(f"Preparing {index}/{min(len(jobs), args.limit)}: {job.get('title') or 'Untitled role'}", flush=True)
        from app.applications.tracking import job_key
        try:
            previous = ledger.get(job_key(job))
            if previous and (previous["status"] in {"submitted", "submission_unknown"} or previous.get("submission_attempted")):
                record = {**previous, "cached": True}
            else:
                enriched = prepare_job(job, full_description_file=args.description_file, fetch_full=not args.no_enrich)
                record = workflow.prepare(enriched, profile, cv_text, retry=args.retry)
        except ValueError as exc:
            print(f"Could not prepare job: {exc}", file=sys.stderr)
            failures += 1
            continue
        _print_record(record)
        failures += record["status"] == "failed"
        write_dashboard(ledger, args.output)
    print("\nApplication dashboard: " + str((args.output / "applications.html").resolve()))
    print("OpenAI usage: " + json.dumps(provider.usage))
    return 1 if failures else 0


def open_application(args: argparse.Namespace, ledger: ApplicationLedger) -> int:
    from app.applications.browser import assist_application, validate_application_url
    record = _record(ledger, args.id)
    if record["status"] != "ready" or record.get("submission_attempted"):
        raise ValueError("Only a ready application can be opened. Verify any earlier attempt and record its outcome before retrying.")
    validate_application_url(record["application_url"])
    if not Path(record.get("artifacts", {}).get("pdf", "")).is_file():
        raise ValueError("The prepared CV PDF is missing. Recreate the application package first.")
    # Any visible browser session can lead to manual submission. Persist before opening,
    # so a killed process cannot later assume it is safe to apply again.
    record = ledger.claim_open(record["job_key"])
    print("Complete the application in the browser, then close the window. Its outcome will be recorded.", flush=True)
    result = assist_application(record, submit=args.submit, before_submit=lambda: None)
    print(result["detail"])
    if result["status"] == "submitted" and result.get("confirmation"):
        ledger.mark_submitted(record["job_key"], result["confirmation"])
        print("Submission confirmed and saved.")
    else:
        print("No confirmed receipt saved. Check the employer's outcome; use mark-submitted or resolve-not-submitted before another attempt.")
    write_dashboard(ledger, args.output)
    return 0


def doctor() -> int:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    missing = []
    for name in ("openai", "pydantic", "dotenv", "bs4", "pypdf", "docx", "reportlab", "playwright"):
        present = importlib.util.find_spec(name) is not None
        print(f"{name}: {'installed' if present else 'missing'}")
        if not present:
            missing.append(name)
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        print(f"{name}: {'configured' if os.getenv(name, '').strip() else 'not set'}")
    print("No API request was made.")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and track job applications using OpenAI.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Private application packages and ledger directory.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("prepare", help="Match jobs, tailor your CV, evaluate it and save application packages.")
    run.add_argument("--cv", type=Path)
    run.add_argument("--profile", type=Path)
    run.add_argument("--model", help="Responses/Structured Outputs model; defaults to OPENAI_MODEL.")
    run.add_argument("--jobs-dir", type=Path, default=Path("data/processed_jobs"))
    run.add_argument("--job", type=Path, action="append", help="One processed job JSON file (repeatable).")
    run.add_argument("--url", action="append", help="One employer job URL (repeatable).")
    run.add_argument("--fetch", action="store_true", help="Collect new jobs using existing sources before preparing.")
    run.add_argument("--sources", nargs="+", default=["adzuna"], choices=["adzuna", "arbeitnow", "greenhouse", "lever", "the_muse", "remotive"])
    run.add_argument("--query")
    run.add_argument("--location", default="United Kingdom")
    run.add_argument("--board-token")
    run.add_argument("--company-slug")
    run.add_argument("--limit", type=int, default=5)
    run.add_argument("--description-file", type=Path, help="Full employer description for one --job or --url.")
    run.add_argument("--no-enrich", action="store_true", help="Leave summary-only jobs for review without network enrichment.")
    run.add_argument("--match-threshold", type=int, default=70)
    run.add_argument("--cv-threshold", type=int, default=75)
    run.add_argument("--max-revisions", type=int, choices=[0, 1, 2], default=1)
    run.add_argument("--retry", action="store_true", help="Reassess saved preparation; never retries submitted/uncertain jobs.")
    run.add_argument("--dry-run", action="store_true", help="Validate CV and preview work without network or model calls.")
    sub.add_parser("doctor", help="Check dependencies and configuration without showing secrets.")
    sub.add_parser("list", help="Show application states and scores.")
    show = sub.add_parser("show", help="Show one saved assessment and its files.")
    show.add_argument("id")
    browser = sub.add_parser("open", help="Open a ready application and assist with supported employer forms.")
    browser.add_argument("id")
    browser.add_argument("--submit", action="store_true", help="Submit a recognizable complete form; other forms stay open for you.")
    submitted = sub.add_parser("mark-submitted", help="Record a submission you verified yourself.")
    submitted.add_argument("id")
    submitted.add_argument("--confirmation", required=True, help="Receipt/reference or confirmation details you observed.")
    reset = sub.add_parser("resolve-not-submitted", help="Allow retry only after you checked that an earlier session did not submit.")
    reset.add_argument("id")
    reset.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "prepare":
            return prepare(args)
        ledger = ApplicationLedger(args.output / "applications.db")
        if args.command == "list":
            for record in ledger.list():
                _print_record(record)
        elif args.command == "show":
            print(json.dumps(_record(ledger, args.id), ensure_ascii=False, indent=2))
        elif args.command == "open":
            return open_application(args, ledger)
        elif args.command == "mark-submitted":
            ledger.mark_submitted(_record(ledger, args.id)["job_key"], args.confirmation)
            print("Submission confirmation recorded.")
        elif args.command == "resolve-not-submitted":
            ledger.resolve_not_submitted(_record(ledger, args.id)["job_key"], args.note)
            print("Checked outcome recorded. Application is ready for another attempt.")
        write_dashboard(ledger, args.output)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
