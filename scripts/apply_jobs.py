"""Local or OpenAI application workflow: prepare packages, inspect them, and apply."""
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
MAX_SELECTION_PLAN_BYTES = 250_000


def selected_provider(requested: str | None) -> str:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    name = (requested or os.getenv("APPLICATION_PROVIDER", "openai")).strip().lower()
    if name not in {"openai", "ollama"}:
        raise ValueError("APPLICATION_PROVIDER must be openai or ollama, or pass --provider explicitly.")
    return name


def selected_matcher(requested: str | None) -> str:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    name = (requested or os.getenv("APPLICATION_MATCHER", "classifier")).strip().lower()
    if name not in {"classifier", "central", "llm"}:
        raise ValueError("APPLICATION_MATCHER must be classifier, central or llm, or pass --matcher explicitly.")
    return name


def build_matcher(name: str):
    if name == "llm":
        return None
    if name == "central":
        from app.applications.central_requirements import CentralMatcher
        return CentralMatcher()
    from app.applications.classifier_matching import ClassifierMatcher
    return ClassifierMatcher()


def build_provider(name: str, model: str | None):
    # Never construct or fall back to the paid provider in local mode.
    if name == "ollama":
        from app.providers.ollama_provider import OllamaProvider
        return OllamaProvider(model=model)
    from app.providers.openai import OpenAIProvider
    return OpenAIProvider(model=model)


def selected_critic_model(requested: str | None, *, disabled: bool = False) -> str | None:
    """Resolve the opt-in critic without importing a model SDK or reading weights."""
    if disabled:
        return None
    if requested is not None and not requested.strip():
        raise ValueError("--critic-model needs an installed local model name; use --no-critic to disable it.")
    return (requested if requested is not None else os.getenv("CV_CRITIC_MODEL", "")).strip() or None


def build_local_role_provider(model: str, role: str):
    # Metadata, clients and model checks are deliberately deferred until generation.
    from app.providers.local_critic_runtime import build_local_role_provider as build
    return build(model=model, role=role)


def local_role_config(model: str, role: str) -> dict:
    from app.providers.local_critic_runtime import local_role_config as config
    return config(model=model, role=role)


class DeferredGenerationProvider:
    """Keep classifier-only decisions independent of generation SDKs and keys."""

    def __init__(self, name: str, model: str | None, *, role: str | None = None):
        if role not in {None, "writer", "critic"} or (role is not None and name != "ollama"):
            raise ValueError("Separate local generation roles must be an Ollama writer or critic.")
        self.name = name
        self.role = role
        self._requested_model = model
        setting = "OLLAMA_MODEL" if name == "ollama" else "OPENAI_MODEL"
        fallback = "qwen3:8b" if name == "ollama" else ""
        self._model = (model if model is not None else (os.getenv(setting) or fallback)).strip() or None
        self._role_config = local_role_config(self._model, role) if role is not None else None
        self._provider = None

    @property
    def fingerprint(self):
        return dict(self._role_config) if self._role_config is not None else None

    @property
    def model_digest(self):
        return getattr(self._provider, "model_digest", None) if self._provider is not None else None

    @property
    def budget_diagnostics(self):
        return getattr(self._provider, "budget_diagnostics", []) if self._provider is not None else []

    @property
    def model(self):
        return self._provider.model if self._provider is not None else self._model

    @property
    def usage(self) -> dict[str, int]:
        if self._provider is not None:
            return self._provider.usage
        return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _load(self):
        if self._provider is None:
            if self.role is not None:
                # The ledger fingerprint was captured before matching. Do not
                # silently use changed environment settings when loading later.
                try:
                    unchanged = local_role_config(self._model, self.role) == self._role_config
                except ValueError:
                    unchanged = False
                if not unchanged:
                    from app.providers.base import ModelProviderError
                    raise ModelProviderError(
                        "Local generation configuration changed after this preparation started. "
                        "Start a fresh preparation; no model request was sent."
                    )
                self._provider = build_local_role_provider(self._model, self.role)
            else:
                self._provider = build_provider(self.name, self._requested_model)
        return self._provider

    def generate(self, prompt: str) -> str:
        return self._load().generate(prompt)

    def generate_structured(self, prompt: str, schema: type, system_prompt: str | None = None):
        return self._load().generate_structured(prompt=prompt, schema=schema, system_prompt=system_prompt)


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


def _load_selection_plan(path: Path, cv_text: str, job_path: Path):
    """Validate a reviewed plan and exact saved inputs before any model or writes.

    Errors intentionally omit validation payloads: Pydantic errors can otherwise
    echo private CV/plan text into the terminal.
    """
    from app.applications.cv_evidence import build_cv_sources
    from app.applications.cv_selection import SelectionPlan, compile_selection_plan
    from app.applications.jobs import load_jobs, prepare_job

    try:
        if path.stat().st_size > MAX_SELECTION_PLAN_BYTES:
            raise ValueError("oversized plan")
        plan = SelectionPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ValueError(
            "Could not load --selection-plan: provide valid UTF-8 SelectionPlan JSON of at most 250 KB."
        ) from None
    try:
        jobs = load_jobs(job_path.parent, [job_path])
        if len(jobs) != 1:
            raise ValueError("not one job")
        job = prepare_job(jobs[0], fetch_full=False)
        compiled = compile_selection_plan(plan, build_cv_sources(cv_text), job)
    except (OSError, UnicodeError, ValueError):
        raise ValueError(
            "Selection plan validation failed: check the saved job, source/job fingerprints, "
            "source IDs and selection/length limits. No model request or application write was made."
        ) from None
    return plan, job, compiled.summary


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

    if args.limit < 1 or args.limit > 100:
        raise ValueError("--limit must be between 1 and 100; start with a small batch.")
    if not 0 <= args.match_threshold <= 100 or not 0 <= args.cv_threshold <= 100:
        raise ValueError("Score thresholds must be between 0 and 100.")
    if args.fetch and (args.job or args.url):
        raise ValueError("Use --fetch for discovery, or --job/--url for specific jobs.")
    if args.description_file and (len(args.job or []) + len(args.url or []) != 1 or args.fetch):
        raise ValueError("--description-file must accompany exactly one --job or --url.")
    if args.entry_by_entry and (args.selection_plan is None or args.max_revisions != 0):
        raise ValueError("--entry-by-entry requires --selection-plan and --max-revisions 0; critic-driven revisions are disabled.")
    if args.selection_plan is not None and (
            len(args.job or []) != 1 or args.limit != 1 or args.url or args.fetch
            or args.description_file or not args.no_enrich or args.match_only):
        raise ValueError(
            "A selection plan requires exactly one explicit --job, --limit 1 and --no-enrich; "
            "no --url, --fetch, --description-file or --match-only."
        )
    manual_note = args.manual_approval_note
    if manual_note is not None:
        if not manual_note.strip() or len(manual_note.strip()) > 2000:
            raise ValueError("--manual-approval-note needs a nonempty note of at most 2,000 characters.")
        if (len(args.job or []) != 1 or args.limit != 1 or args.url or args.fetch
                or args.description_file or not args.no_enrich or args.match_only):
            raise ValueError("Manual tailoring requires exactly one explicit --job, --limit 1 and --no-enrich; no --url, --fetch, --description-file or --match-only.")
    profile, cv_path = load_profile(args.profile, args.cv)
    cv_text = read_cv(cv_path)
    selection_plan = None
    selection_job = None
    selection_summary = None
    if args.selection_plan is not None:
        selection_plan, selection_job, selection_summary = _load_selection_plan(
            args.selection_plan, cv_text, args.job[0],
        )
    provider_name = selected_provider(args.provider)
    matcher_name = selected_matcher(args.matcher)
    critic_model = selected_critic_model(args.critic_model, disabled=args.no_critic)
    writer_model = DeferredGenerationProvider(provider_name, args.model).model
    if critic_model is not None and critic_model == writer_model:
        raise ValueError("The critic model must differ from the writer model; use --no-critic for the existing self-audit mode.")
    if manual_note is not None and matcher_name != "classifier":
        raise ValueError("Manual tailoring requires --matcher classifier and an existing classifier REVIEW.")
    display_name = "Ollama (local)" if provider_name == "ollama" else "OpenAI"
    if args.entry_by_entry:
        print("Entry-by-entry writing: the writer sees only the source lines assigned to each rewrite slot; the full source catalog is retained for audit.")
        print("Retry limit: at most 2 attempts per rewrite slot (one local validation retry); no critic-driven revisions.")
    if args.dry_run:
        paths = args.job or ([] if args.url else sorted(args.jobs_dir.glob("*.json")))
        print(f"CV loaded: {cv_path.name} ({len(cv_text):,} characters)")
        print(f"Jobs: {'discovery planned' if args.fetch else str(min(args.limit, len(paths) + len(args.url or []))) + ' selected'}")
        print(f"Provider: {display_name}")
        print(f"Matcher: {matcher_name}")
        if selection_summary is not None:
            print(
                "Reviewed selection plan: "
                f"{selection_summary['slot_count']} slots "
                f"({selection_summary['fixed_slots']} fixed, {selection_summary['rewrite_slots']} rewrite); "
                f"{selection_summary['noncontact_entries']} non-contact entries; "
                f"maximum {selection_summary['word_budget']} words."
            )
            print("Selection plan source/job fingerprints and bounds validated against these exact saved inputs.")
        matching_calls = 1 if matcher_name in {"llm", "central"} else 0
        drafts = 0 if args.match_only else args.max_revisions + 1
        writer_calls = 2 * selection_summary["rewrite_slots"] if args.entry_by_entry else drafts
        audit_calls = 1 if args.entry_by_entry else drafts
        generation_calls = writer_calls + (audit_calls if critic_model is None else 0)
        retry_note = "; SDK may retry transient failures." if provider_name == "openai" else "; no OpenAI requests."
        budget_label = f"writer provider calls ({display_name})" if critic_model is not None else f"{display_name} calls"
        print(f"Maximum logical {budget_label}: {args.limit * (matching_calls + generation_calls)}{retry_note}")
        if args.entry_by_entry and critic_model is None:
            print(f"Maximum logical writer calls: {args.limit * writer_calls}; matching calls: {args.limit * matching_calls}")
            print(f"Maximum logical writer self-audit calls: {args.limit * audit_calls}; full-CV audit after all entries pass validation.")
        if critic_model is not None:
            print(f"CV critic: Ollama (local); model: {critic_model}")
            print(f"Maximum logical writer calls: {args.limit * writer_calls}; matching calls: {args.limit * matching_calls}")
            print(f"Maximum logical critic calls: {args.limit * audit_calls}; replaces writer self-audits, no cloud fallback.")
            print("Writer and critic run sequentially; local models unload after each request.")
        if args.match_only:
            print("Match-only: save decisions without drafting CVs or opening applications.")
        if manual_note is not None:
            print("Manual tailoring only: the saved classifier REVIEW must match these exact inputs; drafts cannot open applications.")
        if matcher_name == "classifier":
            print("Classifier matching uses local cached NLI weights; dry run does not load them.")
        elif matcher_name == "central":
            print("Central matching uses the local cached NLI baseline plus one local Ollama central-requirement selection per job.")
        print("Constraints: " + json.dumps(profile.preferences.constraints(), ensure_ascii=False))
        print("Dry run complete. No network requests, model calls, or application writes.")
        return 0
    matcher = build_matcher(matcher_name)
    provider = (DeferredGenerationProvider(provider_name, args.model, role="writer")
                if critic_model is not None and provider_name == "ollama" else
                DeferredGenerationProvider(provider_name, args.model) if matcher_name == "classifier"
                else build_provider(provider_name, args.model))
    critic = DeferredGenerationProvider("ollama", critic_model, role="critic") if critic_model is not None else None
    print(f"Matcher: {matcher_name}", flush=True)
    print(f"Generation provider: {display_name}; model: {provider.model or 'not configured (needed only for CV generation)'}", flush=True)
    if critic is not None:
        print(f"CV critic: Ollama (local); model: {critic.model}; sequential draft/critique, no self-audit fallback", flush=True)
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
    jobs = [selection_job] if selection_job is not None else _read_jobs(paths)
    if manual_note is not None and len(jobs) != 1:
        raise ValueError("The manually approved --job file must contain exactly one valid job.")
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
                                   cv_threshold=args.cv_threshold, max_revisions=args.max_revisions,
                                   matcher=matcher, **({"critic_provider": critic} if critic is not None else {}),
                                   **({"selection_plan": selection_plan} if selection_plan is not None else {}),
                                   **({"entry_by_entry": True} if args.entry_by_entry else {}))
    for index, job in enumerate(jobs[:args.limit], start=1):
        print(f"Preparing {index}/{min(len(jobs), args.limit)}: {job.get('title') or 'Untitled role'}", flush=True)
        from app.applications.tracking import job_key
        try:
            previous = ledger.get(job_key(job))
            if manual_note is None and previous and (previous["status"] in {"submitted", "submission_unknown"} or previous.get("submission_attempted")):
                record = {**previous, "cached": True}
            else:
                enriched = (selection_job if selection_job is not None else
                            prepare_job(job, full_description_file=args.description_file, fetch_full=not args.no_enrich))
                manual_options = {"manual_approval_note": manual_note} if manual_note is not None else {}
                record = workflow.prepare(enriched, profile, cv_text, retry=args.retry, match_only=args.match_only, **manual_options)
        except ValueError as exc:
            print(f"Could not prepare job: {exc}", file=sys.stderr)
            failures += 1
            continue
        _print_record(record)
        failures += record["status"] == "failed"
        write_dashboard(ledger, args.output)
    print("\nApplication dashboard: " + str((args.output / "applications.html").resolve()))
    print(display_name + " usage: " + json.dumps(provider.usage))
    if critic is not None:
        print("Ollama critic usage: " + json.dumps(critic.usage))
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


def doctor(provider: str | None = None, matcher: str | None = None, *,
           model: str | None = None, critic_model: str | None = None, no_critic: bool = False) -> int:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    name = selected_provider(provider)
    matcher_name = selected_matcher(matcher)
    critic_model = selected_critic_model(critic_model, disabled=no_critic)
    if critic_model is not None and critic_model == DeferredGenerationProvider(name, model).model:
        raise ValueError("The critic model must differ from the writer model; use --no-critic for the existing self-audit mode.")
    print("Provider: " + name)
    print("Matcher: " + matcher_name)
    missing = []
    modules = [name, "pydantic", "dotenv", "bs4", "pypdf", "docx", "reportlab", "playwright"]
    if critic_model is not None and "ollama" not in modules:
        modules.append("ollama")
    if matcher_name in {"classifier", "central"}:
        modules.extend(["torch", "transformers"])
    if matcher_name == "central" and "ollama" not in modules:
        modules.append("ollama")
    for module in modules:
        present = importlib.util.find_spec(module) is not None
        print(f"{module}: {'installed' if present else 'missing'}")
        if not present:
            missing.append(module)
    if name == "openai":
        for setting in ("OPENAI_API_KEY", "OPENAI_MODEL"):
            print(f"{setting}: {'configured' if os.getenv(setting, '').strip() else 'not set'}")
    else:
        print("OLLAMA_MODEL: " + (os.getenv("OLLAMA_MODEL", "").strip() or "qwen3:8b (default)"))
        print("OLLAMA_BASE_URL: " + ("configured" if os.getenv("OLLAMA_BASE_URL", "").strip() else "localhost:11434 (default)"))
        print("OPENAI_API_KEY: not required for local mode")
        print("Run ollama list to verify the local model is installed; doctor does not contact the server.")
    if critic_model is not None:
        print("CV critic: Ollama (local); model: " + critic_model)
        settings = DeferredGenerationProvider("ollama", critic_model, role="critic").fingerprint
        print("CV_CRITIC_NUM_CTX: " + str(settings["num_ctx"]))
        print("CV_CRITIC_NUM_PREDICT: " + str(settings["num_predict"]))
        print("CV_CRITIC_TIMEOUT_SECONDS: " + str(settings["timeout_seconds"]))
        print("Critic weights: must already be installed locally; doctor does not contact Ollama or verify model files.")
        print("Critic scores are advisory CV quality estimates, not validated ATS scores.")
    if matcher_name in {"classifier", "central"}:
        sentencepiece_present = importlib.util.find_spec("sentencepiece") is not None
        print("sentencepiece: " + ("installed" if sentencepiece_present else "optional; only needed for some tokenizer formats"))
        print("Matching baseline: pretrained NLI plus rules; no validated custom CV-matching checkpoint is supplied.")
        if matcher_name == "central":
            print("Central policy: local Ollama selects 2-4 central job requirements only; NLI/rules still verify CV evidence.")
        print("REQUIREMENT_NLI_MODEL: " + (os.getenv("REQUIREMENT_NLI_MODEL", "").strip() or "cross-encoder/nli-deberta-v3-small (default)"))
        print("REQUIREMENT_NLI_REVISION: " + ("configured" if os.getenv("REQUIREMENT_NLI_REVISION", "").strip() else "default cached revision"))
        print("REQUIREMENT_NLI_DEVICE: " + (os.getenv("REQUIREMENT_NLI_DEVICE", "").strip() or "cpu (default)"))
        print("Classifier weights: local cache only; doctor does not load or verify model files.")
        print("Install classifier libraries with: .venv/bin/python -m pip install -r requirements-classifier.txt")
    print("No API or model request was made.")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and track job applications using local Ollama or OpenAI.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Private application packages and ledger directory.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("prepare", help="Match jobs, tailor your CV, evaluate it and save application packages.")
    run.add_argument("--cv", type=Path)
    run.add_argument("--profile", type=Path)
    run.add_argument("--provider", choices=["openai", "ollama"], help="Model backend; defaults to APPLICATION_PROVIDER, then openai.")
    run.add_argument("--matcher", choices=["classifier", "central", "llm"], help="Matching backend; defaults to APPLICATION_MATCHER, then classifier. llm explicitly opts into model-generated matching.")
    run.add_argument("--match-only", action="store_true", help="Save match decisions without CV generation; classifier mode does not construct a generation provider.")
    run.add_argument("--manual-approval-note", help="Explicitly approve one saved classifier REVIEW for tailoring only. Requires one --job, --limit 1 and --no-enrich; never makes the draft browser-ready.")
    run.add_argument("--selection-plan", type=Path, help="Reviewed source/job-bound CV selection JSON. Requires one saved --job, --limit 1 and --no-enrich; validates selected entries and word limits before generation.")
    run.add_argument("--entry-by-entry", action="store_true", help="Write one source-isolated rewrite slot per request, with at most one local validation retry per slot. Requires --selection-plan and --max-revisions 0; full-CV audit remains mandatory.")
    run.add_argument("--model", help="Model override; otherwise uses OPENAI_MODEL or OLLAMA_MODEL for the selected provider.")
    critique = run.add_mutually_exclusive_group()
    critique.add_argument("--critic-model", help="Use a separate installed local Ollama model to critique each draft; defaults to CV_CRITIC_MODEL (disabled if unset).")
    critique.add_argument("--no-critic", action="store_true", help="Disable CV_CRITIC_MODEL and retain the writer's existing self-audit mode.")
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
    health = sub.add_parser("doctor", help="Check selected-provider dependencies and configuration without showing secrets.")
    health.add_argument("--provider", choices=["openai", "ollama"])
    health.add_argument("--matcher", choices=["classifier", "central", "llm"])
    health.add_argument("--model", help="Writer model override for configuration checks.")
    health_critique = health.add_mutually_exclusive_group()
    health_critique.add_argument("--critic-model", help="Inspect separate local critic configuration without contacting Ollama.")
    health_critique.add_argument("--no-critic", action="store_true", help="Ignore CV_CRITIC_MODEL for these checks.")
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
            return doctor(args.provider, args.matcher, model=args.model,
                          critic_model=args.critic_model, no_critic=args.no_critic)
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
