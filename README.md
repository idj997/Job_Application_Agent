# Job Application Agent

Collect jobs, match them against your master CV, generate a tailored CV, evaluate
it, and prepare an application. The OpenAI workflow runs alongside the existing
Ollama experiments and requirement-classifier dataset tools.

## Choose a workflow

| Capability | OpenAI application mode | Local / Ollama research mode |
| --- | --- | --- |
| Collect, normalize and deduplicate jobs | Yes | Yes, using the same collectors |
| Extract structured job requirements | OpenAI assessments | Ollama extraction and local model experiments |
| Compare jobs with a master CV | Yes | Not connected to a local application command |
| Tailor and evaluate PDF/DOCX CVs | Yes | Not connected to a local application command |
| Assist with application forms and track outcomes | Yes, supported forms | Not connected |
| Review requirement annotations and build datasets | Shared scripts, independently of OpenAI | Yes |
| Fine-tune a custom requirement classifier | No training runner provided | Dataset preparation and model experiments only |
| Model execution | OpenAI API | Ollama server / local Hugging Face models |

Use **OpenAI mode** to prepare job applications now. Use **local mode** for job
extraction, requirement analysis and developing your classifier dataset. The
local workflow has working components but is not a second end-to-end application
agent. There is currently no `--provider ollama` switch on `app.py prepare`.

Detailed guides:

- [OpenAI mode: setup, preferences, preparing CVs and applying](docs/OPENAI_MODE.md)
- [Local mode: Ollama extraction, model experiments and annotation datasets](docs/LOCAL_MODE.md)
- [Role catalog and collection batches](config/README.md)
- [Annotation schema and review conventions](data/labelled/README.md)
- [Local data layout and backups](data/README.md)

## Prerequisites

The Python code requires Python 3.10 or newer. Use a Python version supported by
your selected dependencies; local PyTorch/CUDA packages may have narrower wheel
availability. The current checkout was tested with Python 3.14. Linux/macOS shell
commands below assume the repository root; on Windows use `.venv\Scripts\python`
for the Python executable and the corresponding PowerShell syntax.

OpenAI mode needs an API key and model access, plus a master CV with extractable
text. A graphical desktop is needed only when opening application forms. Local
mode needs the Ollama service and a downloaded model for extraction; classifier
experiments additionally use PyTorch and Hugging Face model weights. Job
discovery and first-time model downloads require network access in either mode.

## Start applying

Clone the repository and create an isolated Python environment:

```bash
git clone git@github.com:idj997/Job_Application_Agent.git
cd Job_Application_Agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-openai.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/.venv/playwright-browsers" .venv/bin/python -m playwright install chromium
```

If `.env` does not exist, copy `.env.example` to `.env`. Add the following entries
to your `.env`, keeping your existing source API keys:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_OUTPUT_TOKENS=6000
```

The model is configurable using `--model` or `OPENAI_MODEL`. The example uses a
model supporting Responses and Structured Outputs; your API project must have
access. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-5-mini).

Save your master CV under `data/cv/` (PDF, DOCX, UTF-8 text or Markdown). For
matching preferences and browser autofill, copy
`config/candidate_profile.example.json` to `config/candidate_profile.json` and
fill in your real details. `cv_path` resolves relative to the profile file;
`--cv` overrides it and resolves relative to your shell directory. No CV details,
work authorization or personal preferences are inferred from the example.

```bash
.venv/bin/python app.py doctor
.venv/bin/python app.py prepare --profile config/candidate_profile.json --limit 3 --dry-run
.venv/bin/python app.py prepare --profile config/candidate_profile.json --limit 3
```

Without a profile, use `--cv /path/to/master.pdf`. That can match and tailor from
your CV, but browser autofill needs the explicit contact fields in the profile.
Greenhouse uses `first_name` and `last_name`; Lever uses `name`.

Preparation reads the newest stored jobs from `data/processed_jobs/`. To target a
specific employer page or stored job:

```bash
.venv/bin/python app.py prepare --cv data/cv/master.pdf --url "https://jobs.lever.co/company/job-id" --limit 1
.venv/bin/python app.py prepare --cv data/cv/master.pdf --job data/processed_jobs/JOB_ID.json --limit 1
```

To fetch jobs and prepare packages in one run:

```bash
.venv/bin/python app.py prepare --profile config/candidate_profile.json --fetch --sources adzuna --query "data engineer" --location "United Kingdom" --limit 5
```

Adzuna needs `ADZUNA_APP_ID` and `ADZUNA_APP_KEY`. Other collectors are available
via `--sources`; Greenhouse needs `--board-token`, Lever needs `--company-slug`.
The existing `scripts.scrape_jobs` and `scripts.collect_role_corpus` commands also
continue to work. Discovery stores jobs in the shared input directory; the
preparation batch selects the newest available records, which can include older
stored jobs if discovery returns fewer new ones than the batch limit.

Adzuna descriptions are summaries. The workflow tries to retrieve the full
employer description before matching. If the employer page requires JavaScript,
login or blocks retrieval, save the complete description as text and pass it for
exactly one job:

```bash
.venv/bin/python app.py prepare --cv data/cv/master.pdf --job data/processed_jobs/JOB_ID.json --description-file data/cv/job-description.txt --limit 1
```

`--no-enrich` disables that network retrieval; summary-only jobs become REVIEW
without a model call. A dry run performs no network or model calls and writes no
application artifacts. Scanned PDFs need OCR or a text/DOCX source CV.

## Review and submit

Preparation writes `data/applications/applications.html` with links to each job's
CV and assessment. Each tailored package includes `cv.pdf`, `cv.docx`, `cv.md`
and `assessment.json`. A failed CV evaluation can still produce a review draft;
only a **ready** package can enter the application browser.

```bash
.venv/bin/python app.py list
.venv/bin/python app.py show APPLICATION_ID
.venv/bin/python app.py open APPLICATION_ID
```

Use the short application ID printed by `list`. `open` launches a visible browser,
fills recognizable Greenhouse/Lever contact fields and uploads the generated
PDF. Review the form, complete additional questions, submit, then close the
browser. Other employer sites open for manual completion.

For recognizable, complete simple forms, the explicit submission option is:

```bash
.venv/bin/python app.py open APPLICATION_ID --submit
```

This clicks submit only when the supported control is identified, expected
fields are populated, and no unresolved custom question or verification remains.
Changed layouts, embedded forms, CAPTCHAs and unsupported sites need manual
completion. LinkedIn Easy Apply automation is not implemented. A graphical
desktop is needed for the visible browser.

An application is recorded as **submitted** only after detecting an employer
receipt or after you supply a confirmation you checked yourself:

```bash
.venv/bin/python app.py mark-submitted APPLICATION_ID --confirmation "Employer receipt/reference I verified"
```

Opening a session is recorded before the browser launches. This prevents another
process or a later restart from blindly submitting the same job. If you closed
the browser without applying, or it could not start, verify that no application
was submitted and explicitly release the saved attempt:

```bash
.venv/bin/python app.py resolve-not-submitted APPLICATION_ID --note "Checked the employer portal: no application was submitted"
```

Confirmed submissions cannot be reset. Job identity uses employer/requisition
when available, otherwise normalized application/source URLs. Different boards
can still represent the same vacancy with different identifiers; review those
before applying. The application ledger is intended for one candidate.

## Matching and CV evaluation

Each extracted requirement has a job quote, CV evidence, importance and match
status. Verbatim evidence is checked in code. Requirements accepting alternatives
(for example Python **or** Java) are assessed as a group in the prompt.

The match score is a weighted fraction of requirements: CORE 5, IMPORTANT 3,
PREFERRED 1; MET gets full credit, PARTIAL half, others zero. Explicit failed
constraints produce SKIP. Unknown constraints, unresolved mandatory requirements
or unverified evidence produce REVIEW. The model must also recommend APPLY and
the score must reach `--match-threshold` (default 70) before CV generation.

Tailoring preserves the master CV as the source of facts. Every generated entry
carries source quotations. A separate model call evaluates the complete document
for unsupported claims, missing critical information, requirement coverage and
clarity. The CV score is 70% coverage and 30% clarity; default threshold is 75.
Factual issues block ready status regardless of the numerical score. These are
internal rubric scores, not an employer's ATS score or a guarantee of selection.
Model audits can still make mistakes, so read the generated CV before using it.

By default a job allows one initial CV and one revision (`--max-revisions 0..2`).
One successful first-pass package normally makes three model calls: matching,
tailoring and evaluation. Each revision adds two calls. `--limit` defaults to 5;
reported usage includes input/output tokens, but does not estimate account bills.
The SDK can retry transient failures. Identical completed preparations are reused;
`--retry` reassesses a package but cannot retry an opened or submitted application.

CV text and job data are sent to OpenAI when you run preparation. Requests set
`store=False`; this is a response-storage setting, not a claim of zero retention.
The local ledger and artifacts contain personal data and are excluded by
`.gitignore`. The workflow treats job descriptions as untrusted data.
[Official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Verification

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/scraper tests/datasets tests/applications -q
```

The suite uses mocked OpenAI responses and local browser fixtures; it makes no
paid API calls or real applications. Browser fixture tests skip when Chromium
cannot launch. The existing top-level `tests/test_model.py` and classifier scripts
load local/downloadable models at import time, so they are outside this offline
application test command.

## Using local mode

The local mode uses the existing Python components directly. The
[complete local guide](docs/LOCAL_MODE.md) includes installation, structured
extraction, model experiments and the complete annotation workflow.

Install the lightweight local dependencies, install/start Ollama using its
[official instructions](https://docs.ollama.com/quickstart), and download the
model used by the current provider:

```bash
.venv/bin/python -m pip install -r requirements-local.txt
ollama pull qwen3:8b
```

With the Ollama service running on `http://localhost:11434`, run the included
extraction example:

```bash
.venv/bin/python -m tests.test_model
```

It prints normalized job fields extracted from a built-in sample description.
It does not compare a CV or submit an application. To extract a collected job,
use `JobExtractor(OllamaProvider(...)).extract(job["description"])`; the full guide
contains a runnable example. The provider defaults to `qwen3:8b` and localhost.
`OLLAMA_MODEL` and `OLLAMA_BASE_URL` in `.env` are not automatically read by this
legacy provider; pass your chosen values to its constructor as shown in the guide.

The shared dataset workflow can run without invoking any model:

```bash
.venv/bin/python -m scripts.scrape_jobs --sources adzuna --query "data engineer" --location "United Kingdom" --limit-per-source 10
.venv/bin/python -m scripts.process_jobs
```

Review `data/labelled/requirement_candidates.json`, set each valid record's
`annotation_status` to `approved`, and correct `reviewed_label` where necessary.
Then validate, promote and build the dataset:

```bash
.venv/bin/python -m scripts.process_jobs --validate-reviews data/labelled/requirement_candidates.json
.venv/bin/python -m scripts.process_jobs --promote-approved data/labelled/requirement_candidates.json
.venv/bin/python -m scripts.report_dataset_coverage
.venv/bin/python -m scripts.build_dataset
```

The dataset builder requires reviewed examples for all five labels and enough
distinct jobs for its split. Follow the local guide for alternative requirement
groups, counterfactual negatives and coverage targets. The builder generates
training/evaluation **data**; it does not train a model.

## Project layout

```text
app.py                       OpenAI application CLI entry point
app/applications/            CV import/export, workflow, job enrichment, browser, ledger
app/providers/              OpenAI and Ollama providers; other provider placeholders
app/services/               Shared collection, role corpus and structured extraction
app/scraper/                Source adapters, cleaning, quality checks and job storage
app/classifiers/            Importance rules and local model experiments
app/datasets/               Annotation, validation, coverage and dataset construction
scripts/                    Application, collection and dataset commands
config/                     Role catalog and candidate profile example
docs/                       Detailed usage guide for each workflow
tests/                      Offline suites, fixtures and legacy model demonstrations
data/                       Your local job/CV/application data (excluded from Git)
```

## Current limits

- `app.py` currently exposes the OpenAI application workflow; local components
  have their own Python APIs and scripts.
- Form automation covers recognizable Greenhouse/Lever layouts. Dynamic,
  embedded or custom questions often need manual completion. No LinkedIn Easy
  Apply automation is provided.
- Requirement matching and CV auditing use model judgments. Source-quote checks
  and deterministic thresholds help catch errors but do not guarantee truth or
  hiring outcomes.
- Job-page enrichment is best-effort. Summary-only jobs stay in REVIEW until
  a complete description is available.
- `scripts/evaluate_classifier.py`, `app/models.py` and unused provider stubs are
  placeholders, not completed training/evaluation features. The multi-source
  ingestion implementation plan is a historical design document; use the guides
  and executable code for current behavior.
- `.env`, personal profiles, virtual environments and `data/` runtime contents
  are excluded from publication. Synthetic fixtures and documentation are
  included. Keep your CVs and application ledger backed up separately.
