# Job Application Agent

Collect jobs, match them against your master CV with a local classifier baseline,
then use Qwen through Ollama or the OpenAI API to tailor and audit a CV for APPLY
decisions. Both generation providers use the same document exports, dashboard
and browser assistance. Requirement-classifier research tools remain available
independently.

## Choose a workflow

| Capability | Local / Ollama applications | OpenAI applications |
| --- | --- | --- |
| Collect, normalize and deduplicate jobs | Shared collectors | Shared collectors |
| Match a master CV and produce APPLY / SKIP / REVIEW | Local classifier baseline by default | Same local classifier baseline |
| Tailor, evaluate and export PDF/DOCX CVs | Yes | Yes |
| Assist with supported application forms and track outcomes | Yes | Yes |
| CV generation and audit | Installed Ollama model on this machine | OpenAI API |
| OpenAI key or SDK required | No | For CV generation or explicit LLM matching; not classifier match-only |
| Requirement annotation and dataset preparation | Shared independent research scripts | Shared independent research scripts |
| Custom classifier training runner | Not implemented | Not implemented |

The recommended direct-match workflow uses `--matcher central`: the existing
rules plus cached pretrained NLI verify CV evidence, while one local Ollama call
selects only the 2-4 central technical requirements from requirements already
extracted from the advert. The selector never sees the CV and cannot invent a
requirement. A central gap therefore cannot be hidden by many secondary matches.

`--matcher classifier` preserves the older no-generative matching baseline.
Both modes retain deterministic evidence and constraint gates. Neither is a
custom trained or calibrated hiring model, and uncertain evidence stays REVIEW.

Use `prepare --provider ollama` to write and audit CVs using your installed Qwen model.
There is no cloud fallback. `--provider openai` explicitly selects the API.
The provider otherwise comes from `APPLICATION_PROVIDER`, with `openai` as the
fallback when that setting is absent. `.env.example` selects `ollama`.
`APPLICATION_MATCHER` selects `central`, `classifier` or `llm`, with
`classifier` as the code fallback when the setting is absent. The checked-in
`.env.example` recommends `central`. `--matcher llm` explicitly restores the
older full-LLM matching path.
Shared validation rules do not imply equal model accuracy: review generated CVs
and evidence with either generation provider.

### Direct-match verdicts

With `--matcher central`, decisions also include a fit class:

- **DIRECT**: every selected central technical requirement is MET, mandatory requirements and constraints pass, and the numeric threshold is met.
- **STRETCH**: one central technical requirement is missing/partial/unknown, with no hard-constraint failure.
- **POOR**: multiple central requirements are clearly missing, a hard constraint fails, or the score is below threshold after central checks.
- **UNRESOLVED**: the description, sponsorship, extraction, selector or evidence is too uncertain for a reliable fit decision.

The default role catalog is intentionally focused on CPU/systems performance, AI/ML research engineering, and data science/applied ML.

Drafting selects from the complete master CV using source IDs; the application
resolves those references into exact quotes and audits their support for each
claim. Concision limits keep overlong drafts in REVIEW even with a high model
score. CVs target two pages, but factual accuracy and rendered layout still need
your review. The source catalog and reference output are saved with each package.

Detailed guides:

- [OpenAI mode: setup, preferences, preparing CVs and applying](docs/OPENAI_MODE.md)
- [Local mode: Qwen applications, extraction and annotation datasets](docs/LOCAL_MODE.md)
- [Role catalog and collection batches](config/README.md)
- [Annotation schema and review conventions](data/labelled/README.md)
- [Local data layout and backups](data/README.md)

## Prerequisites

The Python code requires Python 3.10 or newer. Use a Python version supported by
your selected dependencies; local PyTorch/CUDA packages may have narrower wheel
availability. The current checkout was tested with Python 3.14. Linux/macOS shell
commands below assume the repository root; on Windows use `.venv\Scripts\python`
for the Python executable and the corresponding PowerShell syntax.

Both modes need a master CV with extractable text. Default matching needs the
classifier libraries and cached NLI weights; no classifier training is required.
Local CV generation needs Ollama and a downloaded Qwen model; OpenAI generation
needs API credentials and model access. Classifier-only matching needs neither generation provider. Central matching additionally needs the configured local Ollama model for requirement selection. A graphical
desktop is needed only when opening application forms. Job discovery, employer
pages and first-time downloads require network access; local model prompts stay
on the configured loopback Ollama server.

## Start applying with local Qwen

Clone the repository and create an isolated Python environment:

```bash
git clone git@github.com:idj997/Job_Application_Agent.git
cd Job_Application_Agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local-applications.txt
.venv/bin/python -m pip install -r requirements-classifier.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/.venv/playwright-browsers" .venv/bin/python -m playwright install chromium
```

Reuse your installed Ollama service and model. If you have not set them up, see
the [local installation guide](docs/LOCAL_MODE.md#2-install-ollama-and-download-the-local-model).
If `.env` does not exist, copy `.env.example` to `.env`. Add or update these
entries, keeping your existing source API keys:

```dotenv
APPLICATION_PROVIDER=ollama
APPLICATION_MATCHER=central
REQUIREMENT_NLI_MODEL=cross-encoder/nli-deberta-v3-small
REQUIREMENT_NLI_DEVICE=cpu
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=32768
OLLAMA_NUM_PREDICT=4096
OLLAMA_TIMEOUT_SECONDS=600
```

The local provider uses structured responses with thinking disabled. `--model`
overrides `OLLAMA_MODEL`. Its server must use a loopback address; it does not send
requests to remote Ollama hosts or fall back to OpenAI.

The matching baseline reads cached NLI model/tokenizer files only; it does not
download them automatically. If you maintain a complete local model directory,
set `REQUIREMENT_NLI_MODEL` to that path. See the [local guide](docs/LOCAL_MODE.md)
for classifier setup and the distinction between dependency checks and model
availability.
Use trusted local model weights and a trusted local service: client-side endpoint
checks cannot prevent a modified daemon or cloud-backed alias from forwarding
prompts. The installed `qwen3:8b` weights are the intended local setup.

Save your master CV under `data/cv/` (PDF, DOCX, UTF-8 text or Markdown). For
matching preferences and browser autofill, copy
`config/candidate_profile.example.json` to `config/candidate_profile.json` and
fill in your real details. `cv_path` resolves relative to the profile file;
`--cv` overrides it and resolves relative to your shell directory. No CV details,
work authorization or personal preferences are inferred from the example.

```bash
.venv/bin/python app.py doctor --provider ollama --matcher classifier
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --limit 1 --match-only --dry-run
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --limit 1 --match-only
```

Inspect the saved decision with `list`/`show`. An APPLY verdict from `--match-only`
has status **matched**, with no CV artifacts, and cannot enter the application
browser. When ready to generate documents for the selected job, rerun without
`--match-only`:

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --limit 1 --max-revisions 0
```

Without a profile, use `--cv /path/to/master.pdf`. That can match and tailor from
your CV, but browser autofill needs the explicit contact fields in the profile.
Greenhouse uses `first_name` and `last_name`; Lever uses `name`.

`doctor` reports dependencies and configuration without contacting Ollama or
loading/checking classifier weights. Default matching makes zero Qwen/OpenAI
requests; a successful first CV draft needs two sequential generation requests,
for tailoring and audit. CPU-only Qwen inference can be slow. Increase batches
only after checking runtime and outputs; the local guide explains context/output
budgets and bounded revisions. If sponsorship is required in your profile, an
advertisement with no clear sponsorship answer must remain REVIEW.

Preparation reads the newest stored jobs from `data/processed_jobs/`. To target a
specific employer page or stored job:

```bash
.venv/bin/python app.py prepare --provider ollama --cv data/cv/master.pdf --url "https://jobs.lever.co/company/job-id" --limit 1 --max-revisions 0
.venv/bin/python app.py prepare --provider ollama --cv data/cv/master.pdf --job data/processed_jobs/JOB_ID.json --limit 1 --max-revisions 0
```

To fetch jobs and prepare packages in one run:

```bash
.venv/bin/python app.py prepare --provider ollama --profile config/candidate_profile.json --fetch --sources adzuna --query "data engineer" --location "United Kingdom" --limit 1 --max-revisions 0
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
.venv/bin/python app.py prepare --provider ollama --cv data/cv/master.pdf --job data/processed_jobs/JOB_ID.json --description-file data/cv/job-description.txt --limit 1 --max-revisions 0
```

`--no-enrich` disables that network retrieval; summary-only jobs become REVIEW
without a model call. A dry run performs no network or model calls and writes no
application artifacts. Scanned PDFs need OCR or a text/DOCX source CV.

### Optional OpenAI API mode

Install `requirements-openai.txt` and configure `OPENAI_API_KEY`, `OPENAI_MODEL`
and optionally `OPENAI_MAX_OUTPUT_TOKENS` only if you want to use the paid API:

```bash
.venv/bin/python -m pip install -r requirements-openai.txt
.venv/bin/python -m pip install -r requirements-classifier.txt
.venv/bin/python app.py doctor --provider openai --matcher classifier
.venv/bin/python app.py prepare --provider openai --matcher classifier --profile config/candidate_profile.json --limit 1 --dry-run
```

For this provider, `--model` overrides `OPENAI_MODEL`. See the
[OpenAI guide](docs/OPENAI_MODE.md) for configuration and a live preparation
command. Inspection and browser commands work with packages from either provider.

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
status. Verbatim evidence is checked in code. The default classifier handles
simple OR/AND groups in code; complex combinations require REVIEW. The explicit
legacy LLM matcher instead receives grouping instructions in its prompt.

Classifier matching preserves shared qualifiers in supported skill lists and
retains verified evidence as PARTIAL when the whole requirement is unresolved.
An open-ended list never becomes fully MET just because its named skills match.
Narrow completed-degree rules preserve the required level and subject alternatives;
unsupported grades, equivalences and incomplete study remain unresolved. Clear
company background is excluded from scoring, and a locally preferred language
does not make an entire stack requirement optional.

Role/location checks can cite the exact structured job title or location using
`job_evidence_source`; conflicting description evidence requires review. The
scorer verifies the whole referenced field and only permits these sources in
classifier mode. A location field or sponsorship question does not establish a
sponsorship offer.

The match score is a weighted fraction of requirements: CORE 5, IMPORTANT 3,
PREFERRED 1; MET gets full credit, PARTIAL half, others zero. Explicit failed
constraints produce SKIP. Unknown constraints, unresolved mandatory requirements
or unverified evidence produce REVIEW. The classifier's uncertainty checks and
the deterministic verdict rules must pass, and the score must reach
`--match-threshold` (default 70), before CV generation. The pretrained NLI model's
confidence is not a calibrated probability of candidate suitability.
Numeric experience requirements and complex eligibility conditions need review
when the rules cannot establish them. Bounded evidence retrieval and input caps
also produce REVIEW instead of silently dropping requirements; see the
[local matching limits](docs/LOCAL_MODE.md#runtime-limits-and-review).

Tailoring preserves the master CV as the source of facts. Every generated entry
carries source quotations. A separate model call evaluates the complete document
for unsupported claims, missing critical information, requirement coverage and
clarity. The CV score is 70% coverage and 30% clarity; default threshold is 75.
Factual issues block ready status regardless of the numerical score. These are
internal rubric scores, not an employer's ATS score or a guarantee of selection.
Model audits can still make mistakes, so read the generated CV before using it.

If the writer ignores selection/length instructions, use a reviewed
`--selection-plan path/to/plan.json` for one saved job with `--limit 1 --no-enrich`.
This optional mode fixes the sections, source IDs, role bullet counts and per-entry
word budgets before generation. Contact details, employment/date headings,
education and project status headings are assembled from source text. The writer
fills only permitted rewrite slots; violations are rejected, never truncated.
The plan is bound to the complete source catalog and exact saved job, so changed
inputs require a new review. It does not change classifier verdicts or establish
the truth of rewritten claims. See [reviewed selection plans](docs/LOCAL_MODE.md#reviewed-selection-plans).

For smaller local writers, add `--entry-by-entry --max-revisions 0` to a planned
draft. Each request receives only one entry's assigned source lines; fixed content
is assembled locally. Each entry allows at most one deterministic repair, then
the complete CV is audited. Reviewed `required_spans` can protect exact titles,
metric qualifiers and important outcomes. This mode does not apply critic-driven
revisions; [setup and limits](docs/LOCAL_MODE.md#entry-by-entry-writing) explain
its higher request count and continued need for factual review.

For an experimental independent local reviewer, add `--critic-model ministral-3:8b` (install once
with `ollama pull ministral-3:8b`). The classifier still matches; Qwen writes; Ministral
audits the complete source-backed CV and suggests up to six grounded changes;
Qwen revises within `--max-revisions`; the critic rechecks. A higher score cannot
override factual or layout checks. The best valid audited draft is retained
if a later revision regresses; failures or unresolved criticism require review.

```bash
OLLAMA_NUM_CTX=12288 OLLAMA_TIMEOUT_SECONDS=900 .venv/bin/python app.py prepare --provider ollama --matcher classifier --critic-model ministral-3:8b --profile config/candidate_profile.json --job data/processed_jobs/JOB_ID.json --limit 1 --max-revisions 1 --no-enrich
```

Set `CV_CRITIC_MODEL=ministral-3:8b` in `.env` to opt in by default, or pass
`--no-critic` to restore self-audit. No model is automatically downloaded.
The models run sequentially and unload between calls. See the
[critic setup and safeguards](docs/LOCAL_MODE.md#separate-local-cv-critic)
for configuration, artifacts and how to interpret scores. This also works
with an explicitly selected OpenAI writer, but only the critic stays local.

Live testing of Gemma 3 4B on a complete master CV found false factual objections
and output-limit failures. The plumbing is tested, but this model/configuration
is not validated for unattended CV revision. It remains disabled by default;
no score improvement should be assumed from enabling it. A nine-case, entry-scoped
Ministral 3 8B check returned structurally valid reports in all cases, but missed
an invented CNN claim and gave some incorrect or incomplete edit advice. This
small check is not an accuracy benchmark; its scores do not demonstrate CV
improvement. A complete-CV follow-up also falsely objected to an exact source
quotation and targeted the wrong entries. Keep revisions manually reviewed.
The guarded runtime supports the inspected standard
Qwen 3, Gemma 3 and Ministral 3 text tokenizers/templates, not arbitrary model tags.
Ministral 3 14B now has a separately inspected template hash; custom templates
remain rejected. Roles can be reversed with
`--provider ollama --model ministral-3:14b --critic-model qwen3:14b`
when both models are installed and pass validation.
Larger models may need partial CPU offload on an 8 GB GPU; size alone does not
establish better writing or criticism. See [model-role options](docs/LOCAL_MODE.md#larger-models-and-reversed-roles).

An experimental [per-entry factual-report API](docs/LOCAL_MODE.md#experimental-per-entry-factual-reports-python-api)
is available programmatically, but is not wired into the CLI and cannot approve
CVs or applications, revise text, or replace the classifier.

By default a job allows one initial CV and one revision (`--max-revisions 0..2`).
With classifier matching, one successful first-pass package normally makes two
Qwen/OpenAI calls: tailoring and evaluation. Each revision adds two calls.
`--match-only` makes none. Explicit `--matcher llm` adds one matching request;
with that option, match-only still makes the matching request. `--limit` defaults to 5;
reported usage includes input/output tokens, but does not estimate account bills.
The OpenAI SDK can retry transient failures. Identical completed preparations are reused;
`--retry` reassesses a package but cannot retry an opened or submitted application.

Default matching uses the local cached classifier. With `--provider ollama`,
CV-writing/audit prompts go to the local Ollama server; there is
no OpenAI request or automatic cloud fallback. Fetching job pages and submitting
forms still use the Internet. With `--provider openai`, CV text and job data are
sent to OpenAI; requests set `store=False`, a response-storage setting that does
not imply zero retention.
The local ledger and artifacts contain personal data and are excluded by
`.gitignore`. The workflow treats job descriptions as untrusted data.
[Official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Verification

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/scraper tests/datasets tests/applications -q
```

The suite uses mocked model responses and local browser fixtures; it makes no
paid API calls, live Ollama inference or real applications. Browser fixture tests skip when Chromium
cannot launch. The existing top-level `tests/test_model.py` and classifier scripts
load local/downloadable models at import time, so they are outside this offline
application test command.

## Local extraction and classifier research

These optional experiments are independent of the local application command
above. The [complete local guide](docs/LOCAL_MODE.md) preserves the structured
extraction API, model experiments and annotation workflow.

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
`OLLAMA_MODEL` and `OLLAMA_BASE_URL` are read from configuration; explicit
constructor arguments override them, as shown in the guide.

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
app.py                       Application CLI with Ollama/OpenAI provider selection
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

- `app.py prepare --provider ollama` and `--provider openai` share the application
  workflow, with classifier matching by default. Optional extraction/classifier experiments have separate Python APIs
  and scripts.
- Form automation covers recognizable Greenhouse/Lever layouts. Dynamic,
  embedded or custom questions often need manual completion. No LinkedIn Easy
  Apply automation is provided.
- Requirement matching uses a pretrained NLI baseline plus rules; the project
  does not ship a trained custom CV-match checkpoint. Source-quote checks,
  uncertainty gates and CV audits help catch errors but do not guarantee truth
  or hiring outcomes.
- Job-page enrichment is best-effort. Summary-only jobs stay in REVIEW until
  a complete description is available.
- `scripts/evaluate_classifier.py`, `app/models.py` and unused provider stubs are
  placeholders, not completed training/evaluation features. The multi-source
  ingestion implementation plan is a historical design document; use the guides
  and executable code for current behavior.
- `.env`, personal profiles, virtual environments and `data/` runtime contents
  are excluded from publication. Synthetic fixtures and documentation are
  included. Keep your CVs and application ledger backed up separately.
