# Local mode: Ollama extraction and requirement datasets

Local mode supports collecting job descriptions, extracting structured job facts
with an Ollama model, experimenting with requirement classifiers, and preparing
reviewed training and evaluation datasets. It does not currently provide a local
command that matches your CV, tailors it, scores it and submits applications.
Use the [OpenAI application guide](OPENAI_MODE.md) for that implemented workflow.

There are two independent paths through the local tooling:

```text
Job collection → stored job descriptions → Ollama JobExtractor → structured facts

Job collection → stored job descriptions → rule-generated candidate queue
    → human review → approved annotations → coverage report → dataset build
```

The annotation and dataset commands do not call Ollama. Running
`scripts.process_jobs` creates annotation candidates; it is not a batch LLM
extraction command. The classifier experiments are separate from both paths.

All commands below run from the repository root using a POSIX shell. Python
commands use `.venv/bin/python`; on Windows, adapt the executable path and shell
syntax. An OpenAI API key is not needed for this guide. Job-source credentials may
still be needed when fetching jobs.

## 1. Install the Python dependencies

Create the virtual environment once if it does not already exist:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
```

Use a Python version supported by the dependencies; Python 3.11 or newer is a
practical starting point. Optional classifier libraries also need compatible
PyTorch wheels for your operating system and Python version.

[requirements-local.txt](../requirements-local.txt) installs the Ollama Python
client, Pydantic, and the lightweight collection dependencies. It does not install
the Ollama server, model weights, PyTorch, Transformers, OpenAI, or browser tools.
The local and OpenAI requirement files can be installed into the same virtual
environment when you want both workflows.

If you only want candidate review and dataset construction, proceed to collection
or the offline dataset example; an Ollama server is unnecessary for those tasks.

## 2. Install Ollama and download the local model

The Python client connects to a separate Ollama server. On Linux, the official
installation command is:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This installs system software and may request administrator access. Other
platforms and manual installation are covered by the
[official Ollama installation documentation](https://docs.ollama.com/linux)
and [downloads page](https://ollama.com/download).

If Ollama is already running as a service or desktop application, keep using that
instance. Otherwise start the server in a separate terminal:

```bash
ollama serve
```

In your working terminal, download the model used by this repository and check
that it is available:

```bash
ollama pull qwen3:8b
ollama ls
```

Model download requires Internet access. The published `qwen3:8b` tag is an 8B
model with an approximately 5.2 GB download; runtime memory also depends on the
model, context size and server configuration. Consult the
[official model entry](https://ollama.com/library/qwen3:8b) for the current tag.
The [Ollama CLI reference](https://docs.ollama.com/cli) documents `serve`, `pull`,
`ls`, and `ps`. A different model can be selected in Python after you download its
exact tag.

The examples below connect to `http://localhost:11434`. Local inference can work
without Internet access after the server and model are installed. Changing
`host` to another machine sends the prompt to that machine.

## 3. Extract one job with Ollama

This runnable example uses the actual
[OllamaProvider](../app/providers/ollama_provider.py),
[JobExtractor](../app/services/job_extractor.py), and
[JobDescription schema](../app/job_parser.py):

```bash
.venv/bin/python - <<'PY'
from app.providers.ollama_provider import OllamaProvider
from app.services.job_extractor import JobExtractor

description = """
JOB TITLE: Data Engineer
COMPANY: Example Company
LOCATION: Bristol
SALARY: GBP 50000 - 65000

Strong SQL experience is essential. You will build pipelines using Python.
Experience with Apache Airflow would be desirable.
Candidates need at least 3 years of data engineering experience.
The role is hybrid, with two days per week in the Bristol office.
"""

provider = OllamaProvider(model="qwen3:8b", host="http://localhost:11434")
job = JobExtractor(provider).extract(description)
print(job.model_dump_json(indent=2))
PY
```

The result includes title, employer, location, salary, canonical required and
preferred skills, experience, work pattern, sponsorship and clearance fields.
The extractor asks the model to use explicit facts and retain unknowns. Review
the output against the source description: a valid JSON result can still contain
an incorrect extraction.

The provider sends `schema.model_json_schema()` as Ollama's `format`, sets
temperature to zero for structured calls, and validates the returned JSON with
Pydantic. This follows Ollama's
[structured-output interface](https://docs.ollama.com/capabilities/structured-outputs).
The provider does not implement a retry loop, explicit response-token budget, or
batch orchestration. `generate(prompt)` also exists, but returns ordinary text.

### Configuration is explicit

The constructor defaults are `model="qwen3:8b"` and
`host="http://localhost:11434"`. This repository's `OllamaProvider` does not load
`.env` or read `OLLAMA_MODEL` to select the model automatically. Pass `model=` and
`host=` in your caller. Ollama server environment configuration and this Python
caller's arguments are separate settings.

There is no `app.py --mode local` or `prepare --provider ollama` switch. The
application CLI is wired to OpenAI; the Python API above is the supported entry
point for the existing local extraction experiment.

## 4. Collect job descriptions

Both modes share the collector and its default directories:

| Location | Contents |
| --- | --- |
| `data/raw_jobs/` | Original HTML or API payloads |
| `data/processed_jobs/` | Cleaned job JSON, source URLs, IDs and metadata |
| `config/job_role_queries.json` | Editable role families and query phrases |

Collection performs network requests. It does not call Ollama or OpenAI and does
not need PyTorch. The collector validates descriptions and deduplicates saved
jobs; discovered-job limits do not guarantee the same number of new records.

### A small search

For the Adzuna adapter, put your own credentials in the untracked `.env` file:

```dotenv
ADZUNA_APP_ID=your-adzuna-app-id
ADZUNA_APP_KEY=your-adzuna-app-key
```

Then collect a small batch:

```bash
.venv/bin/python -m scripts.scrape_jobs \
  --source adzuna \
  --query "data engineer" \
  --location "United Kingdom" \
  --limit 5
```

The adapter defaults to the UK (`gb`). Its stored descriptions are explicitly
marked as summaries. A local extraction can describe facts present in a summary,
but cannot establish that absent qualifications are absent from the full role.
Use a full employer description for serious requirement annotation and matching.
The local collector does not automatically perform the OpenAI workflow's
full-description enrichment step.

Other source examples:

```bash
.venv/bin/python -m scripts.scrape_jobs \
  --sources arbeitnow remotive \
  --query "data engineer" \
  --limit-per-source 5

.venv/bin/python -m scripts.scrape_jobs \
  --source greenhouse --board-token EMPLOYER_BOARD_TOKEN --limit 5

.venv/bin/python -m scripts.scrape_jobs \
  --source lever --company-slug EMPLOYER_SLUG --limit 5
```

Replace uppercase employer placeholders with the identifiers from the employer's
board. These are public-board identifiers, not applicant submission API keys.
The adapters can also read `GREENHOUSE_BOARD_TOKEN` and `LEVER_COMPANY_SLUG`
from `.env`; `THE_MUSE_API_KEY` is optional for the Muse adapter. Availability,
geographical coverage and filters vary by source. See
[source_registry.py](../app/scraper/source_registry.py) and
[scrape_jobs.py](../scripts/scrape_jobs.py) for supported options.

`dwp` is also implemented. The legacy `--source dwp` command only reports
discovered URLs; use `--sources dwp --query "data engineer" --limit-per-source 5`
to route it through collection and storage. A positional employer URL is handled
by the generic JSON-LD extractor, which requires usable structured job data in
the fetched HTML and does not render JavaScript.

### Build a diverse role corpus

Inspect the role catalog and a bounded plan before running network collection:

```bash
.venv/bin/python -m scripts.collect_role_corpus --list-roles

.venv/bin/python -m scripts.collect_role_corpus \
  --sources adzuna \
  --roles data_engineering data_science \
  --location "United Kingdom" \
  --queries-per-role 1 \
  --limit-per-query 5 \
  --dry-run
```

Remove `--dry-run` when ready to fetch. This plan contains two query/source pairs,
each with a discovery limit of five. Role families, source and query provenance
are preserved for later coverage reporting. Cross-query duplicates are skipped,
and a source failure does not cancel other configured sources.

The default plan cap is 50 query/source pairs. Increase the selected roles,
sources or queries deliberately; `--allow-large-run` overrides that cap. The
[role catalog guide](../config/README.md) describes the configuration file.

### Run the extractor on a stored job

Replace `JOB_ID.json` with a file actually saved in `data/processed_jobs/`:

```bash
.venv/bin/python - data/processed_jobs/JOB_ID.json <<'PY'
import json
import sys
from pathlib import Path

from app.providers.ollama_provider import OllamaProvider
from app.services.job_extractor import JobExtractor

source_path = Path(sys.argv[1])
record = json.loads(source_path.read_text(encoding="utf-8"))
provider = OllamaProvider(model="qwen3:8b", host="http://localhost:11434")
result = JobExtractor(provider).extract(record["description"])

destination = Path("data/local_extractions") / source_path.name
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
print(destination)
PY
```

This writes a separate extraction file. It does not update the processed job,
create an approved annotation, match a candidate, or start an application. The
annotation queue below uses the original processed descriptions directly.

## 5. Generate and review annotation candidates

Generate the rule-based queue:

```bash
.venv/bin/python -m scripts.process_jobs \
  --input-dir data/processed_jobs \
  --output data/labelled/requirement_candidates.json
```

The generator uses a canonical-skill dictionary and contextual/section rules in
[annotation_candidates.py](../app/datasets/annotation_candidates.py). Proposed
labels and confidence values are suggestions that require review. Skills outside
the dictionary or ambiguous wording can be missed; inspect the original job.

The human-readable JSON is an array of candidates. Each candidate starts with
`annotation_status: "needs_review"`. It includes a stable `candidate_id`, proposed
label, evidence, nearby `review_context` and a `job_description_ref` pointing to
the processed job. A companion `.jsonl` file preserves full descriptions for
machine processing. Keep referenced processed jobs available: promotion reloads
them and checks their IDs and evidence.

Review each candidate in your editor, preserving its identity and provenance:

| Field | What to set |
| --- | --- |
| `annotation_status` | `approved`, `rejected`, or keep `needs_review` |
| `reviewed_label` | One of the five labels for a correction; leave empty to accept the proposed label after approval |
| `review_notes` | Your reasoning, especially for ambiguity, negation or conditional requirements |

Approve a valid requirement/evidence pair even when correcting its label. Reject
an invalid extraction, such as an irrelevant skill occurrence or an incorrect
evidence span. Merely mentioning a different label in notes does not change the
effective label.

The reviewed dataset uses these five labels:

| Label | Meaning |
| --- | --- |
| `CORE` | Explicitly mandatory, essential, or a must-have |
| `IMPORTANT` | Expected in responsibilities or strongly requested |
| `PREFERRED` | Desirable, advantageous, or a bonus |
| `CONTEXTUAL` | Environment/team context without asking the candidate to possess it |
| `NOT_REQUIREMENT` | Explicitly unnecessary, negated, or an irrelevant mention |

Use a short canonical requirement, such as `SQL`, and an exact `evidence_span`
from the full job description. Do not silently convert ambiguous evidence into a
definite label. The [annotation reference](../data/labelled/README.md) includes the
complete record format and review conventions.

### Alternatives and grouped requirements

If the role accepts Java **or** Python, preserve the alternative relationship.
An annotated member can carry `requirement_group`, `group_operator: "ANY_OF"`,
`group_members` and `group_label`. Use `ALL_OF` only when every member is required.
The group label must equal the member's final label, the member must be in the
group, and at least two distinct members are required. Correct related group
fields together when changing a label.

### Validate and promote the reviewed queue

```bash
.venv/bin/python -m scripts.process_jobs \
  --validate-reviews data/labelled/requirement_candidates.json \
  --review-report data/labelled/review_validation.json

.venv/bin/python -m scripts.process_jobs \
  --promote-approved data/labelled/requirement_candidates.json \
  --approved-output data/labelled/requirements.jsonl \
  --decision-log data/labelled/review_decisions.json
```

Read the validation report's errors and warnings before promotion. The validation
command writes a report and can exit successfully with reported issues; its exit
code alone is not a quality gate. Promotion accepts only explicitly approved
records and rejects invalid evidence, labels, references or conflicting approved
annotations. A queue with no approved records cannot be promoted.

The decision log records approvals, rejections, pending decisions and notes.
Review notes are preserved as metadata and are excluded from classifier input to
avoid leaking the answer into training text.

Regenerating the same candidate queue preserves existing review fields by stable
candidate ID. Promotion **rewrites** the requested annotation output with the
approved contents of that queue; it does not append an earlier annotation file.
Use separate output files for separately reviewed batches, then pass them all to
the coverage reporter and dataset builder.

## 6. Optional reviewed counterfactuals

After approving real examples, generate explicit negative candidates:

```bash
.venv/bin/python -m scripts.generate_counterfactuals \
  --input data/labelled/requirements.jsonl \
  --output data/labelled/not_requirement_candidates.json \
  --variants-per-example 1 \
  --seed 42
```

Only approved positive `CORE`, `IMPORTANT` and `PREFERRED` examples are eligible
by default. The generator uses deterministic negation templates, marks the
results synthetic, and keeps the original job ID so related examples stay in
the same train/evaluation split. Grouped alternatives are skipped.

Review this queue independently. Once approved, promote it to a separate file:

```bash
.venv/bin/python -m scripts.process_jobs \
  --promote-approved data/labelled/not_requirement_candidates.json \
  --approved-output data/labelled/requirements.counterfactual.jsonl \
  --decision-log data/labelled/counterfactual_review_decisions.json
```

Explicit-negation examples must be approved as `NOT_REQUIREMENT` or left
unapproved. They are not evidence of the original employer's actual requirements.
To review both queues together, `process_jobs --include-candidates` can merge
the counterfactual queue while generating the main queue; see the annotation
reference for that alternative workflow. Do not promote both a merged queue and
its separate source files as different evidence.

## 7. Check coverage and build datasets

Measure the approved corpus first:

```bash
.venv/bin/python -m scripts.report_dataset_coverage \
  --input data/labelled/requirements.jsonl \
  --output data/labelled/coverage_report.json
```

Default coverage targets are 20 examples per label, five per role family in the
catalog, 30 unique jobs, at least three sources, and no label above 50% of the
corpus. Review the report's blockers and distributions. These are configurable
readiness checks, not a measured classifier accuracy. Reporting a gap does not
itself train or evaluate a model.

Build the dataset from the reviewed file:

```bash
.venv/bin/python -m scripts.build_dataset \
  --input data/labelled/requirements.jsonl \
  --output-dir data/datasets/requirements \
  --seed 42 \
  --evaluation-ratio 0.20 \
  --noise-ratio 0.30 \
  --minimum-per-label 1 \
  --maximum-synthetic-ratio 0.40
```

If you also reviewed counterfactuals or separate batches, provide multiple files
after `--input` in both commands, for example:

```bash
.venv/bin/python -m scripts.build_dataset \
  --input data/labelled/requirements.jsonl data/labelled/requirements.counterfactual.jsonl \
  --output-dir data/datasets/requirements-with-counterfactuals
```

The builder validates annotations, rejects conflicting labels and excessive
synthetic proportions, and splits by job ID. Variants of a job cannot be spread
across training and evaluation. A small corpus may not support the requested
evaluation fraction while retaining each label in training; inspect the actual
split counts rather than assuming the target ratio was achieved.

| Output | Purpose |
| --- | --- |
| `train.jsonl` | Original examples plus deterministic noisy variants |
| `evaluation/clean.jsonl` | Evidence-only contexts for held-out real examples |
| `evaluation/real_world.jsonl` | Original full descriptions for those examples |
| `evaluation/noisy.jsonl` | Paired full descriptions with controlled clutter |
| `evaluation/synthetic.jsonl` | Held-out synthetic examples, separate from the primary real-world evaluation |
| `manifest.json` | Actual counts, settings, label/role/source distributions and provenance |

Noise metadata is recorded in `noise_types`, and evidence remains unchanged.
`--noise-ratio` targets the noisy fraction of training records. Rebuilding into
the same output directory replaces those files; use a new directory to preserve
an earlier experiment.

The minimum of one example per label is only a construction check. A resulting
file is not proof of enough data for useful training. Coverage targets and
dataset-builder validation are separate checks; building does not automatically
enforce the coverage report's readiness decision.

### Offline example with bundled test data

This command exercises the builder without job downloads, Ollama or human-data
files:

```bash
.venv/bin/python -m scripts.build_dataset \
  --input tests/fixtures/datasets/requirements.jsonl \
  --output-dir data/datasets/demo-requirements
```

The fixture is tiny test data. Use it to verify installation and inspect the
output format, not to make claims about model performance.

## 8. Optional classifier experiments

These modules are exploratory components, not a trained five-label application
decision pipeline:

| Component | Actual output and limitation |
| --- | --- |
| [importance_rule.py](../app/classifiers/importance_rule.py) | Keyword rules returning lowercase `core`, `preferred`, `contextual`, or `ambiguous`; this is a different taxonomy from the five-label dataset |
| [requirement_nli.py](../app/classifiers/requirement_nli.py) | NLI label/probabilities for a supplied context and hypothesis using `cross-encoder/nli-deberta-v3-small` |
| [semantic_relevance.py](../app/classifiers/semantic_relevance.py) | A raw cross-encoder relevance score using `cross-encoder/ms-marco-MiniLM-L6-v2`; it is not a probability or an importance label |
| [test_jobbert.py](../tests/test_jobbert.py) | An executable skill-extraction experiment using `jjzha/jobbert_knowledge_extraction`, followed by rule-based importance |

The simple importance rules match wording without robust negation handling. For
example, a phrase containing “not required” still contains “required”. Do not
treat their output as approved labels. The candidate-queue generator has separate,
more detailed rules and still requires review.

Install PyTorch using the command for your operating system and CPU/CUDA/ROCm
configuration from the [official installation selector](https://pytorch.org/get-started/locally/).
Run that command with `.venv/bin/python -m pip` in place of `pip`/`pip3`. Then
install the optional text-model dependencies:

```bash
.venv/bin/python -m pip install "transformers>=4.45,<6" "sentence-transformers>=3,<6" sentencepiece
```

These dependencies can be large; they are intentionally excluded from the local
base requirement file. See also the
[Sentence Transformers installation guide](https://www.sbert.net/docs/installation.html).

The repository's executable experiments are:

```bash
.venv/bin/python -m tests.test_model
.venv/bin/python -m tests.test_requirement_nli
.venv/bin/python -m tests.test_semantic_inportance
.venv/bin/python -m tests.test_jobbert
```

The spelling `test_semantic_inportance` matches the existing filename. The first
command calls Ollama. The other three load pretrained models and may download
weights/tokenizers from Hugging Face on their first run. They print examples;
they are not a benchmark over the datasets built above. NLI chooses CUDA when
available and otherwise CPU; its tokenizer currently truncates each input pair
to 256 tokens.

After all needed Hugging Face files are cached, `HF_HUB_OFFLINE=1` prevents Hub
HTTP requests and fails if required cached files are unavailable. For example:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m tests.test_requirement_nli
```

See the [official Hugging Face environment-variable reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
for cache and offline configuration. This variable does not control job-source
requests or Ollama's separate server/model storage.

## 9. Verify the local tooling without models or network

Install the test runner separately, then run the deterministic collector and
dataset tests:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/scraper tests/datasets -q
```

These tests use fixtures and mocked source responses. They do not need an Ollama
server, download classifier weights, or call OpenAI. CLI help and the role-corpus
`--dry-run` plan are also available without network requests:

```bash
.venv/bin/python -m scripts.scrape_jobs --help
.venv/bin/python -m scripts.process_jobs --help
.venv/bin/python -m scripts.build_dataset --help
.venv/bin/python -m scripts.report_dataset_coverage --help
```

Avoid an unrestricted `pytest` run when you intend an offline check: the existing
top-level model experiments perform inference/model loading at import time.

## 10. Troubleshooting and current limits

| Symptom | What to check |
| --- | --- |
| `No module named ollama`, `dotenv`, or `bs4` | Install `requirements-local.txt` with the same `.venv/bin/python` used to run commands |
| Ollama connection refused | Start the server or service and ensure `host=` points to its listening address |
| Port 11434 already in use | An Ollama instance may already be running; use it instead of starting a second server |
| Ollama model not found | Pull the exact tag passed as `model=` and check `ollama ls` |
| Slow inference or memory exhaustion | Inspect `ollama ps`; reduce workload/context or explicitly choose a smaller downloaded model |
| Structured-output validation error | Inspect the description/model choice, update the server if necessary, and retry a small example; the provider currently does not repair invalid responses |
| Missing source credentials | Add the adapter's source credentials or employer identifier; Ollama configuration does not supply these |
| Few or zero saved jobs | Inspect discovered, duplicate, rejected and failed counts; source filters, quality gates and prior saved jobs affect results |
| No processed JSON files | Collect jobs first or correct `--input-dir`; local extraction outputs are stored separately |
| No approved examples | Set explicit approval statuses after review; a proposed label alone is insufficient |
| Evidence or reference validation fails | Restore the referenced processed job, verify its ID, and copy evidence exactly from that description |
| Dataset has insufficient labels | Add reviewed examples across all five labels and more independent jobs; inspect the coverage report |
| Missing classifier package, tokenizer or weights | Install optional dependencies and obtain the required model files before using offline mode |

There is currently no classifier training runner, and
[scripts/evaluate_classifier.py](../scripts/evaluate_classifier.py) is empty.
The generated evaluation files are ready for a future evaluation implementation;
no accuracy, F1 or calibration metrics are computed by that placeholder.

The local code does not yet assemble candidate-CV matching, fact-preserving CV
tailoring, CV scoring and browser submission into an Ollama workflow. The
interchangeable provider interface is useful groundwork, but changing a provider
name is not an implemented local application mode. Local extraction output is
also not automatically promoted into the human-reviewed dataset.

Keep source credentials, personal CVs, model caches and generated data out of
Git. Review the repository's ignore rules before publishing any corpus: scraped
job text and annotation notes may need separate handling from reusable code and
the small test fixtures. Return to the [project overview](../README.md) for both
modes and repository setup.
