# Local mode: Qwen applications, extraction and requirement datasets

Local mode uses a cached classifier baseline to match your master CV and decide
APPLY/SKIP/REVIEW. Qwen through Ollama then writes and audits a CV only for APPLY
decisions, before PDF/DOCX export and browser assistance. It requires neither an
OpenAI key nor the OpenAI SDK, and does not fall back to a cloud model.

The matcher combines requirement rules with pretrained NLI. It is not a trained
custom CV-match model, and its scores are not calibrated suitability
probabilities. Uncertainty produces REVIEW; it does not trigger an automatic
LLM replacement for matching. The existing reviewed requirement-importance
dataset is training data for a different task, not a deployed CV-match checkpoint.

The existing extraction, classifier experiments and human-reviewed dataset
tooling remain available independently. There are three paths:

```text
Jobs → full descriptions → classifier CV match → APPLY → Qwen tailor/audit → documents
    → review package → application browser → confirmed outcome

Job collection → stored job descriptions → Ollama JobExtractor → structured facts

Job collection → stored job descriptions → rule-generated candidate queue
    → human review → approved annotations → coverage report → dataset build
```

The annotation and dataset commands do not call Ollama. Running
`scripts.process_jobs` creates annotation candidates; it is not a batch LLM
extraction command. The classifier experiments are separate components.

Application preparation uses the same evidence, scoring, document and browser
rules as [OpenAI mode](OPENAI_MODE.md). This does not establish equal model
accuracy: Qwen can misinterpret a requirement or miss a factual problem, so read
the assessment and generated CV before applying.

All commands below run from the repository root using a POSIX shell. Python
commands use `.venv/bin/python`; on Windows, adapt the executable path and shell
syntax. An OpenAI API key is not needed for this guide. Job-source credentials may
still be needed when fetching jobs.

## Apply using an existing Qwen installation

If Ollama and `qwen3:8b` are already installed, reuse them. Install the application
dependencies in your existing virtual environment:

```bash
.venv/bin/python -m pip install -r requirements-local-applications.txt
.venv/bin/python -m pip install -r requirements-classifier.txt
```

[requirements-local-applications.txt](../requirements-local-applications.txt)
includes the local generator, collectors, PDF/DOCX exporters and browser library.
[requirements-classifier.txt](../requirements-classifier.txt) adds the matching
baseline's PyTorch, Transformers and tokenizer libraries. Neither installs model
weights or a custom trained checkpoint. If you need a
virtual environment or Ollama installation first, follow sections 1 and 2 below,
then install both application and classifier requirement files above.

Add or update these entries in your existing `.env`, preserving your job-source
credentials:

```dotenv
APPLICATION_PROVIDER=ollama
APPLICATION_MATCHER=classifier
REQUIREMENT_NLI_MODEL=cross-encoder/nli-deberta-v3-small
REQUIREMENT_NLI_REVISION=
REQUIREMENT_NLI_DEVICE=cpu
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=32768
OLLAMA_NUM_PREDICT=4096
OLLAMA_TIMEOUT_SECONDS=600
```

`prepare --provider ollama` selects the local CV generator/auditor. Without an explicit
flag, `APPLICATION_PROVIDER` selects the provider; if unset, the CLI defaults to
`openai`. The checked-in `.env.example` selects `ollama`. A command-line `--model`
overrides `OLLAMA_MODEL` for local preparation. Exported shell variables take
precedence over `.env`.

`--matcher classifier` overrides `APPLICATION_MATCHER`, whose fallback is
`classifier`. The matcher loads only cached model/tokenizer files and defaults
to CPU. `REQUIREMENT_NLI_MODEL` can identify the cached pretrained model or a
complete local model directory; `REQUIREMENT_NLI_REVISION` optionally selects a
cached revision. Supply these files separately if they are absent. Neither
ordinary matching nor `doctor` downloads them automatically. Missing/unusable
weights cannot produce an approved match.

For a deliberate comparison with the previous approach, `--matcher llm` opts
into Qwen/OpenAI matching. That path makes an additional generation-model
request. It is never chosen automatically after classifier failure. The default
classifier path does not call Qwen for requirement extraction; the standalone
Ollama extraction API later in this guide remains a separate experiment.

The Ollama endpoint must be a loopback address on this machine, such as
`localhost`, `127.0.0.1` or `[::1]`. Remote/cloud endpoints are rejected. The
provider requests structured JSON with temperature zero and `think=False`.
No API credential is needed for the local server.

Use a trusted local Ollama service and locally installed model weights. The
client rejects known cloud-tag suffixes, remote endpoints, redirects and
environment proxies, but does not verify arbitrary model aliases or prevent a
modified local server from forwarding prompts. Do not substitute a cloud-backed
alias for the installed Qwen model when local-only processing is required.

### Add the master CV and preferences

Create the local CV directory and copy the example profile only if needed:

```bash
mkdir -p data/cv
if [ ! -e config/candidate_profile.json ]; then
    cp config/candidate_profile.example.json config/candidate_profile.json
fi
```

Save your master CV under `data/cv/` and fill in the profile with your own name,
contact details, CV path and preferences. The profile is JSON; relative `cv_path`
values resolve from the profile file's directory, for example
`"../data/cv/master.pdf"`. `--cv` overrides it and resolves from your shell's
directory. Use a text-based PDF, DOCX, UTF-8 text or Markdown CV; scanned PDFs
need OCR or a text export first.

Your master CV remains the factual source. Set
`preferences.requires_sponsorship` to `true` if sponsorship is required. A job
without clear sponsorship evidence must then remain REVIEW, even if its skill
score is high. Neither a location nor a name establishes work authorization.
`preferences.sponsorship_exempt_locations` can name locations where that
requirement does not apply, for example `["India"]` if that accurately describes
your situation. Ambiguous job locations still need review; this is an explicit
preference, not an inferred work-authorization claim.
The [profile field reference](OPENAI_MODE.md#2-add-your-cv-and-profile) applies to
both providers.

### Check and prepare one job

Start with a dry run and a single job:

```bash
.venv/bin/python app.py doctor --provider ollama --matcher classifier
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --limit 1 --match-only --dry-run
```

`doctor` reports the selected generator/matcher dependencies and configuration
without contacting Ollama, downloading weights or loading a model. It identifies
the pretrained NLI baseline and does not claim a custom trained checkpoint. It
cannot establish that the server is running or that classifier/model files are
usable. Dry run
reads your CV/profile and previews the batch; it makes no model or job-network
requests and writes no application package.
`sentencepiece` is an advisory dependency: it is needed for some tokenizer
formats, but a complete cached fast tokenizer can work without it.

When your stored jobs and profile are ready, run matching only first:

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --limit 1 --match-only
```

This selects the newest job from `data/processed_jobs/`, runs the local matcher
and saves its decision. No generation provider is constructed. APPLY produces
status `matched`, without CV files; it is not `ready` and cannot open an
application form. SKIP/REVIEW retain their usual states. Inspect the decision
using `list`/`show`.

To target one saved job explicitly, add
`--job data/processed_jobs/JOB_ID.json`. Remove `--match-only` when you want CV
writing and audit after a successful match:

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --limit 1 --max-revisions 0
```

To fetch and match one job, using your configured source credentials:

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --fetch --sources adzuna --query "data engineer" --location "United Kingdom" --limit 1 --match-only
```

Adzuna supplies summaries. The application workflow attempts to retrieve a full
description before matching; unresolved summaries stay REVIEW without a model
request. If retrieval fails, save the complete employer text and use:

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --job data/processed_jobs/JOB_ID.json --description-file data/cv/job-description.txt --limit 1 --match-only --no-enrich
```

Replace the job filename with your actual saved record. The supplied description
must contain the real role details; do not add text to satisfy completeness
checks. With a stored job and a local full-description file, `--no-enrich` avoids
employer-page retrieval. `--fetch`, `--url` and the application browser still
access external job sites when requested.

### Runtime, limits and review

#### Manually review one classifier REVIEW before testing CV tailoring

If you have checked a specific saved classifier REVIEW and want to inspect a
Qwen draft despite unresolved matching items, explicitly approve tailoring only:

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --profile config/candidate_profile.json --job data/processed_jobs/JOB_ID.json --limit 1 --no-enrich --max-revisions 0 --manual-approval-note "I reviewed this specific job and approve a CV-tailoring test only; unresolved items still need review."
```

Use the exact same saved job, profile and master CV as the original assessment.
Changed inputs require a fresh `--match-only` assessment first. The option
accepts exactly one explicit job, not discovery, URLs, batches, replacement
descriptions or match-only runs. SKIP, incomplete, opened or attempted records
cannot be overridden. Old reviews with nondefault score thresholds may also
need a fresh assessment to establish their source fingerprint.

The original classifier verdict, score, requirements and reasons remain intact;
the ledger separately records the manual APPLY-for-tailoring note, timestamp
and original assessment. It reuses that assessment without another matching
request. Even a high-scoring CV stays `review` and cannot be opened or submitted
through the application commands: this approval is **not submission approval**.
Read the generated CV and source evidence yourself before deciding next steps.
The same configured context/output limits still apply; manual approval does not
make an oversized prompt safe or repair unavailable GPU drivers.
Manual drafts are checkpointed as `cv_draft.md` and `cv_draft.json` before the
Qwen audit; if evaluation or export fails, those unaudited files remain
inspectable in the ledger's artifacts. A failed manual run can be retried with
the same exact sources and a fresh explicit approval note (add `--retry`);
the previous approval is retained in history.

Preparation is sequential. Classifier matching and classifier `--match-only`
make zero Qwen/OpenAI requests; NLI inference runs locally on the configured
device. An APPLY decision with CV preparation normally needs two Qwen requests:
tailoring and evaluation. Each CV revision adds two. `--max-revisions 0` keeps the
first draft run short. With explicit `--matcher llm`, add one matching request;
that request still happens in LLM match-only mode. The
general default is one revision, and at most two are allowed. Increase to one
revision when you want audit feedback used to correct a draft, rather than
increasing the batch size immediately.

### Reviewed selection plans

Word limits in a prompt are not a reliable selection mechanism for a small model.
The optional `--selection-plan` mode fixes the document structure locally and
asks the writer only for text in named rewrite slots. It works with either the
writer's self-audit or a separately selected critic; matching is unchanged.

A plan is an explicitly reviewed JSON file using the `SelectionPlan` schema in
[`app/applications/cv_selection.py`](../app/applications/cv_selection.py). It contains:

- `source_catalog_sha256` and `job_sha256`, binding the complete numbered source
  catalog and exact prepared job. Generate these with `source_catalog_fingerprint`
  and `prepared_job_fingerprint`; do not copy IDs or hashes from another CV/job.
- Ordered `sections`, each with `heading` and ordered `slots`. Every slot has a
  unique `slot_id`, assigned `evidence_ids`, `mode` (`fixed` or `rewrite`), `style`
  (`paragraph` or `bullet`) and `max_words`. Optional `required_spans` contains
  `{ "source_id": "S0012", "text": "Contributed to" }` objects, using actual
  assigned IDs and exact source excerpts. These protect reviewed wording that
  must survive shortening, such as titles, approximation or metric qualifiers.
- Explicit `roles`: each declares a fixed `heading_slot_id`, ordered
  `achievement_slot_ids` and a `primary` boolean. Exactly one role is primary;
  at most eight primary and three per earlier role achievement slots are allowed.
  All Experience slots must belong to these declared roles; roles are not inferred
  from punctuation or paragraph style.
- `project_heading_slot_ids`, identifying up to two fixed project headings,
  including their supplied independent scope, dates and ongoing/paused status.

Contact and Education entries, employment headings and project headings must be
fixed source joins, using exact source lines separated by ` | `. Review those
source groups to ensure identity, chronology and status are complete. The compiler
cannot determine whether a person chose the right lines or omitted a qualification.

Fixed text plus all rewrite budgets must fit the 650-word maximum before inference.
The plan also enforces at most 32 non-contact entries, five Skills entries and a
70-word total summary budget; an individual plan can be stricter. Fewer bullets
are allowed when sources are short; there is no requirement to pad the CV.

```bash
.venv/bin/python app.py prepare --provider ollama --matcher classifier --critic-model ministral-3:8b --profile config/candidate_profile.json --job data/processed_jobs/JOB_ID.json --selection-plan data/applications/reviewed_plan.json --limit 1 --no-enrich --max-revisions 0 --dry-run
```

Replace the two placeholder file paths with your saved job and reviewed plan.
Selection-plan mode requires exactly one `--job`, `--limit 1` and `--no-enrich`;
it rejects discovery, URL fetching, description replacement and match-only mode.
Dry-run validates bindings and budgets without loading models or writing a ledger.
Remove `--dry-run` for generation. A classifier REVIEW still requires the separate
draft-only `--manual-approval-note`; a plan is not an override of job eligibility.

The writer receives the full source catalog plus the plan, but can return only
the named rewrite strings. Locally assigned sections, styles, reference IDs and
fixed text cannot be replaced by its output. Missing/extra fields, invalid text
or over-budget slots are rejected before audit; no words or source lines are
silently cut. Exact-source reference repair is disabled for planned entries so
it cannot broaden a slot's assigned evidence. The plan and raw slot response
are saved with the artifacts and the plan fingerprint participates in caching.

Start with zero automatic revisions while testing critic quality. A source ID,
word budget or structurally valid response does not prove that rewritten text
retains every qualifier or attributes an outcome correctly. Review claims against
their sources, and check the rendered PDF: a word budget alone cannot guarantee
two pages. Without `--selection-plan`, the existing drafting mode is unchanged.

Live local testing confirmed that fixed slots and per-entry rejection work, but
did not establish reliable shortening: Qwen returned the permitted rewrite fields
yet exceeded three slot budgets in both an initial run and a supervised retry.
Those responses were rejected before critique and export. This mode enforces
boundaries; it does not promise that every generation will satisfy them or
automatically repair a rejected response.

### Entry-by-entry writing

Add `--entry-by-entry --max-revisions 0` to a reviewed selection-plan run to write
one entry at a time. This is useful when Qwen ignores budgets or borrows facts
from other parts of a large CV. It is opt-in; whole-plan and ordinary drafting
remain available.

Each rewrite request receives only its reviewed, assigned source lines, entry
type, word budget and any required source spans. It receives no job description,
other CV entries or critic feedback. This is explicit source selection, not
token-budget truncation. The full master catalog is still saved and supplied to
the final audit, and the classifier's matching inputs and verdict are unchanged.
Fixed contact, identity, date, education and project-heading entries make no
generation calls.

The writer aims below the hard word cap. Locally checked formatting, word-limit,
missing required-span or source-absent proficiency-label problems can trigger
one retry for that same entry with the same sources and limits. There is no
provider/network-error retry in this loop. An unresolved entry stops assembly;
partial outputs remain diagnostic artifacts, not an approved CV. Text is never
trimmed to force a pass and caps are never raised automatically.
The retry rebuilds from the original assigned sources and local validation
feedback. Rejected prose stays in private checkpoints but is not sent back as a
draft to imitate. The prompt uses neutral skill wording and prioritises required
facts before optional detail.
For an over-budget response, the retry states the locally measured count, the
unchanged cap and target, and asks for a compact rewrite with optional detail
removed. Only trusted counts enter that instruction; source text is never promoted
to system instructions.

Required spans must be exact substrings of assigned source lines. Generated
text must retain them as standalone phrases (case may vary). A phrase's presence
does not prove correct attribution or causality. Similarly, blocking unsupported
words such as “expertise” is a conservative lexical check, not a factuality
classifier. Independently review the complete wording and PDF layout.
Choose short, meaningful spans: separate degree and subject excerpts can allow
natural connecting words while retaining both facts. Keep critical metric and
qualifier phrases together. An exact-span failure can reject a truthful paraphrase;
it is a wording-contract failure, not necessarily a hallucination. Changes to these
reviewed requirements must be explicit, never an automatic response to rejection.

Example with separate local critique and a smaller writer context/output budget:

```bash
OLLAMA_NUM_CTX=4096 OLLAMA_NUM_PREDICT=384 CV_CRITIC_NUM_CTX=12288 CV_CRITIC_NUM_PREDICT=3072 .venv/bin/python app.py prepare --provider ollama --matcher classifier --critic-model ministral-3:8b --profile config/candidate_profile.json --job data/processed_jobs/JOB_ID.json --selection-plan data/applications/reviewed_plan.json --entry-by-entry --limit 1 --no-enrich --max-revisions 0 --dry-run
```

Replace the saved-job and reviewed-plan placeholders. These budgets are starting
points, not a guarantee for every source group. Remove `--dry-run` to generate;
an existing classifier REVIEW still needs explicit drafting-only manual approval.
Use a full-document context budget when the writer also performs self-audit;
the 4,096-token example is for the separate-critic setup only.

For `N` rewrite slots there are at most `2 × N` drafting requests and one full
audit request, plus any explicitly selected model-matching call. Usually each
slot needs one request; all-fixed plans need zero drafting calls. This increases
request counts even though individual inputs are smaller. Models remain serial.
`max_revisions=0` disables whole-CV/critic-driven revisions; it does not disable
the one permitted deterministic repair per entry. CLI dry-run reports both.

`selection_response*_entries/` checkpoints retain exact requests, schemas, raw
responses, local counts/errors and completion status. Completed entries assemble
into the same referenced-CV schema, with locally fixed reference IDs. The mode,
writing/repair prompts and attempt limit are included in cache identity. Failed checkpoints are
not automatically resumed as a complete CV.

A supervised local run on one reviewed plan produced a 540-word, two-page CV:
all 12 rewrite slots met their original caps in 13 Qwen calls, including one
successful length repair. Independent source review found no factual corrections
needed in that final draft. Earlier prompt variants failed wording or length checks,
so this is evidence of improvement on that case, not a general reliability rate.

### Separate local CV critic

Experimental: a live full-CV trial with Gemma 3 4B produced false factual
objections and hit output limits, including a 4,096-token retry. The feature's
separation, source-reference gates and bounded revision behavior are tested;
this model's critique quality is not validated for unattended use. Keep it opt-in
and inspect reports. A smaller, section-scoped review protocol needs separate
validation before treating it as a reliable automatic editor.

A same-input, nine-case entry-scoped check of Ministral 3 8B returned nine
structurally valid reports. It detected inflated numbers, a changed metric and
exaggerated ownership, but missed an invented CNN architecture. Some proposed
repairs were incomplete or introduced false objections. All entry coverage
scores were zero, so they are not evidence of CV improvement. This is a small
diagnostic set, not a measured general accuracy rate. The model is supported,
but automatic revision remains unvalidated; review its advice against sources.
A complete-CV follow-up completed within budget but falsely objected to an exact
source quotation and targeted the wrong entries in both suggestions. Valid JSON,
existing reference IDs and high scores do not make an edit safe to apply.

Optional independent feedback uses a different installed local model, for
example [Ministral 3 8B](https://ollama.com/library/ministral-3:8b), alongside Qwen 3 8B:

```text
Classifier match → Qwen draft → Ministral critique → Qwen revision → Ministral recheck
```

The critic does not decide APPLY/SKIP/REVIEW or replace the classifier. It receives
the complete source catalog, job description, current CV with canonical source
IDs, and deterministic quality checks. It returns the existing coverage/clarity/
factual rubric plus at most six targeted suggestions. Added or rewritten claims
must cite existing source IDs; any job quotation must match exactly. IDs establish
traceability, not proof that the model's interpretation is correct.

The prompt explicitly makes the master CV the only source of candidate facts;
the generated draft is unverified and the job is relevance context only. It asks
the critic to compare source/draft numbers, units, metric meanings and qualifiers
before scoring, then give at most two short edit instructions with exact source
excerpts. It forbids calculated percentages, inferred architecture/proficiency,
stronger ownership wording and reassigned outcomes. Generic comparison examples
illustrate these rules without supplying candidate facts. These are prompt
instructions, not proof of model compliance: inspect actual reports and retain
the factual/source gates. The schema still permits six suggestions for backwards
compatibility. Matching and the writer remain separate from this critique task.

Install the critic once:

```bash
ollama pull ministral-3:8b
```

Add these settings to `.env` (the critic is disabled when its model is blank):

```dotenv
CV_CRITIC_MODEL=ministral-3:8b
CV_CRITIC_NUM_CTX=12288
CV_CRITIC_NUM_PREDICT=4096
CV_CRITIC_TIMEOUT_SECONDS=900
OLLAMA_NUM_CTX=12288
OLLAMA_NUM_PREDICT=4096
OLLAMA_TIMEOUT_SECONDS=900
```

These are starting limits for sequential use on an 8 GB GPU, not a guarantee
that every CV/job fits or that inference finishes within the timeout. The local
role runtime validates installed Qwen 3/Gemma 3/Ministral 3 vocabularies and templates,
reserves the entire output budget, reconciles reported prompt counts, and rejects
budget overflow or incomplete output without dropping source text. Both models
use `keep_alive=0`, six CPU threads and batch size 128. It never downloads a model
or falls back to a cloud provider. Different tags pointing at identical model
weights do not count as an independent critic and are rejected.

Ministral uses separate system/user messages and no thinking option. Its standard
template has a default Le Chat/date preamble when no explicit system message is
provided; the guarded counter requires explicit system instructions to prevent
that hidden content. Structured CV requests always supply them. Plain `generate`
calls without a system message are rejected for this model. Its cached Tekken
vocabulary is inspected locally; no tokenizer assets are fetched automatically.
The standard 8B and 14B templates are separately allowlisted by exact normalized
hash. The inspected [14B template](https://ollama.com/library/ministral-3:14b/blobs/aa65b5aebf89)
differs only in its unused default model-name string; its vocabulary, merges and
special tokens match 8B. This does not permit arbitrary custom templates.
The `tokenizers` package is required for this counter (also installed by the
classifier requirements). Gemma 3 remains supported with `--critic-model gemma3:4b`,
but the failures described above are not bypassed or repaired by switching tags.

Revision prompts also include the previous draft and feedback. If the writer's
complete prompt plus its reserved output exceeds context, a more concise report
or an explicit smaller `OLLAMA_NUM_PREDICT` (for example `3072`, if sufficient
for a complete CV) can help. Increasing context uses more memory. Neither stage
silently reduces inputs or makes an automatic budget change.

```bash
.venv/bin/python app.py doctor --provider ollama --critic-model ministral-3:8b
.venv/bin/python app.py prepare --provider ollama --matcher classifier --critic-model ministral-3:8b --profile config/candidate_profile.json --job data/processed_jobs/JOB_ID.json --limit 1 --max-revisions 1 --no-enrich --dry-run
```

Remove `--dry-run` when ready to prepare that saved job. Doctor and dry-run
inspect configuration without contacting or loading either model; they are not
live model-availability tests. Matching-only, SKIP and ordinary REVIEW jobs make
no generation requests. A previous classifier REVIEW still needs the existing
explicit `--manual-approval-note` to run a drafting-only test; this does not make
the package eligible for submission. `--no-critic` overrides the environment and
restores the existing writer self-audit mode.

With `--max-revisions 1`, there are at most two writer calls and two critic calls;
`0` means draft plus critique only, and `2` allows one more pair. A substantive
source-supported suggestion can trigger revision even above the score threshold.
The workflow retains the best factually and deterministically valid audited draft
if a later revision scores lower. Invalid critic output is not passed to the
writer. Later failures retain the earlier draft for REVIEW, not automatic use.
Unresolved substantive criticism also blocks ready status after the budget ends.

The assessment records writer/critic identities, selected attempt, each attempt's
score, validation errors and artifacts. `attempt_01/`, `attempt_02/`, etc. retain
drafts, raw/canonical references, critic reports and writer feedback. The final
CV files correspond to the selected attempt; earlier files are not overwritten
by a lower-scoring revision. Scores are this critic's internal rubric, not ATS
scores or measured hiring probability. Compare drafts using the same critic;
do not interpret a new critic's score versus an old Qwen self-score as improvement.

### Larger models and reversed roles

Writer and critic roles are not tied to Qwen and Ministral respectively. With
installed, validated models, the same CLI supports explicit role selection:

```text
--provider ollama --model ministral-3:8b --critic-model qwen3:8b
--provider ollama --model ministral-3:14b --critic-model qwen3:14b
```

These are options to add to your existing `prepare` command, not standalone
commands. Keep `--matcher classifier`; entry-by-entry runs still require a
reviewed selection plan and `--max-revisions 0`. Check `ollama list` first;
preparation never downloads missing models. The writer reads `OLLAMA_*` limits
and the critic reads `CV_CRITIC_*` limits regardless of family or size. Keep a
separate critic selected for this guarded, family-adapted runtime: `--no-critic`
returns to the legacy self-audit provider path. A changed model requires fresh
output review, not comparison of scores as though they shared a calibrated scale.

On an 8 GB GPU, larger models can require partial CPU offload and substantial
system RAM, increasing latency. `ollama ps` shows CPU/GPU placement while a model
is loaded; see [Ollama's memory-placement explanation](https://docs.ollama.com/faq#how-can-i-tell-if-my-model-was-loaded-onto-the-gpu).
Context and output budgets still need to fit, and models unload between requests.
Neither a larger parameter count nor successful loading establishes better CV
quality, factual accuracy or critic reliability.

### Experimental per-entry factual reports (Python API)

[`entry_fact_critic.audit_entries`](../app/applications/entry_fact_critic.py) is a
separate experimental API, not a CLI mode. `--critic-model` continues to select
the existing full-CV critic; it does not activate this entry-scoped protocol.
The API accepts `ask`, `compiled_selection`, completed rewrite `values`, a new
empty `checkpoint_dir`, and optional `system_prefix`. It first validates the
complete draft against the plan, then makes one request per rewrite slot using
only that candidate entry and its assigned source lines. Fixed entries are not
sent. The caller supplies the model/runtime and must make `ask(task, schema, payload)`
use the documented exact system framing; no model is selected or
downloaded by this module.

Each report contains a factual-consistency flag and up to three issues, with
exact draft/source substrings and assigned source IDs. Schema or quotation
failures remain `needs_review`, not repaired or retried; other entries are still
checked once. Private checkpoints preserve requests, raw reports and input
bindings. `no_issue_reported` means only that no validated objection was reported,
not that every fact is correct. This API neither rewrites CVs nor changes scores,
classifier verdicts, application records, readiness or submission status. It is
not a replacement for reviewing the complete document and its layout.

### Local context and inference limits

CPU-only Qwen inference can be slow; long prompts may take minutes per request.
The defaults allow a 32,768-token context, at most 4,096 generated tokens, and a
600-second request timeout. These are limits, not a speed or completion promise.
CV text, job text, schema/instructions and output all need room in the context.
Evaluation includes both the master CV and proposed CV; revisions also include
the previous draft and feedback. Matching can fit while a later stage exceeds
the budget. If evaluation fails, the workflow records FAILED and does
not export the generated draft as an audited package. Explicit manual-tailoring
tests retain an unaudited Markdown/JSON checkpoint as described above.
This context can require substantial RAM/VRAM, especially when combined with
other loaded models. Raising context/output limits increases resource needs.
The provider checks its conservative input/output budget before sending a
request and rejects inputs that may not fit. A truncation or budget error
requires a deliberate budget change or a concise factual source;
do not remove mandatory job requirements to make an application pass.

The classifier checks source evidence using requirement rules and NLI, and the
workflow applies deterministic mandatory-constraint and score gates. The
pretrained model is not fine-tuned on your reviewed importance dataset. Complex
requirements, uncertain evidence or unsupported conditions can require REVIEW;
a numeric confidence alone cannot override them.
For narrowly recognized CV frames such as "Contributed to ..." or "Worked on
...", the NLI input makes the implicit candidate subject explicit. Ambiguous
participle openings such as "Supported ..." stay unchanged. Saved evidence
remains the exact original CV quote. Coherent achievements rank ahead of bare
keyword fragments; this formatting does not establish professional duration or
calibrated accuracy.
Simple OR/AND requirements are grouped in code. Supported shared skill-list
qualifiers are copied to every component hypothesis; matched components retain
their CV quotes as PARTIAL if other obligations remain unresolved. Open-ended
lists retain a coverage warning, even if every named component is supported.
Mixed or complex groups and numeric experience-duration comparisons still need
manual interpretation. Narrow education rules can establish a completed degree
with an allowed level and subject from explicit completion or a past dated
education entry. They do not verify credentials externally or infer grades,
accreditation, equivalence or completed study from enrolment.

Structured title/location evidence is labelled by `job_evidence_source` and
validated against the entire original field. Conflicting or conditional
description statements block a metadata-based pass. Sponsorship still needs an
explicit offer or an applicable verified exemption, not a location inference.
Employer background is not a candidate requirement; unknown sections retain
their coverage guard. A parenthesized language preference does not downgrade
the entire stack clause, and a clearly optional advantage is not mandatory.

The matcher
bounds work to 64 requirements, 180 CV text units, 640-character evidence windows,
8 candidate windows per atomic requirement and 384 NLI pairs per job. An NLI
pair must fit 256 tokenized input tokens; it is never silently truncated. Exceeded
coverage/input limits add review reasons instead of silently omitting evidence
and approving a job. These are conservative operating limits, not calibrated
accuracy guarantees. See [classifier_matching.py](../app/applications/classifier_matching.py)
and [requirement_nli.py](../app/classifiers/requirement_nli.py).

For an explicitly managed low-memory Qwen test, the Python provider also accepts
an `input_token_counter` built by
[`build_qwen_token_counter`](../app/providers/ollama_token_budget.py) from the
same installed model's local `show(verbose=True)` metadata. This does not download
a tokenizer or load model weights. Only the inspected standard Qwen3 template
and validated vocabulary are accepted. The caller must pin/check the model's
digest between inspection and inference. The provider adds its 512-token chat
reserve, reserves the complete output budget and checks Ollama's reported token
count. Its default byte-based guard is unchanged. `keep_alive=0` can unload the
model between drafting and auditing; this does not eliminate CPU memory needs.
These are opt-in Python controls, not automatic CLI behavior or an accuracy claim.

CV drafting uses a complete, numbered catalog of nonblank master-CV lines.
Qwen returns `ReferencedCV` entries with `evidence_ids`; the application resolves
those IDs into exact source quotations in the saved `TailoredCV`. It never asks
Qwen to construct quotations by joining lines. Unknown or malformed IDs fail
closed. `cv_sources.json` and `cv_references_N.json` retain the mapping for review.
If an entry exactly equals a complete original source line but Qwen cited other
valid lines, the application can relink it to that exact source. It never uses
fuzzy matching or alters the claim. `cv_model_references_N.json` preserves the raw
model output; `evidence_reconciliations` records each change. Invalid IDs still
fail before reconciliation. Paraphrases still need a semantic support check.
This proves where a quote came from, **not** that it supports the entire claim.
The separate audit receives the full original master CV and the proposed entries
with their resolved evidence, and must check both factual meaning and selection.

Drafting targets a concise two-page CV, selecting relevant projects and preserving
independent-work qualifiers. Deterministic limits are 650 words, a 70-word summary,
five Skills entries and 32 non-contact entries; exceeding a limit keeps the
document in REVIEW regardless of self-score. These limits do not guarantee the
rendered page count, so inspect the PDF. Role/project labels stay with the first
following bullet where possible. The original master is never shortened in place.
The generation policy and prompt version are part of the preparation cache key.
Prompt limits alone are not reliable with a small local model. The live Qualcomm
test needed an assistant-reviewed source-selection plan before Qwen produced a
concise draft; that guided result is not evidence of reliable autonomous selection.

The workflow checks verbatim evidence, mandatory constraints and match scores.
Only APPLY outside match-only mode normally proceeds to CV generation; the explicit
manual REVIEW override described above authorizes a draft only. Ready status requires valid source quotes,
a successful factual audit, no reported unsupported claims or missing critical
information, and the configured CV score. Unknown required information leads to
REVIEW. The evaluation is another request to the same configured local model;
it is not an independent factual guarantee or an employer ATS score.

Identical unattempted preparations can be cached. A match-only result cannot
stand in for a complete CV preparation, and matcher configuration is part of the
cache identity. `--retry` requests a new
assessment but does not release an opened/submitted application. Provider/model
changes do not bypass the submission history. Keep using the same ledger for
the same candidate when switching providers.

### Review the package and apply

Ready packages contain `cv.md`, `cv.pdf`, `cv.docx` and `assessment.json`; the
dashboard is `data/applications/applications.html`. Inspect them first:

```bash
.venv/bin/python app.py list
.venv/bin/python app.py show APPLICATION_ID
```

For browser assistance, install Chromium once if it is not already available:

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/.venv/playwright-browsers" .venv/bin/python -m playwright install chromium
.venv/bin/python app.py open APPLICATION_ID
```

The visible browser requires a graphical desktop. Recognized Greenhouse/Lever
forms receive explicit profile values and your generated PDF; complete unknown
questions yourself. `open APPLICATION_ID --submit` requests submission only for
recognized complete simple forms. CAPTCHA, custom questions, unresolved fields
and changed layouts require manual completion. Browser actions use the shared
code and selectors, not Qwen-generated browser commands.

An attempt is recorded before the browser opens. A submission is marked complete
only with an employer receipt or a confirmation you verified yourself:

```bash
.venv/bin/python app.py mark-submitted APPLICATION_ID --confirmation "Employer receipt I verified"
```

If no submission occurred, first verify the employer's state and then release
the saved attempt with `resolve-not-submitted APPLICATION_ID --note "..."`.
Never infer failure solely from an absent receipt. The
[browser/outcome guide](OPENAI_MODE.md#5-review-and-apply) describes the shared
submission protections in detail.

## 1. Install Python dependencies for extraction and research

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
For local application preparation, also install
`requirements-local-applications.txt` plus `requirements-classifier.txt`; these
add document/browser dependencies and the default matching baseline. The OpenAI requirement file is
optional and only needed when selecting that provider.

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
without Internet access after the server and model are installed. This
application's provider restricts `host` to a local loopback endpoint.

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
The provider has explicit context, response-token and timeout settings and
disables thinking for these structured calls. `generate(prompt)` also exists,
but returns ordinary text. Batch orchestration is provided by the application
workflow; this extraction example makes one request.

### Configuration and overrides

The provider reads `.env` and the Ollama environment settings described in the
application quickstart. Model and host fall back to `qwen3:8b` and
`http://localhost:11434`; explicit constructor arguments override them. Server
process configuration and Python client settings are separate: configuring the
client does not start the server or download a model.

The Python API above remains useful for standalone extraction. For the complete
CV/application workflow, use `app.py prepare --provider ollama`.

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
The standalone collection command does not automatically perform the application
workflow's full-description enrichment step; `app.py prepare` adds that step for
either provider.

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

The rule and semantic-relevance modules remain exploratory components. The
application matcher also reuses the NLI component as a pretrained baseline;
none of this is a custom trained five-label or CV-match checkpoint:

| Component | Actual output and limitation |
| --- | --- |
| [importance_rule.py](../app/classifiers/importance_rule.py) | Keyword rules returning lowercase `core`, `preferred`, `contextual`, or `ambiguous`; this is a different taxonomy from the five-label dataset |
| [requirement_nli.py](../app/classifiers/requirement_nli.py) | Pretrained NLI label/probabilities for a context/hypothesis pair; reused by classifier application matching, without claiming trained CV-match accuracy |
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

These dependencies can be large. The application baseline needs
`requirements-classifier.txt`; Sentence Transformers and the JobBERT experiment
are additional research choices, not requirements for application matching. See also the
[Sentence Transformers installation guide](https://www.sbert.net/docs/installation.html).

The repository's executable experiments are:

```bash
.venv/bin/python -m tests.test_model
.venv/bin/python -m tests.test_requirement_nli
.venv/bin/python -m tests.test_semantic_inportance
.venv/bin/python -m tests.test_jobbert
```

The spelling `test_semantic_inportance` matches the existing filename. The first
command calls Ollama. The other three use pretrained classifier models. The
application NLI baseline uses cached files only, defaults to CPU and rejects
oversized input pairs instead of silently truncating them. Separate legacy
semantic/JobBERT experiments may download model files when run. These scripts
print examples; they are not a benchmark over the datasets built above.

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
| `No module named ollama`, `dotenv`, or `bs4` | Install `requirements-local-applications.txt` for applications, or `requirements-local.txt` for extraction/research, using the same `.venv/bin/python` as the command |
| Ollama connection refused | Start the server or service and ensure `host=` points to its listening address |
| Port 11434 already in use | An Ollama instance may already be running; use it instead of starting a second server |
| Ollama model not found | Pull the exact tag passed as `model=` and check `ollama ls` |
| Remote Ollama URL rejected | Use a loopback endpoint on this machine; local preparation does not permit a remote/cloud fallback |
| OpenAI configuration requested unexpectedly | Pass `--provider ollama` explicitly and check `APPLICATION_PROVIDER`; its fallback is `openai` when unset |
| Classifier libraries or weights unavailable | Install `requirements-classifier.txt` and supply the complete cached NLI model/tokenizer or a local directory; matching never downloads or switches to Qwen automatically |
| MATCHED but no CV files | `--match-only` deliberately stops after the verdict; rerun without that flag to request CV preparation |
| Many classifier REVIEW outcomes | Read source evidence, unresolved constraints and diagnostics; the pretrained baseline is conservative and is not a custom trained CV-match model |
| Slow inference or memory exhaustion | Inspect `ollama ps`; reduce workload/context or explicitly choose a smaller downloaded model |
| Context/output limit or incomplete-response error | Check `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT` and input size; keep complete role requirements and source facts |
| Request timeout | Check local server/model availability and runtime; CPU inference may need more time than your configured `OLLAMA_TIMEOUT_SECONDS` |
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

The application workflow defaults to classifier matching with Qwen reserved for
CV writing/audit when `--provider ollama` is selected. It does not replace human
annotation or train the experimental classifiers. Standalone local
extraction output is not automatically promoted into the reviewed dataset.
Model-based matching and factual audits can still be wrong. Unsupported browser
forms and unknown application answers remain manual work, even when the CV is
ready.

Keep source credentials, personal CVs, model caches and generated data out of
Git. Review the repository's ignore rules before publishing any corpus: scraped
job text and annotation notes may need separate handling from reusable code and
the small test fixtures. Return to the [project overview](../README.md) for both
modes and repository setup.
