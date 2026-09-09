# Job Application Agent — Multi-Source Job Ingestion Implementation Plan

## Objective

Extend the existing `Job_Application_Agent` project so it can collect real-world job descriptions from multiple structured job sources, normalize them into a common representation, deduplicate them, store both raw and processed data, and prepare a high-quality corpus for training and evaluating the custom requirement classifier.

The implementation should be **API/ATS-first** rather than relying primarily on fragile HTML scraping.

The initial source priority is:

1. Arbeitnow UK
2. Adzuna
3. Greenhouse
4. Lever
5. The Muse
6. Remotive

Existing fallback scrapers should remain available:

- Generic JSON-LD extraction
- Generic HTML extraction
- DWP adapter, but DWP should be treated as experimental / best-effort because the service has been returning 503 responses

Do **not** implement LinkedIn or Google Jobs scraping in this phase.

---

# Existing Project Structure

The project currently looks like this:

```text
Job_Application_Agent/
│
├── app/
│   │
│   ├── classifiers/
│   │   ├── importance_rule.py
│   │   ├── requirement_nli.py
│   │   ├── semantic_relevance.py
│   │   └── requirement_classifier.py
│   │
│   ├── providers/
│   │   ├── base.py
│   │   ├── huggingface.py
│   │   ├── llama_cpp.py
│   │   ├── ollama_provider.py
│   │   └── openai.py
│   │
│   ├── services/
│   │   ├── job_extractor.py
│   │   ├── metadata_extractor.py
│   │   └── job_service.py
│   │
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── crawler.py
│   │   ├── extractor.py
│   │   ├── cleaner.py
│   │   ├── deduplicator.py
│   │   ├── storage.py
│   │   │
│   │   └── sources/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── generic_jsonld.py
│   │       ├── company_careers.py
│   │       └── dwp.py
│   │
│   ├── job_parser.py
│   ├── models.py
│   └── scoring.py
│
├── data/
│   ├── candidate_profile.yaml
│   ├── jobs.db
│   ├── raw_jobs/
│   ├── processed_jobs/
│   ├── labelled/
│   └── evaluation/
│
├── scripts/
│   ├── __init__.py
│   ├── scrape_jobs.py
│   ├── process_jobs.py
│   ├── build_dataset.py
│   └── evaluate_classifier.py
│
├── tests/
│   ├── __init__.py
│   ├── test_jobbert.py
│   ├── test_model.py
│   ├── test_requirement_nli.py
│   ├── test_semantic_importance.py
│   │
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── test_extractor.py
│   │   ├── test_cleaner.py
│   │   └── test_deduplicator.py
│   │
│   └── fixtures/
│       └── sample_job_pages/
│
├── pytest.ini
├── .env
└── app.py
```

---

# Existing Working Behaviour

The following pipeline already works for individual URLs:

```text
URL
 ↓
HTTP Fetch
 ↓
Raw HTML
 ↓
JSON-LD extraction if available
 ↓
Generic HTML fallback if needed
 ↓
Minimal cleaning
 ↓
URL canonicalization
 ↓
Deduplication
 ↓
Save raw HTML
 ↓
Save processed JSON
```

The scraper has already successfully extracted a real Huxley job posting.

The system also correctly canonicalizes tracking URLs such as:

```text
/job/123/
/job/123/?utm_source=indeed&utm_medium=cpc
```

into the same stored job ID.

Do not break this behaviour.

---

# Key Design Principle

Use a normalized internal job representation so downstream classifier logic does not care where the job came from.

The intended flow is:

```text
Source API / ATS / webpage
          ↓
     JobReference
          ↓
      Raw source data
          ↓
     ExtractedJob
          ↓
   Minimal normalization
          ↓
      Deduplication
          ↓
        Storage
          ↓
  Requirement classifier
```

---

# Phase 1 — Add Shared Models

Update `app/scraper/sources/base.py`.

Keep the existing `RawJobPage` and `ExtractedJob`, but introduce a `JobReference`.

Example:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class JobReference:
    source: str
    url: str

    external_id: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None


@dataclass
class RawJobPage:
    url: str
    html: str
    source: str
    status_code: int = 200


@dataclass
class ExtractedJob:
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    date_posted: Optional[str] = None
    salary: Optional[str] = None

    source: Optional[str] = None
    source_url: Optional[str] = None

    external_id: Optional[str] = None
    raw_description: Optional[str] = None

    metadata: Optional[dict] = None
```

`metadata` should be used for source-specific data we may want to retain without polluting the core model.

Examples:

```json
{
  "remote": true,
  "visa_sponsorship": true,
  "employment_type": "full_time"
}
```

---

# Phase 2 — Introduce a Source Contract

The current `JobSource` interface should be extended so each source can support discovery.

Use an interface roughly like:

```python
from abc import ABC, abstractmethod


class JobSource(ABC):
    name: str

    @abstractmethod
    def discover_jobs(
        self,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobReference]:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, url: str):
        raise NotImplementedError

    @abstractmethod
    def extract(self, raw) -> ExtractedJob:
        raise NotImplementedError
```

Important:

API-based sources may not use HTML pages at all.

Therefore, `fetch()` and `extract()` should not be tightly coupled to `RawJobPage`.

Consider introducing a more generic object:

```python
@dataclass
class RawJobPayload:
    source: str
    source_url: str
    data: object
    content_type: str
```

Possible `content_type` values:

```text
application/json
text/html
```

Do not overengineer this if the current code becomes significantly harder to maintain. A source-specific implementation is acceptable as long as the normalized output is always `ExtractedJob`.

---

# Phase 3 — Arbeitnow UK Adapter

Create:

```text
app/scraper/sources/arbeitnow.py
```

This should be the first source implemented.

## Requirements

The adapter should:

- call the Arbeitnow UK job API
- retrieve job listings
- support pagination if exposed by the API
- support a configurable result limit
- optionally use a visa sponsorship filter if supported
- convert every remote job record into `ExtractedJob`
- preserve the original job URL
- preserve raw API data where useful
- avoid unnecessary page fetching if the API already provides the full job description

## Suggested class

```python
class ArbeitnowSource(JobSource):
    name = "arbeitnow"
```

## Required methods

```python
discover_jobs(...)
fetch_job(...)
extract(...)
```

If the API returns full job descriptions in the listing response, it is acceptable for `discover_jobs()` to return references plus a cache of raw payloads internally.

Prefer simple, testable code over architectural cleverness.

## Normalization mapping

Map source fields into:

```text
title
company
location
description
date_posted
source
source_url
external_id
metadata
```

If the API exposes:

```text
remote
visa_sponsorship
tags
employment type
```

store these in `metadata`.

---

# Phase 4 — Adzuna Adapter

Create:

```text
app/scraper/sources/adzuna.py
```

Adzuna requires credentials.

Read from `.env`:

```text
ADZUNA_APP_ID=
ADZUNA_APP_KEY=
```

Do not hardcode secrets.

## Requirements

Support:

```text
query
location
limit
salary_min
salary_max
full_time
permanent
```

Where practical.

The generic source contract may only expose:

```python
discover_jobs(query, location, limit)
```

Source-specific filters can be optional constructor or method arguments.

Example:

```python
source = AdzunaSource(
    salary_min=60000,
    permanent=True,
)
```

Normalize returned jobs into `ExtractedJob`.

If the API only returns a summary description and the original employer page is available, preserve:

```text
description_type = "summary"
```

inside `metadata`.

Do not silently pretend a short snippet is a complete JD.

---

# Phase 5 — Greenhouse Adapter

Create:

```text
app/scraper/sources/greenhouse.py
```

Greenhouse should be treated as an ATS source.

A Greenhouse board is employer-specific.

Example concept:

```python
source = GreenhouseSource(
    board_token="company-name"
)
```

## Requirements

Support retrieving all published jobs for a configured board.

Where possible request full content.

Normalize:

```text
job id
title
location
description/content
updated timestamp
absolute URL
```

The company name may be supplied in configuration because a Greenhouse board endpoint may not always expose it directly.

Possible constructor:

```python
GreenhouseSource(
    board_token="example",
    company="Example Ltd",
)
```

---

# Phase 6 — Lever Adapter

Create:

```text
app/scraper/sources/lever.py
```

Lever is also employer-specific.

Suggested configuration:

```python
LeverSource(
    company_slug="example"
)
```

Normalize each published posting into the shared schema.

Preserve useful fields such as:

```text
team
department
workplace type
commitment
```

inside `metadata`.

---

# Phase 7 — The Muse Adapter

Create:

```text
app/scraper/sources/the_muse.py
```

Read any API key from `.env`.

Example:

```text
THE_MUSE_API_KEY=
```

Support query/category/location parameters where the API supports them.

Normalize results into `ExtractedJob`.

---

# Phase 8 — Remotive Adapter

Create:

```text
app/scraper/sources/remotive.py
```

This source will primarily add remote job diversity.

Normalize:

```text
title
company
candidate location
description
publication date
employment type
job URL
category
salary if available
```

Preserve source-specific metadata.

---

# Phase 9 — Source Registry

Create:

```text
app/scraper/source_registry.py
```

Purpose:

centralize source creation.

Example:

```python
SOURCE_REGISTRY = {
    "arbeitnow": ArbeitnowSource,
    "adzuna": AdzunaSource,
    "dwp": DwpSource,
}
```

For ATS sources needing employer configuration, do not blindly instantiate without required arguments.

A factory is preferable:

```python
def build_source(name: str, **kwargs) -> JobSource:
    ...
```

Unknown sources should raise a useful exception.

---

# Phase 10 — Batch Collection Service

Create:

```text
app/services/job_collection_service.py
```

The script should not contain all collection logic.

Move orchestration into a reusable service.

Suggested class:

```python
class JobCollectionService:
    def __init__(
        self,
        storage: JobStorage,
        cleaner: JobCleaner,
        deduplicator: JobDeduplicator,
    ):
        ...
```

Suggested method:

```python
collect(
    source,
    query,
    location,
    limit,
)
```

Responsibilities:

1. discover jobs
2. fetch/extract each job
3. minimally clean data
4. validate quality
5. deduplicate
6. save processed data
7. save raw payload where practical
8. continue if one individual job fails
9. return a collection summary

---

# Phase 11 — Collection Result Summary

Return something like:

```python
@dataclass
class CollectionSummary:
    source: str
    discovered: int = 0
    saved: int = 0
    duplicates: int = 0
    failed: int = 0
    rejected: int = 0
```

This should allow terminal output such as:

```text
Source: arbeitnow

Discovered: 100
Saved:       87
Duplicates:   8
Rejected:     2
Failed:       3
```

When multiple sources are collected:

```text
Arbeitnow     87 saved
Adzuna        61 saved
Greenhouse    42 saved
Lever         31 saved

Total        221 saved
```

---

# Phase 12 — Source Failure Handling

Create custom exceptions.

Suggested file:

```text
app/scraper/exceptions.py
```

Example:

```python
class ScraperError(Exception):
    pass


class SourceUnavailableError(ScraperError):
    pass


class ExtractionError(ScraperError):
    pass


class InvalidJobError(ScraperError):
    pass
```

Persistent HTTP errors such as:

```text
429
500
502
503
504
```

should not crash the entire multi-source collection run.

The source should be marked failed/unavailable and processing should continue with other sources.

---

# Phase 13 — Preserve Realistic Noise

Important training-data principle:

> Preserve semantic noise; remove extraction noise.

Do **not** aggressively sanitize job descriptions.

Keep realistic content such as:

```text
Apply now
Recruiter boilerplate
Benefits text
Company marketing text
Inconsistent punctuation
Odd salary formatting
Irrelevant technology mentions
```

These are useful because the classifier should learn to distinguish requirement-bearing text from irrelevant text.

Remove only obvious extraction garbage such as:

```text
cookie menus
navigation
script text
login controls
header/footer duplication caused by parsing mistakes
```

Maintain:

```text
raw_description
description
```

Where:

```text
raw_description
    = extracted job-region content before normalization

description
    = minimally normalized text used by classifiers
```

Minimal normalization may include:

```text
HTML removal
HTML entity decoding
whitespace normalization
```

Do not rewrite the prose.

---

# Phase 14 — Quality Gate

Implement a basic quality validator.

Create:

```text
app/scraper/quality.py
```

Suggested checks:

```python
title exists
description exists
description length >= 300 characters
```

Optional later checks:

```text
description-to-navigation ratio
duplicate-line ratio
language detection
suspiciously short job body
```

Do not reject a JD merely because it contains boilerplate.

A result like:

```text
Back to search
Data Scientist
Apply
```

should be rejected because extraction failed.

A full job advertisement containing recruiter/legal text should be accepted.

---

# Phase 15 — Deduplication

Existing URL canonicalization must remain.

Deduplication should operate at multiple levels.

## Level 1 — URL duplicate

Remove:

```text
utm_source
utm_medium
utm_campaign
utm_term
utm_content
gclid
fbclid
msclkid
```

and canonicalize trailing slashes.

## Level 2 — Exact content duplicate

Hash normalized:

```text
title
company
location
description
```

## Level 3 — Cross-source near duplicates

Do not implement fuzzy matching yet unless it is easy to isolate.

Future design may use:

```text
title similarity
company similarity
location similarity
description similarity
```

or embeddings.

For now, avoid accidentally merging legitimate related vacancies.

---

# Phase 16 — Storage

Keep existing:

```text
data/raw_jobs/
data/processed_jobs/
```

Processed job JSON should include:

```json
{
  "id": "...",
  "external_id": "...",
  "title": "...",
  "company": "...",
  "location": "...",
  "raw_description": "...",
  "description": "...",
  "date_posted": "...",
  "salary": "...",
  "source": "arbeitnow",
  "source_url": "...",
  "canonical_url": "...",
  "content_hash": "...",
  "scraped_at": "...",
  "metadata": {}
}
```

Raw API responses may be saved as:

```text
data/raw_jobs/<job-id>.json
```

HTML sources remain:

```text
data/raw_jobs/<job-id>.html
```

---

# Phase 17 — CLI

Refactor:

```text
scripts/scrape_jobs.py
```

so it supports both single-URL scraping and source discovery.

## Existing mode

```bash
python -m scripts.scrape_jobs \
"https://example.com/job/123"
```

This must continue working.

## Source mode

Example:

```bash
python -m scripts.scrape_jobs \
    --source arbeitnow \
    --query "data engineer" \
    --location "United Kingdom" \
    --limit 100
```

Example:

```bash
python -m scripts.scrape_jobs \
    --source adzuna \
    --query "senior data engineer" \
    --location "London" \
    --limit 100
```

Eventually:

```bash
python -m scripts.scrape_jobs \
    --sources arbeitnow adzuna \
    --query "data engineer" \
    --location "United Kingdom" \
    --limit-per-source 100
```

Do not implement complicated CLI syntax before the basic single-source paths work.

---

# Phase 18 — Tests

Add source-specific tests.

Suggested structure:

```text
tests/
└── scraper/
    ├── test_arbeitnow.py
    ├── test_adzuna.py
    ├── test_greenhouse.py
    ├── test_lever.py
    ├── test_the_muse.py
    ├── test_remotive.py
    ├── test_quality.py
    ├── test_storage.py
    └── test_collection_service.py
```

Do not make unit tests depend on live APIs.

Use saved JSON fixtures.

Example:

```text
tests/fixtures/api/
├── arbeitnow_page_1.json
├── adzuna_page_1.json
├── greenhouse_jobs.json
├── lever_jobs.json
└── remotive_jobs.json
```

Mock HTTP calls using:

```text
unittest.mock
```

or a lightweight HTTP mocking library already available in the project.

Do not introduce a large dependency just for mocking unless necessary.

---

# Phase 19 — Integration Tests

Keep live API tests separate.

Example:

```text
tests/integration/
```

Mark them:

```python
@pytest.mark.integration
```

They should not run as part of the normal test suite.

Normal:

```bash
python -m pytest
```

Live:

```bash
python -m pytest -m integration
```

---

# Phase 20 — Training Dataset Preparation

Do not immediately feed all scraped JDs into training.

Later, `scripts/build_dataset.py` should produce structured requirement-classification examples.

Target schema:

```json
{
  "job_id": "...",
  "job_title": "Senior Data Engineer",
  "job_description": "...",
  "requirement": "PySpark",
  "label": "core",
  "evidence_span": "You will build scalable pipelines using PySpark..."
}
```

Initial labels:

```text
CORE
IMPORTANT
PREFERRED
CONTEXTUAL
NOT_REQUIREMENT
```

The classifier must eventually learn distinctions such as:

```text
"Strong PySpark experience is essential."
→ CORE

"You will build pipelines using PySpark."
→ CORE or IMPORTANT

"PySpark experience would be advantageous."
→ PREFERRED

"Our platform uses PySpark."
→ CONTEXTUAL

"You will work alongside a team using PySpark."
→ CONTEXTUAL

"No previous PySpark experience is required."
→ NOT_REQUIREMENT
```

Do not train the model as part of this ingestion implementation unless explicitly requested later.

---

# Phase 21 — Evaluation Corpus

Reserve part of the scraped dataset for evaluation.

Suggested folders:

```text
data/evaluation/clean/
data/evaluation/real_world/
data/evaluation/noisy/
```

The important principle is that the classifier should not only perform well on pristine job descriptions.

It should also handle:

```text
recruiter boilerplate
salary noise
duplicated headings
irrelevant technology mentions
odd formatting
marketing copy
```

---

# Phase 22 — Logging

Replace development `print()` calls gradually with Python `logging`.

Recommended pattern:

```python
logger = logging.getLogger(__name__)
```

Use:

```text
INFO    normal collection progress
WARNING individual failed job
ERROR   source unavailable
DEBUG   URLs and raw extraction diagnostics
```

Do not log API keys.

---

# Phase 23 — Configuration

Create:

```text
config/
└── job_sources.yaml
```

if useful.

Possible format:

```yaml
sources:

  arbeitnow:
    enabled: true

  adzuna:
    enabled: true

  dwp:
    enabled: false

  greenhouse:
    enabled: false
    boards: []

  lever:
    enabled: false
    companies: []

  remotive:
    enabled: true
```

Avoid making configuration mandatory in the first implementation if it slows down the initial Arbeitnow adapter.

---

# Phase 24 — Implementation Order

Implement in this exact order.

## Milestone 1

Arbeitnow only.

```text
Arbeitnow API
   ↓
discover 5 jobs
   ↓
normalize
   ↓
quality gate
   ↓
deduplicate
   ↓
save
```

Command:

```bash
python -m scripts.scrape_jobs \
    --source arbeitnow \
    --query "data engineer" \
    --limit 5
```

Do not proceed until this works.

---

## Milestone 2

Add tests for Arbeitnow.

Required:

```text
JSON parsing
normalization
pagination
storage
duplicate handling
quality rejection
```

---

## Milestone 3

Add Adzuna.

---

## Milestone 4

Add Greenhouse.

Use one known board as a test fixture.

---

## Milestone 5

Add Lever.

---

## Milestone 6

Add The Muse and Remotive.

---

## Milestone 7

Create multi-source orchestration.

---

# Important Constraints

1. Do not break the existing Huxley generic HTML extraction.

2. Do not remove the existing JSON-LD fallback.

3. Do not aggressively clean job-description text.

4. Do not hardcode API credentials.

5. Do not scrape LinkedIn in this phase.

6. Do not scrape Google Jobs in this phase.

7. DWP should remain optional / best-effort.

8. A failed source must not terminate a multi-source run.

9. Unit tests must not make live network requests.

10. Prefer structured APIs over browser automation.

11. Avoid Playwright/Selenium unless a future source truly requires browser rendering.

12. Avoid anti-bot circumvention.

13. Keep each source adapter isolated.

14. All source-specific records must normalize into `ExtractedJob`.

15. Preserve source metadata where useful.

---

# Definition of Done for This Task

The first implementation is complete when all of the following work:

```bash
python -m scripts.scrape_jobs \
    --source arbeitnow \
    --query "data engineer" \
    --limit 5
```

and the output produces several valid files under:

```text
data/processed_jobs/
```

Each saved record should contain:

```text
title
company
location
description
source
source_url
canonical_url
content_hash
scraped_at
```

and where available:

```text
date_posted
salary
external_id
raw_description
metadata
```

The normal test suite must pass:

```bash
python -m pytest
```

The implementation should produce a terminal collection summary similar to:

```text
Source: arbeitnow

Discovered: 5
Saved:      5
Duplicates: 0
Rejected:   0
Failed:     0
```

After completing Arbeitnow, stop and report:

1. files created
2. files modified
3. any assumptions made
4. sample CLI command
5. test output
6. one example normalized job record

Do not continue automatically to Adzuna until the Arbeitnow implementation is confirmed working.

---

# Human Review Semantics

Requirement candidates use three review states:

- `approved`: the requirement/evidence pair is valid. If `reviewed_label` is
  empty, the proposed label is accepted. Set `reviewed_label` to override an
  incorrect proposal.
- `rejected`: the requirement/evidence pair itself is invalid and must not be
  included in the dataset. Do not use rejection only to correct a label.
- `needs_review`: no final human decision has been made.

Status and label values are read case-insensitively, but generated files use
canonical lowercase statuses and uppercase labels. `review_notes` are retained
on promoted annotations and in the decision log. Notes provide audit context;
they do not silently override the structured final label.

Audit a queue before promotion with:

```bash
python -m scripts.process_jobs \
    --validate-reviews data/labelled/requirement_candidates.json \
    --review-report data/labelled/review_validation.json
```

The audit flags rejected records whose notes look like label corrections and
approved records whose notes mention alternatives not captured by an explicit
`reviewed_label`.

## Alternative groups and coverage gate

Requirements that appear in an explicit choice list use
`requirement_group`, `group_operator`, `group_members`, and `group_label`.
`ANY_OF` means satisfying one member satisfies the group; `ALL_OF` means every
member is required. The importance label applies to the group. Grouped atomic
members are excluded from explicit-negation counterfactual generation because
negating one member does not negate an alternative group.

Before training, promote reviewed annotations and generate the coverage gate:

```bash
python -m scripts.report_dataset_coverage \
    --input data/labelled/requirements.jsonl \
    --output data/labelled/coverage_report.json
```

Training remains blocked until the report meets its configured label, role,
unique-job, source-diversity, and maximum-label-share targets.

Candidate generation recognizes both explicit sentence cues and common
requirements-section headings. Generic phrases such as `experience with` and
responsibility verbs use lower proposal confidence and always remain subject to
human review. Use `--include-candidates` with `scripts.process_jobs` to combine
real and synthetic candidates into one deduplicated review queue.
