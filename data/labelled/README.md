# Requirement dataset annotations

The dataset builder deliberately consumes reviewed annotations rather than
treating automatically scraped text as ground truth. Put one JSON object per
line in `requirements.jsonl`:

```json
{"job_id":"abc123","job_title":"Senior Data Engineer","job_description":"Strong SQL experience is essential.","requirement":"SQL","label":"CORE","evidence_span":"Strong SQL experience is essential.","annotation_source":"human","role_family":"data_engineering","source":"adzuna"}
```

## Labels

- `CORE`: explicitly mandatory, essential, or a must-have.
- `IMPORTANT`: expected in responsibilities or strongly requested, but not
  explicitly mandatory.
- `PREFERRED`: desirable, advantageous, a bonus, or nice to have.
- `CONTEXTUAL`: mentioned as part of the environment, team, or surrounding
  organisation without asking the candidate to possess it.
- `NOT_REQUIREMENT`: explicitly not required, negated, or an irrelevant mention.

Use the shortest canonical requirement name (`PySpark`, not `Strong PySpark
experience`). Copy `evidence_span` verbatim from `job_description`. If wording
is genuinely ambiguous, do not guess: leave it out for adjudication.

## Review queue

Generate conservative candidates from collected jobs:

```bash
python -m scripts.process_jobs
```

To review real and counterfactual candidates in one consolidated queue, merge
the counterfactual file during generation:

```bash
python -m scripts.process_jobs \
  --input-dir data/processed_jobs \
  --output data/labelled/requirement_candidates.json \
  --include-candidates data/labelled/not_requirement_candidates.json
```

Regeneration deduplicates by `candidate_id` and preserves prior statuses,
reviewed labels, notes, and requirement-group metadata. Counterfactual records
retain their evidence-only `job_description`, so the combined readable JSON can
be promoted directly after review.

Every candidate starts with `annotation_status: "needs_review"` and has no
reviewed label. Set `annotation_status` to `"approved"` to accept a valid
requirement/evidence pair. Leave `reviewed_label` empty to accept the proposed
label, or set it to one of the five labels to record a correction. Use
`"rejected"` only when the extracted requirement/evidence pair is invalid;
label-only corrections belong on approved records. Add `review_notes` whenever
surrounding context or conditional logic affects the decision.

The readable `.json` queue intentionally omits the full job description. It
contains a short `review_context` around the evidence and a
`job_description_ref` pointing to the processed job. Promotion reloads the full
description from that reference, verifies the job ID, and validates the exact
evidence span before producing a training annotation. The compact `.jsonl`
backup retains the full description for machine-oriented processing.
`review_context` is an array, so each original newline is displayed as a
separate readable JSON line rather than an escaped `\n` inside one long string.
Regenerating the same queue preserves existing review status, reviewed labels,
and review notes by stable candidate ID.

Promote only explicitly approved records into the training annotation schema:

```bash
python -m scripts.process_jobs \
  --promote-approved data/labelled/requirement_candidates.json \
  --approved-output data/labelled/requirements.jsonl
```

The proposed label is accepted only after an explicit approval status.
Detailed `review_notes` are preserved on approved annotations and dataset
records. Rejected notes are retained in `review_decisions.json`, together with
counts by rejection rule and proposed-to-reviewed label corrections. Notes are
audit and rule-improvement metadata; they are deliberately not concatenated
into classifier input text, which would leak reviewer reasoning and labels.

## Alternative requirement groups

When a job accepts any member of a list, the label applies to the group rather
than making every member independently mandatory. Keep the atomic requirement
and add validated group metadata:

```json
{
  "requirement": "Java",
  "label": "IMPORTANT",
  "requirement_group": "programming language",
  "group_operator": "ANY_OF",
  "group_members": ["C#", "Java", "Python", "SQL", "Scala"],
  "group_label": "IMPORTANT"
}
```

`group_label` must match `label`, the atomic requirement must occur in
`group_members`, and a group must contain at least two distinct members.
`ALL_OF` is also supported for evidence that explicitly requires every member.
Atomic counterfactual generation skips grouped members because negating one
acceptable alternative would not negate the group requirement.

## Counterfactual negative candidates

After real positive annotations have been approved, generate explicit negated
counterparts to help fill the `NOT_REQUIREMENT` class:

```bash
python -m scripts.generate_counterfactuals \
  --input data/labelled/requirements.jsonl \
  --output data/labelled/not_requirement_candidates.json \
  --variants-per-example 1 \
  --seed 42
```

Counterfactuals are generated only from reviewed `CORE`, `IMPORTANT`, or
`PREFERRED` annotations. They retain the original `job_id` so the source and
counterfactual cannot cross the train/evaluation boundary. Each record includes
`is_synthetic`, `synthetic_type`, `source_example_id`, and `source_label`.

Review and approve these candidates independently, then promote them to a
separate annotation file:

```bash
python -m scripts.process_jobs \
  --promote-approved data/labelled/not_requirement_candidates.json \
  --approved-output data/labelled/requirements.counterfactual.jsonl
```

Pass both reviewed files to the dataset builder. By default, construction fails
if synthetic annotations exceed 40% of the source corpus.

## Quality rules

- Have two reviewers inspect ambiguous and conflicting examples.
- Keep all labels represented across multiple employers and job sources.
- Avoid letting one employer, template, title, or technology dominate a label.
- Do not place variants of the same job in both training and evaluation.
- Review the generated `manifest.json` before training.
- Check `role_distribution` and `source_distribution` in the manifest so
  diversity is measured across both occupations and job providers.

Generate a pre-training coverage report after promotion:

```bash
python -m scripts.report_dataset_coverage \
  --input data/labelled/requirements.jsonl \
  --output data/labelled/coverage_report.json
```

The default readiness targets are 20 examples per label, 5 examples per role
family in the role catalog, 30 unique jobs, 3 sources, and no label exceeding
50% of the corpus. These thresholds can be overridden through CLI flags, but
lowering them should be an explicit experiment decision.

Run:

```bash
python -m scripts.build_dataset \
  --input data/labelled/requirements.jsonl \
  --output-dir data/datasets/requirements \
  --noise-ratio 0.30 \
  --evaluation-ratio 0.20
```

The training file mixes original real-world descriptions with deterministic
noisy variants. Evaluation is emitted in three paired forms:

- `evaluation/clean.jsonl`: only the annotated evidence context.
- `evaluation/real_world.jsonl`: the untouched full job description.
- `evaluation/noisy.jsonl`: the full description plus controlled recruiter,
  salary, formatting, marketing, heading, or unrelated-technology clutter.
- `evaluation/synthetic.jsonl`: held-out synthetic counterfactuals, reported
  separately and excluded from the primary real-world evaluation files.

Noise is always recorded in `noise_types`, and the evidence text remains
verbatim so augmentation cannot change the label semantics.
