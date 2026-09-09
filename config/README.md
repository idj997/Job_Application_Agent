# Job corpus role catalog

`job_role_queries.json` defines the role families used to build a diverse job
description corpus. Edit the JSON to add or refine families without changing
the collector code.

Inspect the catalog and planned API work before collecting:

```bash
python -m scripts.collect_role_corpus --list-roles

python -m scripts.collect_role_corpus \
  --sources adzuna \
  --roles data_engineering data_science ai_engineering machine_learning \
          generative_ai cpu_engineering gpu_engineering \
  --location "United Kingdom" \
  --queries-per-role 1 \
  --limit-per-query 5 \
  --dry-run
```

Remove `--dry-run` to collect. Start with one query per role, inspect the role
and source distributions, and only then increase `--queries-per-role`. The
collector uses shared storage and cross-query deduplication. If one source has
a discovery-level failure, the remaining sources continue, and repeated calls
to the failed source are suppressed for that run.

Employer ATS sources still need their employer identifier (`--board-token` for
Greenhouse or `--company-slug` for Lever). Search APIs such as Adzuna are a
better first choice for broad occupational coverage.
