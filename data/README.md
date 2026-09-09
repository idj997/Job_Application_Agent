# Local data directory

The published repository contains code, configuration examples and synthetic
test fixtures. It does not include your CV, candidate details, collected job
descriptions, annotation queues or application history. Those stay in this
directory on your machine and are excluded by `.gitignore`.

| Path | Purpose | Created by |
| --- | --- | --- |
| `cv/` | Master CV and optional saved full job descriptions | You |
| `raw_jobs/` | Original job HTML/JSON | Existing collectors |
| `processed_jobs/` | Normalized jobs used by either workflow | Existing collectors or OpenAI `prepare --fetch` |
| `labelled/` | Annotation candidates, reviewed labels and review history | Annotation commands |
| `datasets/` | Training splits, augmented records and evaluation datasets | Dataset builder |
| `applications/` | Tailored CVs, evaluations, dashboard and SQLite application ledger | OpenAI workflow |

The OpenAI collection command stores its raw responses in
`applications/raw_jobs/`; the standalone collectors use `raw_jobs/`.

The empty legacy `candidate_profile.yaml` and `jobs.db` files are not used by
the OpenAI workflow. Its candidate profile is a JSON file, usually
`config/candidate_profile.json`, and its ledger is
`data/applications/applications.db`.

Back up your local data separately. In particular, retain the SQLite application
ledger so a new checkout does not lose the record of jobs you already applied to.
Custom `--output` locations outside `data/` need their own ignore rule if they are
inside this repository.

See [annotation formats and review rules](labelled/README.md), the
[OpenAI guide](../docs/OPENAI_MODE.md), and the
[local workflow guide](../docs/LOCAL_MODE.md).
