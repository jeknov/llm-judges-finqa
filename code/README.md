# fqsel — implication-weighted data selection for financial QA

Code for protocol v1 (training-free). The specs in `../specs/` are authoritative; progress, owner
decisions and verification results are tracked in `../specs/IMPLEMENTATION_PLAN.md`.

## Setup

```bash
pip install --user uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"       # add ".[gpu]" on the GPU node
```

Raw data lives in `data/raw/` (tracked by DVC; hashes in `data/raw/MANIFEST`).

## Running

```bash
.venv/bin/python -m fqsel run <stage> [hydra overrides]   # stages: pool verify q3 jev surprisal ...
.venv/bin/python -m fqsel check                          # re-validate all outputs and invariants
dvc repro                                                # every stage, profile from params.yaml
```

Real runs need two API keys in the environment: `TYPESAFE_API_KEY` (Jev) and `OPENROUTER_API_KEY`
(`model.strong`, Qwen3-235B-A22B-Instruct-2507 pinned to the `deepinfra/fp8` endpoint).

Profiles: `profile=tiny` (default; toy model, mock Jev and strong LLM, Hugging Face backend),
`profile=pilot` and `profile=full` (vLLM, real clients). Pilot and full refuse to start while any
owner decision is pending (`???` in `conf/`, or a prompt in `prompts/` not marked `FINAL`).

## Human labeling (CSV, never Argilla)

```bash
.venv/bin/python -m fqsel annotate export q3_validation   # writes annotation/q3_validation/<annotator>.csv
# annotators fill in the label column with yes/no and return the same file to
#   annotation/q3_validation/returned/<annotator>.csv
.venv/bin/python -m fqsel annotate import q3_validation   # strict validation, then labels.parquet
```

`q3_templates` works the same way for the hand-check of template checks.

## Tests

```bash
.venv/bin/python -m pytest -q                 # unit, schema, stats and the tiny end-to-end test
.venv/bin/python -m pytest -m real_data -s    # checks on the downloaded data (executor coverage, ...)
.venv/bin/python -m pytest -m gpu             # on the GPU node
```

`tests/ACCEPTANCE.md` maps every spec acceptance item to its tests.

## Archived results and API responses (DVC)

Four directories are tracked by DVC rather than git; git holds only their `.dvc` pointer files:

| Path | Contents |
| --- | --- |
| `runs/` | Frozen result snapshots, each with a `SHA256SUMS` file: `pilot_2026-10-24` (battery v0), `pilot_v1_2026-10-24`, `pilot_v2_2026-10-25`, `full_v0_2026-10-24` (sensitivity analysis) and `full_v2_2026-10-25` (primary). The analysis notebooks in `../notebooks/` read these. |
| `.cache/jev/` | Every Jev response, keyed by the SHA-256 of (model, state, questions[, repeat]). |
| `.cache/strong/` | Every Qwen (model.strong) response. |
| `.cache/logs/` | Request logs with request IDs, token counts and cost. |

`dvc pull` restores them on another machine once a DVC remote is configured (`dvc remote add -d <name> <url>`, then `dvc push`). After a new run, record the new state with `dvc add runs .cache/jev .cache/strong .cache/logs` and commit the updated `.dvc` files.
