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
