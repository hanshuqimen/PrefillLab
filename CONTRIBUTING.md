# Contributing to PrefillLab

Bring reproducible evidence, clear timing boundaries, and small reviewable changes. Report a problem or propose an experiment before adding a large backend or profiler.

## Setup

Use Python 3.11+ in a virtual environment. Install `pip install -e '.[dev]'`. Run `pytest`, `ruff check .`, `ruff format --check .`, and `mypy prefilllab`. Optional real CPU tests use a tiny local random model: install `.[transformers]` and run `pytest -m integration`. GPU tests must have the `gpu` marker and remain outside default CI.

For frontend work use Node 22.12+ and `cd frontend && npm install && npm run build`. CI uses the committed pnpm lockfile and pnpm 11. `uvicorn prefilllab.api.app:app --reload` and `npm run dev` start the development servers. Rebuild `prefilllab/web` when changing the frontend; the dashboard is included in Python packages.

## Design contract

- Preserve `simulated` provenance and null measurements.
- Keep profiler overhead outside benchmark samples.
- Preserve exclusive versus inclusive timing semantics.
- Add backend factories through `register_backend`; document capabilities and optional packages.
- Add heuristic rules with evidence and confidence, not unproven hardware diagnoses.
- Keep API responses Pydantic-validated and frontend fetch calls in the shared client.
- Add tests for actual behavior, malformed input, error cleanup, and reproducibility where relevant.
- Avoid model downloads or GPU requirements in the default test suite.

Use descriptive commits and explain user-visible behavior, measurement semantics, and validation in your pull request. Apache-2.0 contributions are welcome; submit only material you have the right to contribute.
