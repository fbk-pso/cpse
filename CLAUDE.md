# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CPSE (Constraint Programming Scheduling Engine) is a pure-Python scheduling
solver. It encodes Unified Planning `SchedulingProblem`s as constraint models
and solves them with the CP-SAT solver from Google OR-Tools. It ships **two
engines**:

- **`cpse`** (`CPSE`) — supports optional activities and scoped constraints;
  partial fluent support (only increase/decrease effects on non-parametric
  fluents; no assignment effects).
- **`cpse-timepoints`** (`CPSETimepoints`) — models the timeline as discrete
  timepoints, giving full fluent support, but no optional activities and higher
  computational cost.

The package is consumed as a Unified Planning plugin, not run standalone.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for the environment and
[just](https://github.com/casey/just) as the task runner (`just --list`):

```bash
just install      # uv sync — create .venv from uv.lock
just test         # run the pytest suite
just lint         # ruff check + ruff format --check (src tests ci)
just format       # ruff format + ruff check --fix
just typecheck    # mypy
just precommit    # all pre-commit hooks (this is what CI's lint job runs)
just build        # build sdist + wheel into ./dist/
```

Run a single test:

```bash
uv run pytest tests/test_CPSETimepoints.py::TestCPSETimepoints::test_int_fluents -v
```

Reproduce the full Unified Planning engine report that CI runs (clone UP for its
`up_test_cases`, register the engines, run `report.py`):

```bash
git clone --branch master https://github.com/aiplan4eu/unified-planning.git up-checkout
printf '[engine cpse]\nmodule_name: cpse\nclass_name: CPSE\n\n[engine cpse-timepoints]\nmodule_name: cpse\nclass_name: CPSETimepoints\n' > .up.ini
uv run python up-checkout/up_test_cases/report.py cpse
PYTHONPATH=up-checkout/up_test_cases uv run pytest tests/ -v
```

## Architecture

`CPSEBaseEngine` (`src/cpse/CPSEBaseEngine.py`) is the heart of the project. It
subclasses `up.engines.Engine` + `OneshotPlannerMixin` and translates a UP
`SchedulingProblem` into a CP-SAT model in roughly this order: declare variables
(`new_bool_var` / `new_int_var`), `add_parameters`, `add_presence_expressions`,
`add_activity`, then `add_constraints` / `add_effects` / `add_conditions` /
`add_quality_metrics`. Solving runs CP-SAT and maps the solution back to a UP
`Schedule`. UP `FNode` expressions are walked and converted into CP-SAT
variables/linear expressions.

`CPSE` (`src/cpse/CPSE.py`) and `CPSETimepoints` (`src/cpse/CPSETimepoints.py`)
each subclass `CPSEBaseEngine` and override `name`, `supported_kind`,
`supports`, `check_if_supported_problem`, and the encoding methods
(`add_constraints` / `add_effects` / `add_conditions`). `CPSETimepoints` adds an
entire timepoint layer (`timepoints_setup`, `_add_activity_timepoints`,
parametric-fluent handling) — this is why it is the larger, costlier engine.

When changing shared encoding logic, remember it must keep working for **both**
engines; behavior that differs belongs in the subclass override, not the base.

### Engine registration

Engines are plugins discovered by Unified Planning, either via a `.up.ini`
file (`module_name: cpse`, `class_name: CPSE` / `CPSETimepoints`) or
programmatically: `env.factory.add_engine("cpse", "cpse", "CPSE")`. They are then
invoked through `OneshotPlanner(name="cpse")`. Configurable params: `lower_bound`
(default 0) and `upper_bound` (default INT32_MAX) bound all model variables.

### Tests

`tests/CommonTests.py` defines a `CommonTests` base class holding the entire
shared test suite. `tests/test_CPSE.py` and `tests/test_CPSETimepoints.py`
subclass `CommonTests` (overriding `engine_name` / `engine_class`) so the same
tests run against each engine. The shared `problem` pytest fixture lives in
`tests/conftest.py`, so pytest auto-discovers it for every test module — no
import needed in the test files.

## Conventions and gotchas

- **src/ layout:** code lives under `src/cpse/` but is imported as `cpse`.
- **unified-planning is a dev dependency,** not a runtime dep of the wheel
  (`[dependency-groups] dev` in `pyproject.toml`), and is pinned
  `>=1.3.0.445.dev1` — the `cpse` engine needs optional-activities support
  (`Presence`) that is not in a stable PyPI UP release.
- **ortools is pinned** to `>=9.15,<9.16`; supported Python is **3.10–3.14**.
- Formatting/linting is **ruff** (line length 88); type checking is **mypy**,
  configured in `pyproject.toml`. Four tests are `@pytest.mark.skip`-ed because
  CP-SAT is too slow on them with the pinned ortools.
- Releases: `just bump X.Y.Z` sets `version` in `pyproject.toml` and runs
  `uv lock`, then prints the commit/tag/push commands to run. A `vX.Y.Z` tag
  triggers PyPI publish via Trusted Publishing; `main` merges produce a rolling
  `dev` pre-release. See `.github/workflows/`.
