# CPSE

**CPSE** (Constraint Programming Scheduling Engine) is a scheduling engine that encodes scheduling problems as constraint satisfaction models and solves them using the CP-SAT solver from [Google OR-Tools](https://developers.google.com/optimization).

CPSE offers two scheduling engines, each supporting different problem kinds:

- **cpse**:
  Supports scheduling problems with **optional activities** and **scoped constraints**.
  Partial support for **fluents** is provided: only increase and decrease effects on non-parametric fluents are handled, while assignment effects are **not supported**.

- **cpse-timepoints**:
  Supports scheduling problems without optional activities.
  Offers **full support for fluents**, but is generally more computationally intensive.


## Installation

```bash
pip install up-cpse
```

To try the latest unreleased build, install a wheel directly from the rolling
[`dev` pre-release](https://github.com/fbk-pso/cpse/releases/tag/dev):

```bash
pip install --pre <url-of-wheel-on-dev-release>
```


## Usage

CPSE is fully integrated with the [Unified Planning](https://github.com/aiplan4eu/unified-planning) framework. Before using CPSE, register its engines in the Unified Planning environment:

```python
from unified_planning.shortcuts import *

# Register CPSE engines
env = get_environment()
env.factory.add_engine("cpse", "cpse", "CPSE")
env.factory.add_engine("cpse-timepoints", "cpse", "CPSETimepoints")

# Define your scheduling problem
scheduling_problem = ...

# Solve the problem using the cpse engine
with OneshotPlanner(name="cpse") as planner:
    result = planner.solve(scheduling_problem)
    print(result.plan)
```


## Parameters

The CPSE engines support the following configuration parameters:

| Parameter     | Type | Default Value | Description |
| ------------- | ---- | ------------- | ----------- |
| `lower_bound` | int  | `0`           | Minimum value for all model variables if not explicitly specified in the problem. |
| `upper_bound` | int  | `INT32_MAX`   | Maximum value for all model variables if not explicitly specified in the problem (`INT32_MAX = 2^31 - 1`). |

These parameters can be passed as a dictionary to `OneshotPlanner`:

```python
params = {
  "lower_bound": 1,
  "upper_bound": 100
}

with OneshotPlanner(name="cpse", params=params) as planner:
    result = planner.solve(scheduling_problem)
    print(result.plan)
```

**Tip**: Adjusting these bounds can help restrict variable domains or improve solver performance for specific scheduling problems.


## Development

CPSE uses [uv](https://docs.astral.sh/uv/) to manage the environment and
[just](https://github.com/casey/just) as a task runner. After cloning:

```bash
just install        # uv sync — create .venv from uv.lock
```

Common tasks:

```bash
just test           # run the pytest suite
just lint           # ruff lint + format checks
just format         # auto-fix lint issues and format
just typecheck      # mypy
just precommit      # run all pre-commit hooks against the whole repo
just build          # build sdist + wheel into ./dist/
```

Install the git hook so the checks run automatically on each commit:

```bash
uv run pre-commit install
```

Running `just --list` shows all available recipes.


## References

CPSE has been used in the following research paper:

- Elisa Tosello, Arthur Bit-Monnot, Davide Lusuardi, Alessandro Valentini and Andrea Micheli (2026). *Interleaving Scheduling and Motion Planning with Incremental Learning of Symbolic Space-Time Motion Abstractions.* **ICAPS 2026**


## License

CPSE is released under the GNU General Public License v3.0 (GPL-3.0).
See the `LICENSE` file for full details.


## Contact

For questions, bug reports, or contributions, please open an issue on GitHub or contact the authors at <pso-tools@fbk.eu>.
