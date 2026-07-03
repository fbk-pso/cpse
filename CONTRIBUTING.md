# Contributing to CPSE

Thanks for your interest in improving CPSE! This guide covers both community
processes and the practical developer workflow.

- [Code of Conduct](#code-of-conduct)
- [Asking questions](#asking-questions)
- [Reporting bugs](#reporting-bugs)
- [Suggesting enhancements](#suggesting-enhancements)
- [Contribution workflow](#contribution-workflow)
- [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- Developer guide: [Setup](#developer-guide--setup) ·
  [Common tasks](#common-tasks-via-just) · [Test suite](#running-the-test-suite) ·
  [Development wheels](#development-wheels)
- For maintainers: [Cutting a release](#cutting-a-release) ·
  [Dependencies](#keeping-dependencies-fresh)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold it. Report unacceptable behaviour
to <pso-tools@fbk.eu>; reports are handled confidentially.

## Asking questions

For usage questions, in order:

1. Check the [README](README.md).
2. Search the open and closed [issues](https://github.com/fbk-pso/cpse/issues).
3. If nothing matches, open an issue with the `question` label, including your
   CPSE version, Python version, and OS.

## Reporting bugs

Before filing, confirm you are on the latest version and search for duplicates.
A useful bug report includes:

- a **minimal reproducible example**: the scheduling problem as a Unified
  Planning Python snippet, the solver parameters, expected vs. actual output;
- the full stack trace, if any.
- version information: `pip show cpse unified-planning ortools`,
  `python --version`, and your OS.

Reports without a reproduction get labelled `needs-repro` and won't be
actively worked on until reproducible.

## Suggesting enhancements

For new features, search the issues for prior discussion and then open
one yourself, **before writing code**. Describe:

- The use case and *why it benefits the broader project*, not only your
  immediate need.
- A sketch of the proposed API or behaviour.
- Any alternatives you considered.

This lets us catch scope / design issues early and saves you from
writing a PR that needs to be redone.

## Contribution workflow

For non-trivial changes:

1. Open an issue first (or comment on an existing one) to align on
   scope and approach.
2. Fork the repo and create a feature branch off `main`:
   `git checkout -b feat/short-description`.
3. Make focused, well-described commits. Behaviour changes should come
   with new or updated tests.
4. Run `just precommit` locally — it must pass (this is exactly what CI's lint
   job runs).
5. Run `just test` and fix any failures locally first.
6. Push to your fork and open a pull request against `main`. Give the PR
   a clear, user-facing title: release notes are auto-generated from PR
   titles, so your title becomes a release-note line verbatim.
7. On your first PR the [CLA assistant](https://cla-assistant.io/) bot
   will ask you to sign the CLA — see the next section.
8. Address review comments.

Typo fixes and minor documentation changes can skip the first step.

## Contributor License Agreement (CLA)

Before your first contribution can be merged, you must sign the
[**FBK PSO Unit Individual Contributor License Agreement**](https://gist.github.com/alvalentini/a8c5e371be4e7e43b79035c67dc2a1ac).

CPSE is released under [GPL-3.0](LICENSE); the CLA
defines the licence terms under which you grant FBK the right to use
your contributions across all `fbk-pso` open-source projects.
On your first PR, the [cla-assistant](https://cla-assistant.io/) bot
posts a comment with a sign-in link; you authenticate with GitHub
OAuth and click "I agree". The signature applies to every subsequent
contribution you make to any project under the
[fbk-pso](https://github.com/fbk-pso) organisation.

**Exemptions:** FBK PSO Unit staff (whose contributions are governed
by their employment contracts) and automated accounts (bots) are
whitelisted in the cla-assistant configuration and skip the prompt.

## Developer guide

### Setup

Prerequisites:

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) — Python project / environment manager
- [just](https://github.com/casey/just) — command runner (install with
  `uv tool install rust-just` or your package manager)

Clone and bootstrap:

```bash
git clone https://github.com/fbk-pso/cpse.git
cd cpse
uv sync --all-extras       # creates .venv and installs dev deps
uv run pre-commit install  # run the checks automatically on each commit
```

The repo uses `uv` for environment/lock management, `ruff` for linting and
formatting, `mypy` for type checking, `pre-commit` hooks, and `just` as the task
runner.

### Common tasks

| Command | What it does |
| --- | --- |
| `just install` | `uv sync` — create `.venv` from `uv.lock` |
| `just test` | run the pytest suite |
| `just lint` | ruff lint + format checks |
| `just format` | auto-fix lint issues and reformat via ruff |
| `just typecheck` | mypy |
| `just precommit` | all pre-commit hooks — matches CI's lint job exactly |
| `just build` | build sdist + wheel into `./dist/` |
| `just clean` | remove build/dist/cache directories |

Run `just --list` to see them all.

### Running the test suite

`just test` runs the local pytest suite, which is self-contained.

To reproduce the full **Unified Planning engine report** that CI runs, clone the
UP repository for its `up_test_cases/` package (not shipped on PyPI), register
both engines, and run `report.py`:

```bash
sha=$(python3 -c "import tomllib; d = tomllib.load(open('uv.lock', 'rb')); print(next(p['source']['git'].rsplit('#', 1)[1] for p in d['package'] if p['name'] == 'unified-planning'))")
git clone --filter=blob:none https://github.com/aiplan4eu/unified-planning.git up-checkout
git -C up-checkout checkout "$sha"
printf '[engine cpse]\nmodule_name: cpse\nclass_name: CPSE\n\n[engine cpse-timepoints]\nmodule_name: cpse\nclass_name: CPSETimepoints\n' > .up.ini
uv run python up-checkout/up_test_cases/report.py cpse
uv run python up-checkout/up_test_cases/report.py cpse-timepoints
```

See [.github/workflows/test.yml](.github/workflows/test.yml) for the exact steps.

### Development wheels

CI builds a wheel on every push to `main` and publishes it as a rolling
pre-release tagged `dev` under Releases. These use PEP 440 dev versions like
`0.1.0.dev42+g<sha>`. Install one locally:

```bash
pip install --pre <url-of-wheel-on-dev-release>
```

## For maintainers

The rest of this file covers operations that need push or merge rights on the repo.

### Cutting a release

1. Bump `version` in [pyproject.toml](pyproject.toml).
2. Refresh the lock file:
   ```bash
   uv lock
   ```
3. Commit, tag, and push:
   ```bash
   git commit -am "release: v0.2.0"
   git tag v0.2.0 && git push --follow-tags
   ```

Pushing a `v*` tag triggers `build-and-release.yml`, which publishes the sdist +
wheel to PyPI via Trusted Publishing (OIDC) and creates a GitHub Release with
auto-generated notes and the built artifacts. The `dev-release` and
`github-release` jobs authenticate with the built-in `GITHUB_TOKEN`
(`contents: write`) — no extra secrets required.

### Keeping dependencies fresh

[Dependabot](.github/dependabot.yml) opens weekly pull requests for Python
dependencies (`uv`) and pinned GitHub Actions versions. Review the diff, let CI
run, and merge if green.

## Deeper reference

[CLAUDE.md](CLAUDE.md) documents the architecture, tooling conventions, and
design rationale. It is written primarily for Claude Code agents but is useful
for any contributor.
