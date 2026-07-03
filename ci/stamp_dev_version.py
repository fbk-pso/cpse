"""Stamp a derived dev version into pyproject.toml.

Run only in CI on main-branch builds. The patched file is consumed by the
build job but never committed.

Python (PEP 440):   <base>.dev<N>+g<sha>      e.g. 0.1.0.dev42+gabc1234

<N> is the total commit count on HEAD (monotonic, doesn't depend on tag fetch).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import tomllib


def sh(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    base = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"][
        "version"
    ]
    if not re.match(r"^\d+\.\d+\.\d+$", base):
        sys.exit(f"refusing to stamp on top of an already-suffixed version: {base!r}")

    n = sh("git", "rev-list", "--count", "HEAD")
    sha = sh("git", "rev-parse", "--short=7", "HEAD")
    dev = f"{base}.dev{n}+g{sha}"

    py = pathlib.Path("pyproject.toml")
    py.write_text(
        re.sub(
            r'^version = ".*"',
            f'version = "{dev}"',
            py.read_text(),
            count=1,
            flags=re.M,
        )
    )
    print(f"stamped version = {dev}")


if __name__ == "__main__":
    main()
