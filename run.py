#!/usr/bin/env python3
"""Make-free task runner — the same commands as the Makefile, but with nothing
but a Python interpreter required (no `make`, no `uv`).

    python run.py setup        # create .venv + install the app runtime
    python run.py ui           # launch the Gradio order desk
    python run.py setup-dev    # also install the test/lint tooling
    python run.py test
    python run.py lint
    python run.py eval

It installs from the pinned requirements files (generated from uv.lock) into a
local `.venv/`, then runs every command with that interpreter — so a fresh
checkout is two commands away from a running app:

    python run.py setup
    python run.py ui

If you already have a virtualenv (or conda env) activated, it installs into that
instead of creating `.venv/`. uv users can keep using the Makefile.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _in_active_venv() -> bool:
    """True when the caller already has a virtual/conda env activated."""
    return sys.prefix != sys.base_prefix or "CONDA_PREFIX" in os.environ


def _venv_python() -> Path:
    """The python.exe inside .venv (Windows) or .venv/bin/python (POSIX)."""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _ensure_venv() -> str:
    """Return the interpreter to use, creating .venv on first `setup` if needed."""
    if _in_active_venv():
        return sys.executable
    if not _venv_python().exists():
        print(f"Creating virtual environment in {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    return str(_venv_python())


def _interpreter_for_run() -> str:
    """The interpreter for `ui`/`test`/etc. — fail clearly if setup hasn't run."""
    if _in_active_venv():
        return sys.executable
    if _venv_python().exists():
        return str(_venv_python())
    sys.exit("No environment found. Run `python run.py setup` first.")


def _run(args: list[str]) -> int:
    print("+", " ".join(args))
    return subprocess.call(args, cwd=ROOT)


def _pip_install(requirements: str) -> int:
    py = _ensure_venv()
    rc = _run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if rc:
        return rc
    return _run([py, "-m", "pip", "install", "-r", str(ROOT / requirements)])


# command name -> (help text, callable returning an exit code)
def _setup() -> int:
    return _pip_install("requirements.txt")


def _setup_dev() -> int:
    return _pip_install("requirements-dev.txt")


def _ui() -> int:
    return _run([_interpreter_for_run(), "-m", "src.interfaces.gradio_app"])


def _test() -> int:
    return _run([_interpreter_for_run(), "-m", "pytest", *sys.argv[2:]])


def _lint() -> int:
    return _run([_interpreter_for_run(), "-m", "ruff", "check", "."])


def _eval() -> int:
    return _run([_interpreter_for_run(), "-m", "evals.run_eval"])


def _eval_langsmith() -> int:
    return _run([_interpreter_for_run(), "-m", "evals.langsmith_eval"])


COMMANDS = {
    "setup": ("create .venv + install the app runtime (requirements.txt)", _setup),
    "setup-dev": ("install the runtime + test/lint tooling (requirements-dev.txt)", _setup_dev),
    "ui": ("launch the Gradio order desk (needs GOOGLE_API_KEY)", _ui),
    "test": ("run the test suite (extra args pass through to pytest)", _test),
    "lint": ("ruff check .", _lint),
    "eval": ("score the dataset locally (needs GOOGLE_API_KEY + OPENAI_API_KEY)", _eval),
    "eval-langsmith": ("run the LangSmith-native eval", _eval_langsmith),
}


def _usage() -> None:
    print("Usage: python run.py <command>\n\nCommands:")
    width = max(len(name) for name in COMMANDS)
    for name, (help_text, _) in COMMANDS.items():
        print(f"  {name.ljust(width)}  {help_text}")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        _usage()
        return 0
    command = sys.argv[1]
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"Unknown command: {command}\n")
        _usage()
        return 2
    return entry[1]()


if __name__ == "__main__":
    raise SystemExit(main())