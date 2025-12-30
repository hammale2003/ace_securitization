"""
Local stub for the `readline` module (Windows/Anaconda pytest workaround).

See repository root `readline.py` for details.
This duplicate file exists so running tests from inside `ace_securitization/`
also shadows the broken third-party `readline` package.
"""

from __future__ import annotations

from typing import Optional


def parse_and_bind(_s: str) -> None:
    return None


def read_init_file(_path: Optional[str] = None) -> None:
    return None


def get_history_length() -> int:
    return 0


def clear_history() -> None:
    return None


def add_history(_line: str) -> None:
    return None


