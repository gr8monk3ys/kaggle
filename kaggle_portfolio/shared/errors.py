#!/usr/bin/env python3
"""Errors that cross the command boundary.

Commands used to signal validation failures by raising ``SystemExit`` from deep
inside ``main()``. That worked while every command ran in its own interpreter;
once they share one, a ``SystemExit`` from step 2 of preflight terminates the
dispatcher instead of failing that step.

``CommandError`` is the replacement: raise it for a user-facing failure, and the
dispatcher prints it and turns it into an exit code. ``SystemExit`` stays where it
belongs — the ``__main__`` guard.
"""

from __future__ import annotations


class CommandError(Exception):
    """A command failed in a way the user should be told about.

    Carries the exit code the CLI should return, so a command can distinguish
    "invalid usage" from "the thing you asked about is broken".
    """

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code
