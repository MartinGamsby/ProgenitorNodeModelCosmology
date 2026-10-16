"""
UTF-8 stdout/stderr reconfiguration helper.

Calling configure_utf8_stdout() at process start reconfigures sys.stdout and
sys.stderr to UTF-8 with errors="replace" on Windows cp1252 consoles.  It is
safe to call multiple times (idempotent) and safe when the stream lacks the
reconfigure() method (e.g. StringIO in tests).
"""

import sys


def configure_utf8_stdout() -> None:
    """Reconfigure stdout and stderr to UTF-8 (errors='replace').

    Must be called BEFORE any print() that may emit Greek or non-ASCII
    characters (Ω, Λ, χ², ★, etc.).  Has no effect when already running
    under a UTF-8 locale or when PYTHONIOENCODING has been set externally.
    """
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            try:
                _reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
