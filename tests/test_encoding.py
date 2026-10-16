"""
Unit tests for cosmo.encoding.configure_utf8_stdout().

We verify two properties:
  1. Idempotency — calling configure_utf8_stdout() multiple times does not raise.
  2. Graceful no-op on streams that lack reconfigure() (e.g. StringIO, mocked
     objects) — the function must not raise AttributeError or TypeError.
"""

import sys
import unittest
from io import StringIO

from cosmo.encoding import configure_utf8_stdout


class TestConfigureUtf8Stdout(unittest.TestCase):

    def test_idempotent_does_not_raise(self):
        """Calling the helper twice in a row must not raise."""
        configure_utf8_stdout()
        configure_utf8_stdout()  # second call — must be a no-op

    def test_no_raise_on_stream_without_reconfigure(self):
        """Streams that lack reconfigure() (e.g. StringIO) must be silently skipped."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            # StringIO has no reconfigure() — the function should do nothing
            configure_utf8_stdout()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def test_no_raise_on_object_with_raising_reconfigure(self):
        """Streams whose reconfigure() raises ValueError or OSError must be silently skipped."""

        class _BadStream:
            def reconfigure(self, **_kwargs):
                raise OSError("simulated error")

        old_stdout = sys.stdout
        try:
            sys.stdout = _BadStream()  # type: ignore[assignment]
            configure_utf8_stdout()
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
