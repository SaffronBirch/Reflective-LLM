"""
Runtime-log capture shared by the pipeline's per-turn saver and its batch
runner, so both produce identically-formatted transcripts.

Redirects the process-wide ``sys.stdout`` through a ``TeeLogger`` and attaches
a stdout handler to the ``reflective`` logger for the duration of a block, then
restores both on exit. ``capture_logging`` yields the TeeLogger — call
``.getvalue()`` for the full transcript.

This module depends only on the standard library, so importing it never pulls
in the framework, a provider, or torch. That is the point of the split: the
pipeline can reach its capture helpers without importing the experiment runner,
which is what let the batch runner become a thin shim over the pipeline.

NOTE: this redirects the process-wide ``sys.stdout`` and mutates the shared
``reflective`` logger, so it is not re-entrant or thread-safe — it is for a
single foreground run, not concurrent use.
"""

###################### Imports ######################
import logging
import sys
from contextlib import contextmanager
from typing import List


###################### Capture ######################
class TeeLogger:
    """
    Duplicates everything written to a stream (normally sys.stdout) into an
    in-memory buffer, so the full runtime reasoning log — every print emitted
    while a run proceeds — can be saved with the results.
    """

    def __init__(self, stream):
        self.stream = stream
        self._buffer: List[str] = []

    def write(self, text: str) -> None:
        self.stream.write(text)
        self._buffer.append(text)

    def flush(self) -> None:
        self.stream.flush()

    def getvalue(self) -> str:
        return "".join(self._buffer)


class _StdoutLogHandler(logging.Handler):
    """
    Writes log records to whatever sys.stdout is AT EMIT TIME — i.e. the
    TeeLogger while a capture block is active — so the package's logging output
    lands in the console, the JSON runtime_log, and the .log sidecar, just like
    the prints it accompanies.
    """

    def emit(self, record: logging.LogRecord) -> None:
        sys.stdout.write(self.format(record) + "\n")


@contextmanager
def capture_logging(debug: bool):
    """
    Capture everything printed and logged during the block.

    Redirects sys.stdout through a TeeLogger and attaches a stdout handler to
    the 'reflective' logger for the duration, then restores both on exit.
    Yields the TeeLogger — call ``.getvalue()`` for the full transcript.

    Shared by the pipeline's per-turn saver and its batch runner so both produce
    identical runtime logs. NOTE: this redirects the process-wide sys.stdout and
    mutates the shared 'reflective' logger, so it is not re-entrant or
    thread-safe — it is for a single foreground run, not concurrent use.
    """
    original_stdout = sys.stdout
    tee = TeeLogger(original_stdout)
    sys.stdout = tee

    # The package is quiet by default (NullHandler); a capture block is the
    # verbose context, so attach a handler for its duration. DEBUG-level records
    # (the framework's debug dumps) are gated by the debug flag via the level.
    handler = _StdoutLogHandler(level=logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_logger = logging.getLogger("reflective")
    prior_level = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG)
    try:
        yield tee
    finally:
        sys.stdout = original_stdout
        package_logger.removeHandler(handler)
        package_logger.setLevel(prior_level)