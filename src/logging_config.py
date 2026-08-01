"""Logging setup for the command line entry points.

Library modules never configure logging; they only ask for a logger and write to
it. Configuration happens once, in whichever entry point the user actually ran.
That keeps output under the caller's control: a notebook, a test, or another
program can import any module here without having its own logging reconfigured.
"""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
PLAIN_FORMAT = "%(message)s"


def configure(verbose: bool = False, quiet: bool = False) -> None:
    """Attach a single stream handler to the root logger.

    Progress tables and headline numbers are written without decoration, because
    they are meant to be read. Set ``DATURA_LOG_FORMAT=full`` to get timestamps and
    module names instead, which is what you want when a run is being archived.
    """
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    fmt = DEFAULT_FORMAT if os.environ.get("DATURA_LOG_FORMAT") == "full" else PLAIN_FORMAT

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # These are chatty at DEBUG and say nothing about this pipeline.
    for name in ("matplotlib", "PIL", "numba", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
