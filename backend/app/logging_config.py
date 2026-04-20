"""Central logging config.

Controlled by the LOG_LEVEL env var (default INFO). Call `configure()` once at
app startup to set a format and a single stdout handler on the root logger.

We use `force=True` so reload and re-import don't leave duplicate handlers
stacked up when uvicorn re-executes the app module.
"""

from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s :: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format=_FMT,
        datefmt=_DATEFMT,
        stream=sys.stdout,
        force=True,
    )
    # Quiet noisy libs unless the user explicitly wants DEBUG everywhere.
    if level > logging.DEBUG:
        logging.getLogger("multipart").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
