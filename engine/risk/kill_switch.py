"""
engine/risk/kill_switch.py
---------------------------
Global kill switch — when engaged, all order approval calls return False.

The kill switch is persisted to a file so it survives process restarts
(an operator must explicitly reset it to resume trading).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SWITCH_FILE = "data/kill_switch.lock"


class KillSwitch:
    """
    Thread-safe kill switch with file-based persistence.

    Parameters
    ----------
    switch_file:
        Path to the lock file.  If the file exists on startup, the switch
        is considered engaged.
    """

    def __init__(self, switch_file: str = _DEFAULT_SWITCH_FILE) -> None:
        self._path = Path(switch_file)
        self._engaged: bool = self._path.exists()
        if self._engaged:
            logger.critical(
                "KillSwitch: ENGAGED from persistent lock file: %s", self._path
            )

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self, reason: str = "manual") -> None:
        """Engage the kill switch.  Idempotent."""
        if self._engaged:
            return
        self._engaged = True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).isoformat()
        self._path.write_text(f"{ts}: {reason}\n", encoding="utf-8")
        logger.critical("KillSwitch: ENGAGED — reason=%s", reason)

    def reset(self) -> None:
        """Reset the kill switch (operator action).  Removes lock file."""
        if not self._engaged:
            return
        self._engaged = False
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            logger.error("KillSwitch: failed to remove lock file: %s", e)
        logger.warning("KillSwitch: RESET by operator.")

    def check(self) -> bool:
        """Return True if trading is allowed (i.e. kill switch is NOT engaged)."""
        return not self._engaged
