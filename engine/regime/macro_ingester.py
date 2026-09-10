"""
engine/regime/macro_ingester.py
--------------------------------
Ingests macro proxy data and exposes a current snapshot dict.

Macro proxies used for regime scoring:
  * India VIX        — measures equity market fear
  * USD/INR          — currency stress
  * FII Flow (crore) — foreign institutional buying/selling

In the mock implementation these values are either read from a simple CSV
time-series or hardcoded to the values in config.  The interface is designed
so a live implementation can replace the source (REST call, Redis pub/sub, etc.)
without changing the scorer or the rest of the engine.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MacroSnapshot:
    """Immutable snapshot of macro proxy values at a point in time."""

    __slots__ = ("india_vix", "usd_inr", "fii_flow_crore", "ts")

    def __init__(
        self,
        india_vix: float,
        usd_inr: float,
        fii_flow_crore: float,
        ts: datetime | None = None,
    ) -> None:
        self.india_vix = india_vix
        self.usd_inr   = usd_inr
        self.fii_flow_crore = fii_flow_crore
        self.ts = ts or datetime.now(tz=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "india_vix":      self.india_vix,
            "usd_inr":        self.usd_inr,
            "fii_flow_crore": self.fii_flow_crore,
            "ts":             self.ts.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"MacroSnapshot(vix={self.india_vix}, "
            f"usd_inr={self.usd_inr}, fii={self.fii_flow_crore})"
        )


class MockMacroIngester:
    """
    Returns macro proxy values from config (static mock) or a CSV file.

    CSV format (optional, for time-series replay)::

        timestamp,india_vix,usd_inr,fii_flow_crore
        2024-01-15,18.5,83.2,500
    """

    def __init__(self, proxies_cfg: dict) -> None:
        self._cfg = proxies_cfg
        self._csv_rows: list[MacroSnapshot] = []
        self._csv_index = 0
        self._load_csv()

    def _load_csv(self) -> None:
        path = self._cfg.get("csv_path")
        if not path or not Path(path).exists():
            return
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._csv_rows.append(
                    MacroSnapshot(
                        india_vix=float(row["india_vix"]),
                        usd_inr=float(row["usd_inr"]),
                        fii_flow_crore=float(row["fii_flow_crore"]),
                        ts=datetime.fromisoformat(row["timestamp"]).replace(
                            tzinfo=timezone.utc
                        ),
                    )
                )
        logger.info("MacroIngester: loaded %d rows from CSV", len(self._csv_rows))

    def get_snapshot(self, ts: datetime | None = None) -> MacroSnapshot:
        """Return the current (or next chronological) macro snapshot."""
        if self._csv_rows:
            idx = min(self._csv_index, len(self._csv_rows) - 1)
            snap = self._csv_rows[idx]
            if self._csv_index < len(self._csv_rows) - 1:
                self._csv_index += 1
            return snap

        # Static mock values from config
        return MacroSnapshot(
            india_vix=float(self._cfg.get("india_vix", {}).get("mock_value", 18.0)),
            usd_inr=float(self._cfg.get("usd_inr", {}).get("mock_value", 83.5)),
            fii_flow_crore=float(
                self._cfg.get("fii_flow_crore", {}).get("mock_value", 500.0)
            ),
            ts=ts or datetime.now(tz=timezone.utc),
        )

    def reset(self) -> None:
        self._csv_index = 0
