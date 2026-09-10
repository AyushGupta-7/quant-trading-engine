"""
engine/observability/blotter.py
--------------------------------
Trade blotter — records every fill to CSV and SQLite.

The blotter is the canonical audit trail for all executed trades.
It is append-only and written before any downstream processing.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from engine.core.events import Fill

logger = logging.getLogger(__name__)


class TradeBlotter:
    """
    Dual-writer: appends fills to a CSV file and an SQLite ``trades`` table.

    Parameters
    ----------
    csv_path:
        Path to the CSV blotter file.  Use ":memory:" to disable CSV writing.
    db_path:
        Path to the SQLite database.  Use ":memory:" for in-memory (tests).
    """

    _COLUMNS = [
        "ts", "order_id", "broker_order_id", "symbol",
        "side", "fill_price", "fill_qty", "fees",
    ]

    def __init__(self, csv_path: str = "data/blotter.csv", db_path: str = ":memory:") -> None:
        self._csv_path = csv_path
        self._db_path  = db_path
        self._csv_file = None
        self._csv_writer = None
        self._conn: sqlite3.Connection | None = None
        self._init()

    def _init(self) -> None:
        # CSV
        if self._csv_path != ":memory:":
            Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)
            needs_header = not Path(self._csv_path).exists()
            self._csv_file = open(self._csv_path, "a", newline="", buffering=1)
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._COLUMNS)
            if needs_header:
                self._csv_writer.writeheader()

        # SQLite
        if self._db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT NOT NULL,
                order_id        TEXT NOT NULL,
                broker_order_id TEXT,
                symbol          TEXT NOT NULL,
                side            TEXT NOT NULL,
                fill_price      REAL NOT NULL,
                fill_qty        INTEGER NOT NULL,
                fees            REAL NOT NULL
            )
        """)
        self._conn.commit()

    def record(self, fill: Fill) -> None:
        """Record a fill to CSV and SQLite."""
        row = {
            "ts":             fill.ts.isoformat(),
            "order_id":       fill.order_id,
            "broker_order_id":fill.broker_order_id,
            "symbol":         fill.symbol,
            "side":           fill.side.value,
            "fill_price":     float(fill.fill_price),
            "fill_qty":       fill.fill_qty,
            "fees":           float(fill.fees),
        }
        if self._csv_writer:
            self._csv_writer.writerow(row)

        self._conn.execute("""
            INSERT INTO trades
            (ts, order_id, broker_order_id, symbol, side, fill_price, fill_qty, fees)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            row["ts"], row["order_id"], row["broker_order_id"], row["symbol"],
            row["side"], row["fill_price"], row["fill_qty"], row["fees"],
        ))
        self._conn.commit()

        logger.debug(
            "Blotter: recorded fill sym=%s side=%s qty=%d @%.2f fees=%.2f",
            fill.symbol, fill.side.value, fill.fill_qty, fill.fill_price, fill.fees,
        )

    def get_all_fills(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM trades ORDER BY ts").fetchall()
        return [dict(r) for r in rows]

    def total_fees(self) -> float:
        result = self._conn.execute("SELECT SUM(fees) FROM trades").fetchone()
        return result[0] or 0.0

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        if self._conn:
            self._conn.close()
