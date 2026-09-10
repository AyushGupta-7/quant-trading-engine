"""
engine/oms/state_store.py
--------------------------
SQLite-backed persistent order state store with WAL mode.

All order state changes are written synchronously before the broker call
so that a crash at any point can be recovered by re-reading the store.

Schema::

    orders(
        client_order_id TEXT PRIMARY KEY,
        broker_order_id TEXT,
        symbol TEXT,
        side TEXT,
        qty INTEGER,
        filled_qty INTEGER,
        order_type TEXT,
        price TEXT,         -- Decimal serialised as string
        avg_fill_price TEXT,
        state TEXT,
        strategy_id TEXT,
        tag TEXT,
        created_at TEXT,
        updated_at TEXT
    )
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from engine.core.events import Order, OrderState, OrderType, Side

logger = logging.getLogger(__name__)


class OrderStateStore:
    """
    Synchronous SQLite order store (WAL mode).

    For the asyncio path use ``AsyncOrderStateStore`` (wraps this in a
    thread-pool executor).  For tests and the backtest engine the sync
    version is sufficient.
    """

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self._db_path = db_path
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                client_order_id TEXT PRIMARY KEY,
                broker_order_id TEXT,
                symbol          TEXT NOT NULL,
                side            TEXT NOT NULL,
                qty             INTEGER NOT NULL,
                filled_qty      INTEGER DEFAULT 0,
                order_type      TEXT NOT NULL,
                price           TEXT,
                avg_fill_price  TEXT DEFAULT '0',
                state           TEXT NOT NULL,
                strategy_id     TEXT NOT NULL,
                tag             TEXT DEFAULT '',
                created_at      TEXT,
                updated_at      TEXT
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, order: Order) -> None:
        self._conn.execute("""
            INSERT INTO orders (
                client_order_id, broker_order_id, symbol, side, qty, filled_qty,
                order_type, price, avg_fill_price, state, strategy_id, tag,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(client_order_id) DO UPDATE SET
                broker_order_id = excluded.broker_order_id,
                filled_qty      = excluded.filled_qty,
                avg_fill_price  = excluded.avg_fill_price,
                state           = excluded.state,
                updated_at      = excluded.updated_at
        """, (
            order.client_order_id,
            order.broker_order_id,
            order.symbol,
            order.side.value,
            order.qty,
            order.filled_qty,
            order.order_type.value,
            str(order.price) if order.price is not None else None,
            str(order.avg_fill_price),
            order.state.value,
            order.strategy_id,
            order.tag,
            order.created_at.isoformat() if order.created_at else None,
            order.updated_at.isoformat() if order.updated_at else None,
        ))
        self._conn.commit()

    def get(self, client_order_id: str) -> Order | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return self._row_to_order(row) if row else None

    def get_by_state(self, state: OrderState) -> list[Order]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE state = ?", (state.value,)
        ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get_all(self) -> list[Order]:
        rows = self._conn.execute("SELECT * FROM orders").fetchall()
        return [self._row_to_order(r) for r in rows]

    def exists(self, client_order_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        return Order(
            client_order_id=row["client_order_id"],
            broker_order_id=row["broker_order_id"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            qty=row["qty"],
            filled_qty=row["filled_qty"],
            order_type=OrderType(row["order_type"]),
            price=Decimal(row["price"]) if row["price"] else None,
            avg_fill_price=Decimal(row["avg_fill_price"]),
            state=OrderState(row["state"]),
            strategy_id=row["strategy_id"],
            tag=row["tag"] or "",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def close(self) -> None:
        self._conn.close()
