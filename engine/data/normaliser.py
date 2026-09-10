"""
engine/data/normaliser.py
-------------------------
Converts raw broker tick dictionaries into typed ``Tick`` dataclasses.

Both the mock feed and the Kite WebSocket feed emit raw dicts; every adapter
feeds through this normaliser so the rest of the engine only ever sees
canonical ``Tick`` objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from engine.core.events import Tick


class TickNormaliser:
    """
    Normalises raw tick dicts into ``Tick`` dataclasses.

    Raw dict keys vary by adapter.  The normaliser supports two schemas:

    Mock/Kite-like schema::

        {
            "symbol":     "CRUDEOIL",
            "last_price": 6012.0,
            "bid":        6011.0,
            "ask":        6013.0,
            "volume":     12345,
            "oi":         45000,
            "timestamp":  "2024-01-15T09:15:00+05:30"  # or epoch float
        }

    Minimal schema (synthetic feed)::

        {
            "symbol":    "CRUDEOIL",
            "ltp":       6012.0,
            "timestamp": 1705289700.0   # epoch seconds
        }
    """

    def normalise(self, raw: dict) -> Tick:
        symbol = raw["symbol"]

        # --- price ---
        ltp_raw = raw.get("last_price") or raw.get("ltp") or raw.get("close")
        if ltp_raw is None:
            raise ValueError(f"Cannot determine LTP from tick: {raw!r}")
        ltp = Decimal(str(ltp_raw))

        bid = Decimal(str(raw.get("bid", ltp_raw)))
        ask = Decimal(str(raw.get("ask", ltp_raw)))

        # --- volume / OI ---
        volume = int(raw.get("volume", 0))
        oi     = int(raw.get("oi", 0))

        # --- timestamp ---
        ts = self._parse_timestamp(raw.get("timestamp"))

        return Tick(
            symbol=symbol,
            ltp=ltp,
            bid=bid,
            ask=ask,
            volume=volume,
            oi=oi,
            ts=ts,
        )

    @staticmethod
    def _parse_timestamp(raw_ts) -> datetime:
        if raw_ts is None:
            return datetime.now(tz=timezone.utc)
        if isinstance(raw_ts, datetime):
            return raw_ts.astimezone(timezone.utc) if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        if isinstance(raw_ts, (int, float)):
            return datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        if isinstance(raw_ts, str):
            # ISO 8601 (with or without tz)
            from dateutil.parser import parse
            dt = parse(raw_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        raise ValueError(f"Unsupported timestamp type: {type(raw_ts)} ({raw_ts!r})")
