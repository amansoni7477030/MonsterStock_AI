"""One live table from two real-time feeds: NSE's website API and the Kotak broker.

Both carry the exchange's own figures (after the close all 218 F&O stocks matched on open, high, low, last, previous close and volume,
exactly), so the second feed is not a second opinion on the market but a safeguard against the first feed's faults: NSE answers late, not at
all (datacentre addresses are often refused), or with a name that does not fit. The rules, all per stock and all conservative:

* Each feed's rows are judged BEFORE the feeds are compared: a row is usable only when it is stamped on today's date (a stock that has not
  traded today still carries yesterday's stamp) and every figure a scan needs is there. A stale or incomplete row of one feed can therefore
  neither conflict with nor beat a good row of the other (yesterday's whole-day volume would outvote today's running one).
* A name only one feed has is taken from that feed. With NSE down, the broker alone carries the scan; with the broker down or unusable,
  NSE alone does, exactly as before the broker existed.
* A name both feeds have must AGREE: the previous close within 0.3 %, the open within 0.5 % and the last price within 3 %. Those three do not
  move between two snapshots seconds apart, so a bigger difference is a data fault (a corporate-action adjustment applied by one feed only,
  a wrong instrument): which side is right cannot be told, so the name is left out and counted.
* Among two agreeing rows the FRESHER one is taken whole, never a mix of fields: the day's volume only ever grows, so the row with the
  larger volume has seen more trades. A tie goes to NSE, the feed every model was validated on.
* During the session the broker feed must itself be current: if even the freshest of its ~200 stocks last traded more than
  ``KOTAK_MAX_LAG_SEC`` ago, the feed is lagging and is not used at all (a stock that is not trading is not stale, but not one of
  the most liquid stocks trading for minutes means the feed is). Outside the session no age is judged, so a preview run in the evening
  reads the day's final figures from both.
* The time of the snapshot is that of the rows actually taken: NSE's own stamp when NSE supplied rows, the newest believable trade time
  of the broker rows taken when the broker did (a stamp from the future is not believed). A feed whose rows all lost the comparison does
  not lend its stamp to the table.

The result says which feed every row came from, so the report can name its sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.clock import iso, ist_date, parse_ts
from ingestion.kotak import KotakQuotes, newest_trade
from ingestion.nse import LiveEquities

PREV_TOLERANCE = 0.003  # the previous close of a name: the same in both feeds unless one applied a corporate action the other did not
OPEN_TOLERANCE = 0.005  # the day's open is fixed once the session starts
LAST_TOLERANCE = 0.03  # two snapshots seconds apart never differ this much on a real market; a bigger gap is the wrong instrument
REQUIRED = ("open", "high", "low", "last", "prev_close", "volume")  # the figures a scan needs of a row


@dataclass
class Merged:
    rows: list[dict[str, Any]]  # the chosen rows, each with ``feed`` = "nse" | "kotak"
    source_ts: str | None  # when the rows taken were last updated (UTC): see the module docstring
    feeds: dict[str, int]  # rows taken from each feed
    cross_checked: int = 0  # names both feeds had and agreed on
    conflicts: dict[str, str] = field(default_factory=dict)  # names left out because the feeds disagreed -> why
    broker: str = "off"  # off (not configured) | failed | lagging | unstamped | used
    notes: list[str] = field(default_factory=list)  # what a report must not leave unsaid about the feeds
    stale: int = 0  # names left out because no feed had a row stamped on the day
    incomplete: int = 0  # names left out because no row of the day had every figure


def judge_row(row: dict[str, Any], day: date | None) -> str:
    """Why ``row`` cannot carry a scan: ``"stale"`` (not stamped on ``day``; the date is not judged when ``day`` is ``None``), ``"incomplete"``
    (a figure is missing), or ``""`` when it can."""
    if day is not None:
        stamp = parse_ts(row.get("row_ts"))
        if stamp is None or ist_date(stamp) != day:
            return "stale"
    return "incomplete" if any(row.get(key) is None for key in REQUIRED) else ""


def _usable(rows: list[dict[str, Any]], day: date | None, unusable: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    kept: dict[str, dict[str, Any]] = {}
    for row in rows:
        why = judge_row(row, day)
        if why:
            unusable.setdefault(row["symbol"], set()).add(why)
        else:
            kept[row["symbol"]] = row
    return kept


def _far(a: Any, b: Any, tolerance: float) -> bool:
    return bool(a) and bool(b) and abs(float(a) / float(b) - 1.0) > tolerance


def disagreement(nse: dict[str, Any], broker: dict[str, Any]) -> str:
    """Why two rows of one name cannot both be right (empty when they agree)."""
    if _far(nse.get("prev_close"), broker.get("prev_close"), PREV_TOLERANCE):
        return f"previous close {nse['prev_close']:g} (NSE) against {broker['prev_close']:g} (Kotak)"
    if _far(nse.get("open"), broker.get("open"), OPEN_TOLERANCE):
        return f"open {nse['open']:g} (NSE) against {broker['open']:g} (Kotak)"
    if _far(nse.get("last"), broker.get("last"), LAST_TOLERANCE):
        return f"last price {nse['last']:g} (NSE) against {broker['last']:g} (Kotak)"
    return ""


def merge_live(nse: LiveEquities | None, broker: KotakQuotes | None, *, configured: bool, day: date | None = None,
               lag_limit_sec: float = 0.0, judge_lag: bool = False) -> Merged:
    """The live rows of NSE (``nse``, ``None`` when it did not answer) and of the broker (``broker``, ``None`` when it did not answer or was
    not asked) as one table of rows usable on ``day`` (see ``judge_row``). ``configured`` says whether the broker is switched on at all;
    ``judge_lag`` whether the market is trading now, which is when the broker answer's age (``lag_limit_sec``, 0 = never) is held against it."""
    notes: list[str] = []
    state = "off" if not configured else "failed"
    use_broker = False
    if configured and broker is not None and broker.rows:
        lag = broker.lag_sec
        if judge_lag and lag_limit_sec > 0 and (lag is None or lag > lag_limit_sec):
            state = "unstamped" if lag is None else "lagging"
            notes.append("Kotak live feed not used: its quotes carry no trade time" if lag is None
                         else f"Kotak live feed not used: its newest trade was {lag:.0f} s old")
        else:
            state = "used"
            use_broker = True
    if state == "failed" and nse is not None:
        notes.append("Kotak live feed unavailable: NSE alone was used")
    unusable: dict[str, set[str]] = {}
    nse_rows = _usable(nse.rows if nse is not None else [], day, unusable)
    broker_rows = _usable(broker.rows, day, unusable) if use_broker and broker is not None else {}

    rows: list[dict[str, Any]] = []
    feeds = {"nse": 0, "kotak": 0}
    conflicts: dict[str, str] = {}
    agreed = 0
    for symbol in sorted(nse_rows.keys() | broker_rows.keys()):
        a, b = nse_rows.get(symbol), broker_rows.get(symbol)
        if a is not None and b is not None:
            why = disagreement(a, b)
            if why:
                conflicts[symbol] = why
                continue
            agreed += 1
            pick, feed = (b, "kotak") if b["volume"] > a["volume"] else (a, "nse")
        else:
            pick, feed = (a, "nse") if a is not None else (b, "kotak")
        rows.append({**pick, "feed": feed})
        feeds[feed] += 1
    if conflicts:
        notes.append(f"NSE and Kotak disagreed on {len(conflicts)} name(s), left out: " + ", ".join(sorted(conflicts)[:5])
                     + (" ..." if len(conflicts) > 5 else ""))
    if nse is None and state == "used":
        notes.append("NSE live feed unavailable: the Kotak broker feed alone was used")

    left = unusable.keys() - nse_rows.keys() - broker_rows.keys()
    incomplete = sum(1 for symbol in left if "incomplete" in unusable[symbol])  # a row of today with a figure missing outranks a stale one

    stamps = []
    if feeds["nse"] and nse is not None and (stamp := parse_ts(nse.source_ts)) is not None:
        stamps.append(stamp)
    if feeds["kotak"] and broker is not None and (stamp := newest_trade([r for r in rows if r["feed"] == "kotak"], broker.received)) is not None:
        stamps.append(stamp)
    source_ts = iso(max(stamps)) if stamps else (nse.source_ts if nse is not None else None)
    return Merged(rows, source_ts, feeds, agreed, conflicts, state, notes, stale=len(left) - incomplete, incomplete=incomplete)
