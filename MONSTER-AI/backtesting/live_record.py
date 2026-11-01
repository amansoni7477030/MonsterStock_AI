"""Rolling live record of the FROZEN predictions (blueprint section 15 and acceptance criterion 23.4).

For the last 20 days, the last 60 days and all evaluated days (a window identical to a shorter one is skipped) it measures, from
the picks the system actually froze and the outcomes that followed:

* range-forecast error: MAE of the median forecast, pinball loss at q10 / q50 / q90, empirical coverage of the 80 % interval;
* the top-N hit rate at the primary threshold and at every other configured one (1 / 2 / 5 %);
* direction accuracy on sessions that closed away from the open (secondary: the model is not a trade signal);
* the realised range of the picks (mean and median) and the hit rate, next to every same-day baseline;
* top-k ranking quality of the whole eligible universe: NDCG@5 against realised ranges (with the expectation of a random
  ranking) and the daily rank correlation (rank IC).

Every number carries its sample size ``n`` and a 95 % day-block bootstrap interval (``metrics.day_block_bootstrap``): the picks
of one day share one market, so they are resampled together, never as independent trials. The model's picks are also sliced by
rank, tier, market regime, sector, liquidity tercile and catalyst event type; slices with fewer than ``MIN_SLICE_N`` picks are
not reported. Only effective snapshots are read and every day is scored on one stage (the caller passes ``stages``), exactly
as in ``backtesting.evaluate``. Nothing here measures a trading strategy."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from backtesting import metrics as M
from database.store import Store
from features import regime
from features.labels import threshold_name

WINDOWS: tuple[tuple[str, int | None], ...] = (("20d", 20), ("60d", 60), ("all", None))
MIN_SLICE_N = 5
NDCG_K = 5
INTERVAL = (0.10, 0.90)  # the forecast interval is q10..q90: 80 % nominal coverage
NOMINAL_COVERAGE = 0.8
LIQUIDITY_COLUMN = "t_tv_med20_cr"  # median traded value (Rs crore) in the frozen features
SLICE_METRICS = ("topn_hit_rate", "pick_range_mean", "range_mae_q50", "range_coverage_80")
FLAT = 1e-9  # a close within this many % of the open counts as flat

Row = dict[str, Any]


def _f(value: Any) -> float:
    """A finite float or NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _ok(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


# --------------------------------------------------------------------------------------------- snapshots
@dataclass(frozen=True)
class DayContext:
    """What one frozen snapshot says about its day: the probability of every eligible name and the slice labels."""

    table: dict[str, float]
    regime: tuple[str, ...]
    liquidity: dict[str, str]


def _regime_labels(payload: Mapping[str, Any]) -> tuple[str, ...]:
    market = (payload.get("meta") or {}).get("market") or {}
    try:
        labels = regime.describe(market.get("m_trend"), market.get("m_volreg"), market.get("m_risk"), market.get("m_expiry_share"))
    except (TypeError, ValueError):
        labels = {}
    if not labels:  # a snapshot without the raw market features still carries the labels its report was sent with
        labels = {str(k): str(v) for k, v in ((payload.get("report") or {}).get("regime") or {}).items()}
    out = [v for k, v in labels.items() if k != "event" and v and v != "n/a"]
    share = _f(market.get("m_expiry_share"))
    if np.isfinite(share):
        out.append("expiry-day" if share > 0.5 else "non-expiry-day")
    elif labels.get("event"):
        out.append("expiry-day")
    return tuple(f"regime:{label}" for label in out)


def _liquidity_buckets(features: Any, table: Mapping[str, float]) -> dict[str, str]:
    """symbol -> liquidity tercile (by median traded value) among the day's eligible names."""
    if not isinstance(features, Mapping):
        return {}
    try:
        column = list(features["columns"]).index(LIQUIDITY_COLUMN)
        rows = features["rows"]
    except (KeyError, ValueError, TypeError):
        return {}
    values = {s: _f(rows[s][column]) for s in (table or rows) if s in rows and len(rows[s]) > column}
    values = {s: v for s, v in values.items() if np.isfinite(v)}
    if len(values) < 3:
        return {}
    low, high = np.quantile(list(values.values()), [1.0 / 3.0, 2.0 / 3.0])
    return {s: "liquidity:low" if v < low else "liquidity:mid" if v < high else "liquidity:high" for s, v in values.items()}


def day_context(payload: Mapping[str, Any]) -> DayContext:
    """The probability table, regime labels and liquidity terciles of a frozen snapshot payload."""
    table = {str(s): float(p) for s, p in (payload.get("table") or {}).items() if np.isfinite(_f(p))}
    return DayContext(table, _regime_labels(payload), _liquidity_buckets(payload.get("features"), table))


# ------------------------------------------------------------------------------------------- frame / rows
def _frame(rows: Sequence[Mapping[str, Any]], primary_threshold: float) -> pd.DataFrame:
    """One row per evaluated prediction with the numbers the metrics need (NaN where the forecast or outcome is missing)."""
    records = []
    for r in rows:
        span, probabilities = r.get("range") or {}, r.get("p") or {}
        catalysts = (r.get("explanation") or {}).get("catalysts") or []
        first = catalysts[0] if catalysts and isinstance(catalysts[0], Mapping) else {}
        records.append({
            "trade_date": r["trade_date"], "snapshot_id": int(r["snapshot_id"]), "symbol": r["symbol"], "rank": int(r["rank"]),
            "tier": r["tier"], "is_baseline": int(r["is_baseline"]), "baseline_name": r["baseline_name"] or "",
            "range_pct": _f(r.get("range_pct")), "dir_ret_pct": _f(r.get("dir_ret_pct")), "p_up": _f(probabilities.get("up")),
            "q10": _f(span.get("q10")), "q50": _f(span.get("q50")), "q90": _f(span.get("q90")),
            "event": first.get("event_type"),
        })
    frame = pd.DataFrame.from_records(records)
    frame = frame[frame["range_pct"].notna()].reset_index(drop=True)
    frame["hit"] = (frame["range_pct"] > primary_threshold).astype(float)
    return frame


def _matrix(days: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(days, picks) matrix of ``values`` ordered by day, NaN padded: the input of the day-block bootstrap."""
    days = np.asarray(days, dtype=str)
    order = np.argsort(days, kind="stable")
    days, values = days[order], values[order]
    unique, start = np.unique(days, return_index=True)
    counts = np.diff(np.r_[start, len(days)])
    out = np.full((len(unique), int(counts.max()) if len(counts) else 0), np.nan)
    out[np.repeat(np.arange(len(unique)), counts), np.arange(len(days)) - np.repeat(start, counts)] = values
    return out


def _row(name: str, matrix: np.ndarray, *, slice_: str = "all", stat: Callable[..., Any] = np.nanmean,
         extra: Mapping[str, Any] | None = None) -> Row | None:
    """A metric row: the statistic of every value, its sample size and a 95 % day-block interval (None when too few days)."""
    finite = np.isfinite(matrix)
    n = int(finite.sum())
    if n == 0:
        return None
    value = float(stat(matrix[finite].reshape(1, -1), axis=1)[0])
    low, high = M.day_block_bootstrap(matrix, stat)
    return {"name": name, "slice": slice_, "value": value, "n": n, "ci_low": _ok(low), "ci_high": _ok(high),
            "extra": {"n_days": int(finite.any(axis=1).sum()), **(extra or {})}}


def _pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    diff = y - q
    return np.maximum(alpha * diff, (alpha - 1.0) * diff)


def _pick_series(picks: pd.DataFrame) -> dict[str, tuple[np.ndarray, Callable[..., Any], dict[str, Any]]]:
    """name -> (per-pick values, statistic, extra) of the model's picks; a NaN value means 'not applicable to this pick'."""
    r, q10, q50, q90 = (picks[c].to_numpy(float) for c in ("range_pct", "q10", "q50", "q90"))
    p_up, ret = picks["p_up"].to_numpy(float), picks["dir_ret_pct"].to_numpy(float)
    called = np.isfinite(p_up) & (p_up != 0.5) & np.isfinite(ret)
    moved = called & (np.abs(ret) > FLAT)
    covered = np.isfinite(r) & np.isfinite(q10) & np.isfinite(q90)
    return {
        "topn_hit_rate": (picks["hit"].to_numpy(float), np.nanmean, {}),
        "pick_range_mean": (r, np.nanmean, {}),
        "pick_range_median": (r, np.nanmedian, {}),
        "range_mae_q50": (np.abs(r - q50), np.nanmean, {}),
        "range_pinball_q10": (_pinball(r, q10, INTERVAL[0]), np.nanmean, {"alpha": INTERVAL[0]}),
        "range_pinball_q50": (_pinball(r, q50, 0.5), np.nanmean, {"alpha": 0.5}),
        "range_pinball_q90": (_pinball(r, q90, INTERVAL[1]), np.nanmean, {"alpha": INTERVAL[1]}),
        "range_coverage_80": (np.where(covered, ((r >= q10) & (r <= q90)).astype(float), np.nan), np.nanmean,
                              {"nominal": NOMINAL_COVERAGE}),
        "direction_accuracy": (np.where(moved, ((p_up > 0.5) == (ret > 0)).astype(float), np.nan), np.nanmean,
                               {"flat_excluded": int((called & ~moved).sum()),
                                "up_share": float(np.mean(ret[moved] > 0)) if moved.any() else None}),
    }


def pick_rows(picks: pd.DataFrame, names: Sequence[str] | None = None, slice_: str = "all") -> list[Row]:
    """Metric rows of the model's picks (all of them unless ``names`` is given); a metric that does not apply to any pick,
    e.g. direction accuracy for a model without a direction view, has no row."""
    if picks.empty:
        return []
    days = picks["trade_date"].to_numpy()
    series = _pick_series(picks)
    rows = (_row(name, _matrix(days, series[name][0]), slice_=slice_, stat=series[name][1], extra=series[name][2])
            for name in (series if names is None else names))
    return [row for row in rows if row is not None]


def threshold_rows(picks: pd.DataFrame, thresholds: Sequence[float], primary: float) -> list[Row]:
    """Hit rate of the picks at every configured range threshold other than the primary one (that is ``topn_hit_rate``):
    the blueprint's top-five threshold hit rates, each with n and a day-block interval."""
    if picks.empty:
        return []
    days, ranges = picks["trade_date"].to_numpy(), picks["range_pct"].to_numpy(float)
    rows = (_row(f"topn_hit_rate_{threshold_name(t)}", _matrix(days, np.where(np.isfinite(ranges), (ranges > t).astype(float), np.nan)),
                 extra={"threshold": float(t)}) for t in thresholds if t != primary)
    return [row for row in rows if row is not None]


def baseline_rows(baselines: pd.DataFrame) -> list[Row]:
    """The same-day baselines' hit rate and realised range, to put next to the picks'."""
    rows: list[Row | None] = []
    for name, group in baselines.groupby("baseline_name", sort=True):
        days = group["trade_date"].to_numpy()
        for suffix, column, stat in (("hit_rate", "hit", np.nanmean), ("range_mean", "range_pct", np.nanmean),
                                     ("range_median", "range_pct", np.nanmedian)):
            rows.append(_row(f"baseline_{name}_{suffix}", _matrix(days, group[column].to_numpy(float)), stat=stat))
    return [row for row in rows if row is not None]


# ------------------------------------------------------------------------------------------ ranking quality
def _per_day_ranking(universe: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Per day: NDCG@k of the frozen probability against realised ranges, the random-ranking expectation and the rank IC.

    Calibrated probabilities are step functions with many ties, so ties must not be broken by symbol order (that would
    reward or punish whatever the alphabet happens to do): each day is shuffled with a seed derived from its date."""
    records = {}
    for day, (scores, relevance) in sorted(universe.items()):
        order = np.random.default_rng(zlib.crc32(day.encode())).permutation(len(scores))
        scores, relevance = scores[order], relevance[order]
        records[day] = {"ndcg": M.ndcg_at_k(scores, relevance, NDCG_K), "random": M.ndcg_random_expectation(relevance, NDCG_K),
                        "ic": M.rank_correlation(scores, relevance)}
    return pd.DataFrame.from_dict(records, orient="index", columns=["ndcg", "random", "ic"])


def ranking_rows(per_day: pd.DataFrame) -> list[Row]:
    """NDCG@5 and rank IC over the days of ``per_day`` (n counts days)."""
    if per_day.empty:
        return []
    ndcg = per_day["ndcg"].to_numpy(float).reshape(-1, 1)
    random = per_day.loc[per_day["ndcg"].notna(), "random"]
    rows = [
        _row("ndcg_at_5", ndcg, extra={"k": NDCG_K, "random": _ok(float(random.mean())) if len(random) else None}),
        _row("rank_ic", per_day["ic"].to_numpy(float).reshape(-1, 1)),
    ]
    return [row for row in rows if row is not None]


# ---------------------------------------------------------------------------------------------- slices
def _sector_of(store: Store, symbol: str, day: str) -> str | None:
    """The sector as known on ``day`` (point-in-time history when it exists, else the current one)."""
    row = store.get_instrument(symbol)
    if row is None:
        return None
    attrs = store.instrument_attrs_asof(int(row["instrument_id"]), day) or {}
    return attrs.get("sector") or row["sector"]


def _slice_labels(picks: pd.DataFrame, contexts: Mapping[int, DayContext], store: Store) -> list[list[str]]:
    sectors: dict[tuple[str, str], str | None] = {}
    labels = []
    for r in picks.itertuples(index=False):
        context = contexts.get(r.snapshot_id)
        one = [f"rank:{r.rank}", f"tier:{r.tier}"]
        if context is not None:
            one += context.regime
            if r.symbol in context.liquidity:
                one.append(context.liquidity[r.symbol])
        key = (r.symbol, r.trade_date)
        if key not in sectors:
            sectors[key] = _sector_of(store, r.symbol, r.trade_date)
        if sectors[key]:
            one.append(f"sector:{str(sectors[key])[:40]}")
        if isinstance(r.event, str) and r.event:
            one.append(f"event:{r.event[:40]}")
        labels.append(one)
    return labels


def slice_rows(picks: pd.DataFrame, labels: Sequence[Sequence[str]]) -> list[Row]:
    """The model's picks per slice (a pick belongs to one slice of each dimension). A slice, and any of its metrics, with fewer
    than ``MIN_SLICE_N`` observations is left out: a handful of picks says nothing."""
    if picks.empty:
        return []
    exploded = picks.assign(slice=pd.Series(list(labels), index=picks.index, dtype=object)).explode("slice")
    rows: list[Row] = []
    for label, group in exploded.dropna(subset=["slice"]).groupby("slice", sort=True):
        if len(group) >= MIN_SLICE_N:
            rows += [row for row in pick_rows(group, SLICE_METRICS, str(label)) if row["n"] >= MIN_SLICE_N]
    return rows


# ----------------------------------------------------------------------------------------------- the record
def build(store: Store, settings: Settings, trade_date: str, stages: Mapping[str, str]) -> dict[str, list[Row]]:
    """window label -> metric rows as of ``trade_date``; ``stages`` maps each day to the stage that scores it."""
    rows = [r for r in store.evaluated_rows(until=trade_date) if stages.get(r["trade_date"]) == r["stage"]]
    if not rows:
        return {}
    frame = _frame(rows, settings.primary_threshold)
    if frame.empty:
        return {}
    days = sorted(frame["trade_date"].unique())
    contexts = {int(sid): day_context((store.get_snapshot(int(sid)) or {}).get("payload") or {})
                for sid in frame["snapshot_id"].unique()}
    snapshot_of_day = frame.groupby("trade_date")["snapshot_id"].first()
    realised = store.realised_ranges(days[0], days[-1])
    universe: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for day in days:
        table, ranges = contexts[int(snapshot_of_day[day])].table, realised.get(day) or {}
        names = [s for s in table if s in ranges]
        if len(names) >= 2:
            universe[day] = (np.array([table[s] for s in names]), np.array([ranges[s] for s in names]))
    ranking = _per_day_ranking(universe)
    model = frame[frame["is_baseline"] == 0].reset_index(drop=True)
    labels = _slice_labels(model, contexts, store)
    record: dict[str, list[Row]] = {}
    seen = 0
    for label, size in WINDOWS:
        window = days if size is None else days[-size:]
        if len(window) <= seen:
            continue  # the same days as the shorter window already reported
        seen = len(window)
        in_window = model["trade_date"].isin(window).to_numpy()
        picks = model[in_window]
        record[label] = (
            pick_rows(picks)
            + threshold_rows(picks, settings.range_thresholds, settings.primary_threshold)
            + baseline_rows(frame[(frame["is_baseline"] == 1) & frame["trade_date"].isin(window)])
            + ranking_rows(ranking[ranking.index.isin(window)])
            + slice_rows(picks.reset_index(drop=True), [x for x, keep in zip(labels, in_window, strict=True) if keep])
        )
    return record
