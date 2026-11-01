"""The futures-positioning contradiction must be reachable on the features the builder really produces. It never was: it demanded
that yesterday's price move have the opposite sign of the price/OI code, which is derived from that very move (0 of 6,600 rows)."""

from __future__ import annotations

import numpy as np
import pytest

from features.builder import build_feature_set
from ranking import signals as sg
from tests.synthetic import make_panel

THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
READS = {1.0: ("bullish", "long build-up"), 2.0: ("bearish", "short build-up"), 3.0: ("bullish", "short covering"),
         4.0: ("bearish", "long unwinding")}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, float]]:
    panel = make_panel(n_dates=330, n_symbols=24, seed=11)
    fs = build_feature_set(panel, stage="final", thresholds=THRESHOLDS, unusual_k=1.5, cutoff_hhmm="08:45", with_labels=False,
                           row_slice=slice(130, 330))
    keep = fs.X["f_oi_price_code"].notna()
    return fs.X[keep].to_dict("records")


def positioning(row: dict[str, float]) -> list[str]:
    return [i for i in sg.contradictions(row, [], None, None).items if i.startswith("F&O positioning reads")]


def test_the_price_oi_code_is_fixed_by_the_sign_of_yesterdays_move(rows):
    """Why the old rule (code and the OPPOSITE price move) could never fire."""
    assert len(rows) > 1000
    for row in rows:
        up = row["t_ret1"] > 0
        assert (row["f_oi_price_code"] in (1.0, 3.0)) == up


def test_positioning_contradictions_fire_on_real_feature_rows_and_agree_with_their_reading(rows):
    fired = [(row, positioning(row)) for row in rows]
    hits = [(row, items) for row, items in fired if items]
    assert 20 < len(hits) < len(rows) / 2  # reachable, and still an exception rather than the rule
    for row, items in hits:
        assert len(items) == 1
        read, label = READS[row["f_oi_price_code"]]
        sign = 1 if read == "bullish" else -1
        assert f"reads {read} ({label}" in items[0]
        gap, trend = row.get("p_gap_pct"), row.get("t_ret5")
        against_gap = gap is not None and np.isfinite(gap) and gap * sign < -sg.OI_CONTRA_GAP_PCT
        against_trend = trend is not None and np.isfinite(trend) and trend * sign < -sg.OI_CONTRA_TREND_PCT
        assert against_gap or against_trend
        assert abs(row["f_fut_oi_chg_pct"]) > sg.OI_MIN_CHANGE_PCT


def test_both_price_action_sources_and_both_directions_are_reachable(rows):
    texts = [item for row in rows for item in positioning(row)]
    assert any("indicated" in t for t in texts) and any("5-day return" in t for t in texts)
    assert any("reads bullish" in t for t in texts) and any("reads bearish" in t for t in texts)
