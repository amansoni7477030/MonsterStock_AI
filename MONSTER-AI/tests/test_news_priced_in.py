from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.clock import IST
from news_intelligence.priced_in import Reaction, assess_reaction


def ist(day: str, hh: int, mm: int = 0) -> datetime:
    y, m, d = (int(p) for p in day.split("-"))
    return datetime(y, m, d, hh, mm, tzinfo=IST)


def make_bars(last_day: str, n: int = 60, *, noise: float = 0.005, volume: float = 1000.0, seed: int = 7) -> pd.DataFrame:
    index = pd.bdate_range(end=last_day, periods=n)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0, noise, n))
    open_ = np.r_[100.0, close[:-1]]
    vol = volume * (1 + rng.normal(0.0, 0.05, n))
    return pd.DataFrame({"open": open_, "high": close * 1.002, "low": close * 0.998, "close": close, "volume": vol}, index=index)


def with_bar(bars: pd.DataFrame, day: str, *, open_: float, close: float, volume: float = 1000.0) -> pd.DataFrame:
    out = bars.copy()
    out.loc[pd.Timestamp(day)] = {"open": open_, "high": max(open_, close), "low": min(open_, close), "close": close,
                                  "volume": volume}
    return out.sort_index()


MON, TUE, WED = "2026-11-02", "2026-11-03", "2026-11-04"


@pytest.fixture()
def base():
    bars = make_bars(MON)
    return bars, float(bars["close"].iloc[-1])


def test_gap_up_after_an_evening_announcement_counts_as_reacted(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.06, close=prev * 1.07, volume=3000)
    reaction = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 17))
    assert reaction.reacted is True
    assert reaction.gap_pct == pytest.approx(6.0, abs=0.01) and reaction.move_since_event_pct == pytest.approx(7.0, abs=0.01)
    assert reaction.rel_volume == pytest.approx(3.0, abs=0.3)
    assert "daily bars" in reaction.note and "volume" in reaction.note
    assert set(reaction.to_dict()) == {"reacted", "gap_pct", "move_since_event_pct", "rel_volume", "note"}


def test_flat_session_means_not_priced_in(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.001, close=prev * 1.002)
    reaction = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 17))
    assert reaction.reacted is False and abs(reaction.move_since_event_pct) < 0.5


def test_normal_volatility_scales_the_threshold():
    noisy = make_bars(MON, noise=0.03, seed=3)  # ~3% daily vol: a 3% move is ordinary
    prev = float(noisy["close"].iloc[-1])
    ordinary = with_bar(noisy, TUE, open_=prev * 1.003, close=prev * 1.03)
    assert assess_reaction(ist(MON, 17), None, ordinary, ist(TUE, 17)).reacted is False
    big = with_bar(noisy, TUE, open_=prev * 1.003, close=prev * 1.15)
    assert assess_reaction(ist(MON, 17), None, big, ist(TUE, 17)).reacted is True


def test_session_bar_is_only_visible_after_16_00_ist(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.06, close=prev * 1.07)
    early = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 15, 59))
    assert early.reacted is None and "no post-event price" in early.note
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 16, 0)).reacted is True


def test_preopen_quote_is_used_before_the_open_when_no_bar_exists_yet(base):
    bars, prev = base
    quote_up = {"iep": prev * 1.06}
    reaction = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 10), preopen=quote_up)
    assert reaction.reacted is True and reaction.gap_pct == pytest.approx(6.0, abs=0.01)
    assert "pre-open" in reaction.note
    flat = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 10), preopen={"final_price": prev * 1.001})
    assert flat.reacted is False
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 10), preopen={"iep": float("nan")}).reacted is None
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 10), preopen={"iep": "junk"}).reacted is None
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 10), preopen={"iep": -5}).reacted is None
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 10), preopen=None).reacted is None
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 9, 30), preopen=quote_up).reacted is None  # session already open


def test_before_pre_open_nothing_has_traded_yet(base):
    bars, _ = base
    reaction = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 8, 30))
    assert reaction.reacted is False and "no price discovery" in reaction.note


def test_no_session_since_the_event_means_not_yet_reacted(base):
    bars, _ = base
    reaction = assess_reaction(ist(MON, 17), None, bars, ist(MON, 18, 30))
    assert reaction.reacted is False and reaction.move_since_event_pct is None and "no trading session" in reaction.note


def test_missing_post_event_bars_are_unknown_not_guessed(base):
    bars, _ = base
    stale = assess_reaction(ist(MON, 17), None, bars, ist(WED, 17))
    assert stale.reacted is None and "missing" in stale.note
    holes = with_bar(bars, "2026-11-25", open_=100, close=101)
    late = assess_reaction(ist(MON, 17), None, holes, ist("2026-11-26", 17))
    assert late.reacted is None and "too late" in late.note


def test_pre_market_event_already_reacted_before_we_first_saw_it(base):
    bars, prev = base
    bars = with_bar(bars.iloc[:-1], MON, open_=float(bars["close"].iloc[-2]) * 1.05, close=prev * 1.05)
    reaction = assess_reaction(ist(MON, 18), ist(MON, 8), bars, ist(MON, 18, 30))
    assert reaction.reacted is True and reaction.gap_pct == pytest.approx(5.0, abs=0.01)
    # an event_time later than first_seen is not trusted: we only know the item existed at first_seen (after the close)
    later = assess_reaction(ist(MON, 18), ist(MON, 23), bars, ist(MON, 18, 30))
    assert later.reacted is False and "no trading session" in later.note


def test_intraday_event_is_measured_from_the_session_open(base):
    bars, prev = base
    up = with_bar(bars, TUE, open_=prev, close=prev * 1.04, volume=2000)
    reaction = assess_reaction(ist(TUE, 11), None, up, ist(TUE, 17))
    assert reaction.reacted is True and reaction.gap_pct is None
    assert reaction.move_since_event_pct == pytest.approx(4.0, abs=0.01) and "intraday" in reaction.note
    flat = with_bar(bars, TUE, open_=prev, close=prev * 1.002)
    assert assess_reaction(ist(TUE, 11), None, flat, ist(TUE, 17)).reacted is False
    no_open = up.copy()
    no_open.loc[pd.Timestamp(TUE), "open"] = float("nan")
    assert assess_reaction(ist(TUE, 11), None, no_open, ist(TUE, 17)).reacted is True


def test_volume_surge_with_a_modest_move_counts_but_a_quiet_one_does_not(base):
    bars, prev = base
    loud = with_bar(bars, TUE, open_=prev * 1.001, close=prev * 1.006, volume=4500)
    quiet = with_bar(bars, TUE, open_=prev * 1.001, close=prev * 1.006, volume=1000)
    assert assess_reaction(ist(MON, 17), None, loud, ist(TUE, 17)).reacted is True
    assert assess_reaction(ist(MON, 17), None, quiet, ist(TUE, 17)).reacted is False


def test_sector_wide_moves_are_not_stock_specific_reactions(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.03, close=prev * 1.03)
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 17)).reacted is True
    adjusted = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 17), sector_return=3.0)
    assert adjusted.reacted is False and "sector +3.0%" in adjusted.note
    assert assess_reaction(ist(MON, 17), None, bars, ist(TUE, 17), sector_return=float("nan")).reacted is True


def test_stale_preopen_after_a_session_uses_the_indicative_price(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.001, close=prev * 1.002)
    reaction = assess_reaction(ist(MON, 17), None, bars, ist(WED, 9, 10), preopen={"iep": prev * 1.08})
    assert reaction.reacted is True and "pre-open" in reaction.note
    ignored = assess_reaction(ist(MON, 17), None, bars, ist(WED, 8, 30), preopen={"iep": prev * 1.08})
    assert ignored.reacted is False


@pytest.mark.parametrize(
    ("first_seen", "event_time", "as_of", "bars_arg", "expect"),
    [
        (None, None, ist(TUE, 17), "bars", "timestamps missing"),
        (ist(MON, 17), None, None, "bars", "timestamps missing"),
        (ist(TUE, 17), None, ist(MON, 17), "bars", "after the as-of"),
        (ist(MON, 17), None, ist(TUE, 17), None, "no price bars"),
        (ist(MON, 17), None, ist(TUE, 17), "empty", "no price bars"),
        (ist(MON, 17), None, ist(TUE, 17), "no_close", "no price bars"),
        (ist(MON, 17), None, ist(TUE, 17), "short", "only 5 daily bars"),
    ],
)
def test_missing_inputs_give_none(base, first_seen, event_time, as_of, bars_arg, expect):
    bars, _ = base
    chosen = {
        "bars": bars, None: None, "empty": pd.DataFrame(), "no_close": bars[["open", "volume"]], "short": bars.iloc[-6:-1],
    }[bars_arg]
    reaction = assess_reaction(first_seen, event_time, chosen, as_of)
    assert reaction.reacted is None and expect in reaction.note
    assert reaction.gap_pct is None and reaction.move_since_event_pct is None and reaction.rel_volume is None


def test_flat_prices_cannot_measure_volatility():
    index = pd.bdate_range(end=MON, periods=30)
    bars = pd.DataFrame({"open": 100.0, "close": 100.0, "volume": 1000.0}, index=index)
    bars = bars.iloc[:12]
    reaction = assess_reaction(ist(bars.index[-1].strftime("%Y-%m-%d"), 17), None, bars, ist(WED, 17), min_history=5)
    assert reaction.reacted is None


@pytest.mark.filterwarnings("ignore:Could not infer format")
def test_inputs_are_robust_to_index_types_duplicates_and_bad_rows(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.06, close=prev * 1.07, volume=3000)
    expected = assess_reaction(ist(MON, 17), None, bars, ist(TUE, 17))
    aware = bars.copy()
    aware.index = aware.index.tz_localize("Asia/Kolkata")
    assert assess_reaction(ist(MON, 17), None, aware, ist(TUE, 17)) == expected
    strings = bars.copy()
    strings.index = [d.strftime("%Y-%m-%d") for d in strings.index]
    assert assess_reaction(ist(MON, 17), None, strings, ist(TUE, 17)) == expected
    dirty = pd.concat([bars, bars.iloc[[-1]], bars.iloc[[0]]])
    dirty.loc[pd.Timestamp("2026-09-01")] = {"open": 1, "high": 1, "low": 1, "close": -1.0, "volume": 1}
    assert assess_reaction(ist(MON, 17), None, dirty, ist(TUE, 17)).reacted is True
    text_dates = bars.copy()
    text_dates.index = ["not-a-date"] + [d.strftime("%Y-%m-%d") for d in text_dates.index[1:]]
    assert assess_reaction(ist(MON, 17), None, text_dates, ist(TUE, 17)).reacted is True
    no_open = bars.drop(columns=["open"])
    assert assess_reaction(ist(MON, 17), None, no_open, ist(TUE, 17)).reacted is True


def test_strings_and_naive_timestamps_are_accepted(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.06, close=prev * 1.07)
    reaction = assess_reaction("2026-11-02T11:30:00Z", "2026-11-02T11:00:00Z", bars, "2026-11-03T11:30:00Z")
    assert isinstance(reaction, Reaction) and reaction.reacted is True


def test_missing_volume_leaves_relative_volume_unknown(base):
    bars, prev = base
    bars = with_bar(bars, TUE, open_=prev * 1.06, close=prev * 1.07)
    no_volume = bars.copy()
    no_volume["volume"] = float("nan")
    reaction = assess_reaction(ist(MON, 17), None, no_volume, ist(TUE, 17))
    assert reaction.reacted is True and reaction.rel_volume is None
