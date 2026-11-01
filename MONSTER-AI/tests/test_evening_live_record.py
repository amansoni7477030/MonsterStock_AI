"""The evening report's live record: range-forecast error, interval coverage, direction accuracy, realised range versus baselines
and ranking quality, always with n and an interval, escaped and within Telegram's limits."""

from __future__ import annotations

import pytest

from backtesting import evaluate as ev
from notifications.report import render_evening_report
from tests.record_support import Pick, names, seed_days, seed_universe
from tests.test_notifications_report import assert_telegram_safe, lines, make_summary


def row(name, value, n=None, low=None, high=None, slice_="all", **extra):
    return {"name": name, "slice": slice_, "value": value, "n": n, "ci_low": low, "ci_high": high, "extra": extra or None}


def window_20d():
    return [
        row("topn_hit_rate", 0.6, 100, 0.52, 0.68), row("range_mae_q50", 1.4231, 100, 1.2, 1.65),
        row("range_coverage_80", 0.71, 100, 0.62, 0.79), row("direction_accuracy", 0.54, 96, 0.45, 0.63),
        row("pick_range_mean", 4.12, 100), row("pick_range_median", 3.8, 100),
        row("baseline_atr_range_mean", 3.6, 100), row("baseline_atr_range_median", 3.4, 100),
        row("baseline_random_range_mean", 2.4, 100), row("baseline_random_range_median", 2.2, 100),
        row("ndcg_at_5", 0.71, 20, 0.6, 0.8, random=0.45, k=5), row("rank_ic", -0.12, 18, -0.2, -0.02),
        row("topn_hit_rate", 0.99, 8, slice_="rank:1"), row("range_mae_q50", 9.9, 8, slice_="regime:bull"),  # slices stay in the database
    ]


def window_60d():
    return [row("topn_hit_rate", 0.58, 290, 0.53, 0.63), row("range_mae_q50", 1.5, 290, 1.31, 1.69),
            row("range_coverage_80", 0.74, 290, 0.7, 0.78), row("direction_accuracy", 0.51, 280, 0.47, 0.55),
            row("pick_range_mean", 4.0, 290), row("pick_range_median", 3.7, 290),
            row("baseline_atr_range_mean", 3.5, 290), row("baseline_atr_range_median", 3.3, 290),
            row("ndcg_at_5", 0.69, 58, 0.64, 0.74, random=0.46, k=5), row("rank_ic", 0.08, 55, 0.03, 0.13)]


def render(record, **over):
    return render_evening_report(make_summary(live_record=record, **over))


def test_the_live_record_block_has_one_line_per_metric_with_every_window_n_and_interval():
    messages = render({"20d": window_20d(), "60d": window_60d()})
    assert_telegram_safe(messages)
    got = lines(messages)
    for line in (
        "📚 <b>Live record: frozen picks against realised outcomes (95% day-block intervals)</b>",
        "🔹 Hit rate: 20d 60% [52% to 68%] n=100 | 60d 58% [53% to 63%] n=290",
        "🔹 Range MAE of the median forecast (pp): 20d 1.42 [1.20 to 1.65] n=100 | 60d 1.50 [1.31 to 1.69] n=290",
        "🔹 80% interval coverage (nominal 80%): 20d 71% [62% to 79%] n=100 | 60d 74% [70% to 78%] n=290",
        "🔹 Direction accuracy (non-flat closes): 20d 54% [45% to 63%] n=96 | 60d 51% [47% to 55%] n=280",
        "🔹 Realised range, picks vs baselines (60d): picks mean 4.0% / median 3.7% | atr 3.5% / 3.3%",
        "🔹 Ranking quality (60d): NDCG@5 0.69 [0.64 to 0.74] n=58 days vs random 0.46 | rank IC +0.08 [+0.03 to +0.13] n=55 days",
    ):
        assert line in got, line
    text = "\n".join(got)
    assert "99%" not in text and "9.9" not in text  # slices are not part of the report


def test_the_block_follows_the_existing_lines_and_keeps_them_unchanged():
    got = lines(render({"20d": window_20d()}))
    for line in ("🏁 <b>Top-3 hit rate:</b> 1/2 (CI 9%-91%)", "🆚 <b>Baseline hit rates:</b> atr 40%, random 20%",
                 "📈 <b>Rolling record (out-of-sample):</b> hit_rate_30d 41%, brier_30d 0.190, n_days 42, weird n/a"):
        assert line in got
    assert got.index("🧮 <b>Calibration:</b> Slightly overconfident at the top bin") < got.index(
        "📚 <b>Live record: frozen picks against realised outcomes (95% day-block intervals)</b>")


def test_a_single_window_and_missing_numbers_degrade_to_n_a_without_inventing_anything():
    record = {"20d": [row("range_mae_q50", None, 0), row("range_coverage_80", float("nan"), 4, float("nan"), 0.5),
                      row("topn_hit_rate", 1.7, 12, 0.5, 0.9), row("direction_accuracy", 0.5), row("ndcg_at_5", 0.5, 7)]}
    messages = render(record)
    assert_telegram_safe(messages)
    got = lines(messages)
    assert "🔹 Hit rate: 20d n/a [50% to 90%] n=12" in got  # 170 % is not a probability
    assert "🔹 Range MAE of the median forecast (pp): 20d n/a n=0" in got
    assert "🔹 80% interval coverage (nominal 80%): 20d n/a n=4" in got
    assert "🔹 Direction accuracy (non-flat closes): 20d 50%" in got  # no interval, no n: just the value
    assert "🔹 Ranking quality (20d): NDCG@5 0.50 n=7 days" in got
    assert not [ln for ln in got if "Realised range" in ln]  # nothing to compare


def test_the_ranking_sample_size_is_singular_for_a_single_day():
    got = lines(render({"20d": [row("ndcg_at_5", 0.9, 1, random=0.5), row("rank_ic", 0.3, 1)]}))
    assert "🔹 Ranking quality (20d): NDCG@5 0.90 n=1 day vs random 0.50 | rank IC +0.30 n=1 day" in got


def test_without_a_record_the_report_says_so_instead_of_staying_silent():
    for empty in ({}, None, "oops", {"20d": []}, {"20d": ["not a row", 3, None]}, {"20d": [row("x", 1.0, 1, slice_="rank:1")]}):
        got = lines(render(empty))
        assert "📚 Live record: n/a (no evaluated picks yet)" in got, empty
        assert not any(ln.startswith("🔹 Hit rate:") for ln in got)


def test_every_dynamic_value_of_the_record_is_escaped():
    record = {"<b>9d</b>": [row("topn_hit_rate", 0.5, 10, 0.4, 0.6), row("pick_range_mean", 4.0, 10),
                            row("pick_range_median", 3.0, 10), row("baseline_<i>x</i>&y_range_mean", 3.0, 10),
                            row("baseline_<i>x</i>&y_range_median", 2.0, 10)]}
    messages = render(record)
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "&lt;b&gt;9d&lt;/b&gt; 50%" in text and "&lt;i&gt;x&lt;/i&gt;&amp;y 3.0% / 2.0%" in text


def test_many_windows_and_baselines_stay_within_the_message_limits():
    baselines = [row(f"baseline_b{i}_range_{kind}", 2.0 + i / 10, 50) for i in range(40) for kind in ("mean", "median")]
    record = {f"{w}d": window_60d() + baselines for w in (20, 60, 90, 120)}
    messages = render(record)
    assert_telegram_safe(messages)
    got = lines(messages)
    realised = next(ln for ln in got if ln.startswith("🔹 Realised range"))
    assert "(120d)" in realised and realised.count(" | ") == 8  # picks plus at most 8 baselines: capped, never unbounded
    assert any(ln.startswith("🔹 Hit rate: 20d") and "120d" in ln for ln in got)


def test_windows_are_listed_from_the_shortest_to_all_time():
    record = {"all": window_60d(), "60d": window_60d(), "20d": window_20d()}
    hit = next(ln for ln in lines(render(record)) if ln.startswith("🔹 Hit rate:"))
    assert hit.index("20d") < hit.index("60d") < hit.index("all")


def test_the_dataclass_default_renders_the_empty_block():
    from ranking.types import EveningSummary

    summary = EveningSummary("2026-11-02", "final", 3.0, "fake-model-0.1")
    assert "📚 Live record: n/a (no evaluated picks yet)" in lines(render_evening_report(summary))


# ------------------------------------------------------------------------- from a real evaluation to Telegram
UNIVERSE = names(12)


@pytest.mark.parametrize("days", [3, 25])
def test_an_evaluated_record_reaches_the_report_with_its_sample_sizes(store, settings, days):
    seed_universe(store, UNIVERSE)

    def plan(i, day):
        realised = {s: (1.0 + 0.5 * j + 0.1 * (i % 3), 0.2) for j, s in enumerate(UNIVERSE)}
        table = {s: 0.05 + 0.07 * j for j, s in enumerate(UNIVERSE)}
        return {"picks": [Pick(s, up=0.6) for s in UNIVERSE[-5:]], "table": table, "realised": realised,
                "baselines": {"atr": UNIVERSE[-5:], "random": UNIVERSE[:5]}}

    seed_days(store, days, plan)
    summary = ev.evaluate_day(store, settings, store.prediction_dates()[-1])
    got = lines(render_evening_report(summary))
    assert any(ln.startswith("🔹 Range MAE of the median forecast (pp): 20d ") and f"n={5 * min(days, 20)}" in ln for ln in got)
    assert any(ln.startswith("🔹 Ranking quality (20d): NDCG@5 1.00") or ln.startswith("🔹 Ranking quality (60d): NDCG@5 1.00") for ln in got)
    assert any(ln.startswith("🔹 Realised range, picks vs baselines") and "random" in ln and "atr" in ln for ln in got)
    assert any(ln.startswith("🔹 Hit rate: 20d") for ln in got)
