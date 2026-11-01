"""The exchange calendar must be right for ANY month and year: official lists first, rules where none exists, live confirmation."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from app.clock import at_ist
from automation import scan
from automation.calendar import (
    FIXED_HOLIDAYS,
    MIN_OFFICIAL_HOLIDAYS,
    ExchangeCalendar,
    easter_sunday,
    ensure_rule_holidays,
    refresh_calendar,
    rule_holidays,
    seed_calendar,
)
from automation.health import CALENDAR_HORIZON_WARN_DAYS, evaluate_health
from tests.seed_data import FakeNseApi, make_job_context

UTC = timezone.utc


def ist(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone(timedelta(hours=5, minutes=30))).astimezone(UTC)


@pytest.fixture()
def calendar(store, settings):
    seed_calendar(store, settings, ist(2026, 11, 2, 8, 30))
    return ExchangeCalendar(store)


# --------------------------------------------------------------------------------------------- rules
@pytest.mark.parametrize("year, easter", [
    (2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5)), (2027, date(2027, 3, 28)),
    (2028, date(2028, 4, 16)), (2030, date(2030, 4, 21)), (2038, date(2038, 4, 25)), (2000, date(2000, 4, 23)),
])
def test_easter_matches_the_known_dates(year, easter):
    assert easter_sunday(year) == easter


@pytest.mark.parametrize("year", [2026, 2027, 2028, 2029, 2030, 2031, 2040])
def test_rule_holidays_are_weekday_fixed_dates_plus_good_friday(year):
    found = {r["trade_date"]: r["description"] for r in rule_holidays(year)}
    assert all(date.fromisoformat(d).weekday() < 5 and d.startswith(str(year)) for d in found)
    good_friday = (easter_sunday(year) - timedelta(days=2)).isoformat()
    assert good_friday in found and "Good Friday" in found[good_friday]
    for month, day, name in FIXED_HOLIDAYS:
        d = date(year, month, day)
        assert (d.isoformat() in found) is (d.weekday() < 5), name  # weekend ones are closed anyway: no row needed
    assert all("rule-based" in text for text in found.values())


def test_a_year_without_an_official_list_gets_rule_based_holidays_everywhere(store, settings, calendar):
    assert not calendar.is_trading_day(date(2027, 1, 26))  # Republic Day (a Tuesday)
    assert not calendar.is_trading_day(date(2027, 3, 26))  # Good Friday
    assert not calendar.is_trading_day(date(2028, 12, 25))  # two years ahead
    assert calendar.is_trading_day(date(2027, 3, 29))  # an ordinary Monday
    assert "Republic Day" in calendar.holiday_name(date(2027, 1, 26))
    assert store.get_calendar_day("2027-01-26")["source"] == "rule"


def test_the_official_year_is_untouched_by_rules(store, calendar):
    assert calendar.is_official(date(2026, 6, 1)) and not calendar.is_official(date(2027, 1, 4))
    assert store.get_calendar_day("2026-01-26")["source"] == "seed"  # NSE's own list, not a rule
    assert not calendar.is_trading_day(date(2026, 10, 20)) and calendar.is_trading_day(date(2026, 12, 24))


def test_rule_rows_are_idempotent_and_never_overwrite_existing_rows(store, settings):
    store.upsert_calendar_days([{"trade_date": "2027-01-26", "is_trading_day": True, "session_kind": "special",
                                 "description": "operator says open"}], source="override")
    seed_calendar(store, settings, ist(2026, 11, 2))
    first = store.conn.execute("SELECT COUNT(*) FROM calendar_days WHERE source = 'rule'").fetchone()[0]
    seed_calendar(store, settings, ist(2026, 11, 2))
    assert store.conn.execute("SELECT COUNT(*) FROM calendar_days WHERE source = 'rule'").fetchone()[0] == first > 0
    assert store.get_calendar_day("2027-01-26")["source"] == "override"
    assert ExchangeCalendar(store).is_trading_day(date(2027, 1, 26))


def test_rule_years_follow_the_clock_not_the_machine_time_zone(store, settings):
    seed_calendar(store, settings, datetime(2026, 12, 31, 20, 0, tzinfo=UTC))  # already 1 Jan 2027 in IST
    years = {r[0] for r in store.conn.execute("SELECT DISTINCT substr(trade_date, 1, 4) FROM calendar_days WHERE source = 'rule'")}
    assert years == {"2027", "2028", "2029"}


def test_a_partial_official_list_does_not_count_as_official(store):
    few = [{"trade_date": f"2030-0{m}-10", "is_trading_day": False, "session_kind": "holiday", "description": "x"}
           for m in range(1, MIN_OFFICIAL_HOLIDAYS)]
    store.upsert_calendar_days(few, source="nse_api")
    cal = ExchangeCalendar(store)
    assert not cal.is_official(date(2030, 5, 1))
    assert ensure_rule_holidays(store, [2030]) > 0
    store.upsert_calendar_days([{"trade_date": "2030-09-10", "is_trading_day": False, "session_kind": "holiday",
                                 "description": "x"}], source="nse_api")
    cal.reload()
    assert cal.is_official(date(2030, 5, 1))


def test_status_reports_how_far_the_official_list_reaches(store, calendar):
    status = calendar.status(date(2026, 11, 2))
    assert status.official_years == (2026,) and status.official_through == date(2026, 12, 31) and status.days_left == 59
    assert status.rule_years == (2027, 2028)
    assert ExchangeCalendar(None).status(date(2026, 11, 2)).days_left is None


# --------------------------------------------------------------------------------- the official list arrives
class Nse:
    def __init__(self, days):
        self.days = days

    def holidays(self):
        return [{"trade_date": d, "description": f"holiday {d}", "special": False} for d in self.days]


def official_2027(extra=()):
    base = [f"2027-{m:02d}-{d:02d}" for m, d in ((1, 26), (3, 26), (4, 14), (5, 1), (8, 16), (10, 2), (10, 20), (11, 9), (12, 24))]
    return [*base, *extra]


def test_the_official_list_replaces_the_rules_of_its_year(store, calendar):
    assert store.get_calendar_day("2027-12-24") is None and store.get_calendar_day("2027-08-16") is None
    refresh_calendar(store, Nse(official_2027()))
    calendar.reload()
    assert calendar.is_official(date(2027, 6, 1))
    assert store.conn.execute("SELECT COUNT(*) FROM calendar_days WHERE source = 'rule' AND trade_date LIKE '2027-%'").fetchone()[0] == 0
    assert not calendar.is_trading_day(date(2027, 12, 24)) and not calendar.is_trading_day(date(2027, 8, 16))
    # Independence Day 2027 is a Sunday and Christmas 2027 a Saturday; whatever the rules guessed is gone if NSE did not list it
    assert calendar.is_trading_day(date(2027, 6, 1))
    assert store.get_calendar_day("2028-01-26")["source"] == "rule"  # other years still rule-based


def test_a_partial_api_answer_changes_nothing_it_should_not(store, calendar):
    refresh_calendar(store, Nse(["2027-01-26", "2027-03-26"]))
    calendar.reload()
    assert not calendar.is_official(date(2027, 6, 1)) and store.get_calendar_day("2027-04-14")["source"] == "rule"  # untouched


def test_a_holiday_nse_cancels_becomes_a_trading_day_again_but_the_past_is_kept(store, settings, fake_clock):
    fake_clock.set(ist(2027, 2, 1, 6, 0))
    seed_calendar(store, settings, fake_clock.now())
    refresh_calendar(store, Nse(official_2027(["2027-02-17", "2027-01-20"])))
    refresh_calendar(store, Nse(official_2027()))  # 17 Feb (future) and 20 Jan (past) are no longer listed
    assert store.get_calendar_day("2027-02-17") is None and store.get_calendar_day("2027-01-20") is not None


def test_operator_overrides_survive_upgrades_and_close_or_open_days(store, settings):
    (settings.data_dir).mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "calendar_overrides.json").write_text(json.dumps({
        "sessions": [{"date": "2027-02-06", "open": "09:15", "close": "12:00", "description": "special live session"}],
        "closures": [{"date": "2027-03-02", "description": "unscheduled closure"}]}), encoding="utf-8")
    seed_calendar(store, settings, ist(2026, 11, 2))
    cal = ExchangeCalendar(store)
    assert cal.is_trading_day(date(2027, 2, 6)) and cal.session(date(2027, 2, 6)).kind == "special"  # a Saturday
    assert not cal.is_trading_day(date(2027, 3, 2)) and cal.holiday_name(date(2027, 3, 2)) == "unscheduled closure"


def test_a_corrupt_override_file_is_ignored_with_a_warning(store, settings, caplog):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "calendar_overrides.json").write_text("{nope", encoding="utf-8")
    seed_calendar(store, settings, ist(2026, 11, 2))
    assert "unreadable" in caplog.text and ExchangeCalendar(store).is_trading_day(date(2026, 11, 2))


# ------------------------------------------------------------------------------------------------- health
def settings_with(settings, **kw):
    return settings.__class__(**{**settings.__dict__, **kw})


def codes(report):
    return {p.code for p in report.problems}


def test_health_is_quiet_while_the_official_list_has_plenty_of_runway(store, settings, calendar, fake_clock):
    fake_clock.set(ist(2026, 11, 2, 7, 0))
    report = evaluate_health(store, settings, fake_clock, calendar)
    assert not codes(report) & {"calendar_horizon", "calendar_unofficial"}
    assert report.checks["calendar_official_through"] == "2026-12-31" and report.checks["calendar_days_left"] == 59


def test_health_warns_when_the_official_list_is_about_to_end_or_has_ended(store, settings, calendar, fake_clock):
    fake_clock.set(ist(2026, 12, 31 - CALENDAR_HORIZON_WARN_DAYS, 7, 0))
    assert "calendar_horizon" in codes(evaluate_health(store, settings, fake_clock, calendar))
    fake_clock.set(ist(2027, 1, 5, 7, 0))
    report = evaluate_health(store, settings, fake_clock, calendar)
    assert "calendar_unofficial" in codes(report) and "rules only" in next(p.message for p in report.problems if p.code == "calendar_unofficial")
    refresh_calendar(store, Nse(official_2027()))
    assert "calendar_unofficial" not in codes(evaluate_health(store, settings, fake_clock, ExchangeCalendar(store)))


def test_health_does_not_cry_wolf_about_todays_bars_before_the_end_of_day_file_is_due(store, settings, calendar, fake_clock):
    for i in range(10):
        iid = store.upsert_instrument(f"S{i}", "equity", in_universe=1)
        store.upsert_bars(iid, [{"trade_date": "2026-10-30", "open": 1, "high": 2, "low": 1, "close": 1.5}], source="nse", source_rank=2)
    fake_clock.set(ist(2026, 11, 2, 16, 0))  # session over, end-of-day file not ingested yet (it runs ~17:45)
    assert "stale_bars" not in codes(evaluate_health(store, settings, fake_clock, calendar))
    fake_clock.set(ist(2026, 11, 2, 19, 0))  # three hours after the close the bars are overdue
    report = evaluate_health(store, settings, fake_clock, calendar)
    assert "stale_bars" in codes(report) and "2026-11-02" in next(p.message for p in report.problems if p.code == "stale_bars")


# -------------------------------------------------------------- the final scan confirms rule-based days live
FINAL = ist(2027, 3, 29, 9, 8)  # a Monday in a year without an official list
TODAY_ROW = {"symbol": "AAA", "iep": 100.0, "final_price": 101.0, "prev_close": 99.0}


@pytest.fixture()
def rule_year(settings, store, fake_clock):
    fake_clock.set(FINAL)
    store.upsert_instrument("AAA", "equity", in_universe=1)

    def build(api):
        ctx = make_job_context(settings, store, fake_clock, run_key="2027-03-29")
        ctx._cache["nse_api"] = api
        return ctx

    return build


def test_an_empty_pre_open_book_in_a_rule_based_year_means_a_closed_day(rule_year, store):
    # NSE answers, but with the PREVIOUS session's book (stamped 26 Mar): no pre-open session today
    ctx = rule_year(FakeNseApi(books=[[TODAY_ROW]], book_ts="26-Mar-2027 09:08:30"))
    result = scan.await_preopen(ctx, ["AAA"])
    assert result.reachable and result.usable == 0.0 and result.share == 0.0
    closed = scan.suspected_holiday(ctx, result)
    assert closed is not None and closed.status == "skipped" and "looks like an exchange holiday" in closed.message
    assert not ctx.calendar.is_trading_day(date(2027, 3, 29))  # recorded: later jobs of the day skip too
    assert store.get_calendar_day("2027-03-29")["source"] == "inferred"
    assert any(a["code"] == "holiday_inferred" for a in store.pending_alerts())


def test_an_unreachable_nse_proves_nothing_so_the_scan_goes_on(rule_year):
    ctx = rule_year(FakeNseApi(fail=True))
    result = scan.await_preopen(ctx, ["AAA"])
    assert not result.reachable and scan.suspected_holiday(ctx, result) is None
    assert ctx.calendar.is_trading_day(date(2027, 3, 29))


def test_a_book_with_todays_prices_confirms_the_session(rule_year):
    ctx = rule_year(FakeNseApi(books=[[TODAY_ROW]], book_ts="29-Mar-2027 09:08:30"))
    result = scan.await_preopen(ctx, ["AAA"])
    assert result.usable == 1.0 and result.share == 1.0 and result.reachable and scan.suspected_holiday(ctx, result) is None
    assert scan.wait_for_preopen(ctx, ["AAA"]) == 1.0  # the plain float API is unchanged


def test_an_official_year_is_trusted_without_asking(settings, store, fake_clock):
    fake_clock.set(ist(2026, 11, 2, 9, 8))
    ctx = make_job_context(settings, store, fake_clock, run_key="2026-11-02")
    assert scan.suspected_holiday(ctx, scan.PreopenWait(0.0, 0.0, True)) is None


def test_before_the_book_can_exist_nothing_is_concluded_and_force_overrides(rule_year, fake_clock):
    ctx = rule_year(FakeNseApi())
    fake_clock.set(at_ist(date(2027, 3, 29), "09:01:00"))
    assert scan.suspected_holiday(ctx, scan.PreopenWait(0.0, 0.0, True)) is None
    fake_clock.set(FINAL)
    ctx.force = True
    assert scan.suspected_holiday(ctx, scan.PreopenWait(0.0, 0.0, True)) is None
    ctx.force = False
    assert scan.suspected_holiday(ctx, scan.PreopenWait(0.0, 0.0, True)) is not None


def test_without_nse_access_the_rule_based_calendar_stands(settings, store, fake_clock):
    fake_clock.set(FINAL)
    cfg = settings_with(settings, nse_enabled=False)
    ctx = make_job_context(cfg, store, fake_clock, run_key="2027-03-29")
    assert scan.await_preopen(ctx, ["AAA"]) == scan.PreopenWait(0.0, 0.0, False)
    assert scan.suspected_holiday(ctx, scan.PreopenWait(0.0, 0.0, True)) is None
