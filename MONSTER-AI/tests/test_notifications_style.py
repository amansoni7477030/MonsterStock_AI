"""The visual vocabulary of the Telegram messages: bars, clock faces, badges, severity colours and next-step hints."""

from __future__ import annotations

import shlex
from datetime import date

import pytest

from app import cli
from notifications import style


@pytest.mark.parametrize(("p", "expected"), [
    (0.0, "░░░░░░░░░░"), (0.04, "░░░░░░░░░░"), (0.05, "█░░░░░░░░░"), (0.624, "██████░░░░"), (0.95, "██████████"),
    (1.0, "██████████"),
])
def test_the_bar_rounds_to_the_nearest_tenth(p, expected):
    assert style.bar(p) == expected


@pytest.mark.parametrize("value", [None, True, "abc", float("nan"), float("inf"), -0.2, 1.5, [0.5], {}])
def test_nothing_is_drawn_for_a_value_that_is_not_a_probability(value):
    assert style.bar(value) == ""


def test_the_bar_width_is_configurable_and_always_exact():
    assert style.bar(0.5, width=4) == "██░░" and all(len(style.bar(x / 20)) == style.BAR_WIDTH for x in range(21))


def test_clock_faces_follow_the_twelve_hour_dial():
    assert [style.clock(h) for h in (0, 1, 8, 9, 12, 13, 23)] == ["🕛", "🕐", "🕗", "🕘", "🕛", "🕐", "🕚"]
    for unknown in (None, -1, 25, True):
        assert style.clock(unknown) == style.NO_CLOCK


def test_medals_go_to_the_first_three_and_tiers_have_badges():
    assert [style.rank_badge(n) for n in (1, 2, 3, 4, 9)] == ["🥇", "🥈", "🥉", "🏅", "🏅"]
    assert [style.tier_badge(t) for t in ("STRONG", " moderate ", "WATCH", "other", "")] == ["🔥", "⚡", "👀", "🔹", "🔹"]


@pytest.mark.parametrize(("text", "dot"), [
    ("ok (3 min old)", "🟢"), ("OK", "🟢"), ("WARN - one late", "🟡"), ("degraded", "🟡"), ("STALE - quote is 600 min old", "🟠"),
    ("MISSING - none", "🔴"), ("blocked", "🔴"), ("failed", "🔴"), ("???", "⚪"), ("", "⚪"), (None, "⚪"),
])
def test_status_dots_follow_the_first_word(text, dot):
    assert style.status_dot(text) == dot


def test_severity_colours_form_a_ladder_and_good_news_gets_a_tick():
    assert [style.severity_icon(level) for level in ("critical", "ERROR", " warning ", "warn", "info", "", "weird")] == [
        "🔴", "🟠", "🟡", "🟡", "🔵", "🔵", "⚪"]
    assert style.severity_icon("info", "health_recovered") == "✅" and style.severity_icon("info", "bootstrap_complete") == "🎉"
    assert style.severity_icon("critical", "bootstrap_complete") == "🎉"  # the code decides: it is never a problem


def test_dates_use_english_names_whatever_the_servers_locale():
    assert style.pretty_date(date(2026, 11, 2)) == "Mon 02 Nov 2026"
    assert [style.pretty_date(date(2026, m, 1)).split()[2] for m in range(1, 13)] == list(style.MONTHS)
    assert style.pretty_date(date(2027, 1, 3)) == "Sun 03 Jan 2027"


def test_every_decoration_is_one_code_point_so_it_renders_the_same_everywhere():
    icons = [*style.TIER_BADGES.values(), *(icon for icon, _ in style.REGIME.values()), *style._SEVERITY.values(),
             *style._GOOD_NEWS.values(), *style._CLOCKS, *style._MEDALS, style.NO_CLOCK, style.OTHER_RANK, style.WEAK_RANK,
             style.WEAK_BADGE, style.BULLET, style.UNKNOWN_TIER]
    assert all(len(icon) == 1 for icon in icons), [icon for icon in icons if len(icon) != 1]
    assert style.WARN == "\u26a0\ufe0f"  # the one deliberate exception: the warning sign needs its variation selector
    assert len(style.RULE) == 18 and set(style.RULE) == {"\u2501"}


# ------------------------------------------------------------------------------------------ next steps
@pytest.mark.parametrize(("code", "command"), [
    ("scan_final_failed", "sudo monster-ai logs --job final_scan"),
    ("scan_pre_failed", "sudo monster-ai logs --job prelim_scan"),
    ("job_failed_collect_india", "sudo monster-ai logs --job collect_india"),
    ("job_crashed_train", "sudo monster-ai logs --job train"),
    ("job_missed_backup", "sudo monster-ai jobs"),
    ("daemon_restarted", "sudo monster-ai logs --role daemon"),
    ("health_storage_budget", "sudo monster-ai storage"),
    ("health_retention_stale", "sudo monster-ai logs --job housekeeping"),
    ("health_telegram_not_configured", "sudo monster-ai telegram-test"),
    ("health_source_yahoo", "sudo monster-ai doctor"),
    ("health_db_corrupt", "sudo monster-ai health"),
    ("resources", "sudo monster-ai storage"),
    ("model_load_failed", "sudo monster-ai models list"),
    ("training_failed", "sudo monster-ai logs --role train"),
    ("backup_upload_failed", "sudo monster-ai logs --job backup"),
    ("telegram_delivery_failed", "sudo monster-ai telegram-test"),
    ("report_deadline_missed", "sudo monster-ai jobs"),
])
def test_an_alert_code_maps_to_the_first_command_to_run(code, command):
    assert style.alert_hint(code) == command


@pytest.mark.parametrize("code", ["", None, "feed_down", "something_new", "disk"])
def test_an_unknown_code_has_no_suggestion(code):
    assert style.alert_hint(code) is None


@pytest.mark.parametrize("suffix", ["bad name", "x;rm -rf /", "a" * 41, "", "$(id)", "<b>"])
def test_an_unsafe_job_suffix_is_never_echoed_into_a_command(suffix):
    assert style.alert_hint(f"job_failed_{suffix}") == "sudo monster-ai logs"


def test_a_scan_hint_names_the_job_of_that_stage():
    assert style.scan_hint("pre") == "sudo monster-ai logs --job prelim_scan"
    assert style.scan_hint(" FINAL ") == "sudo monster-ai logs --job final_scan"
    assert style.scan_hint("closing") == "sudo monster-ai logs --job closing_scan"
    assert style.scan_hint("weird") == style.scan_hint(None) == "sudo monster-ai jobs"


def test_every_suggested_command_is_a_real_cli_command_with_valid_options():
    """A hint that names a command or option that does not exist would send the operator on a wild goose chase."""
    parser = cli.build_parser()
    for command in _all_suggested_commands():
        words = shlex.split(command)
        assert words[:2] == ["sudo", "monster-ai"], command
        parser.parse_args(words[2:])  # exits with an error on an unknown command or option


def test_no_suggested_command_changes_anything():
    """An alert tells the operator where to look; the docs promise it never suggests a command that modifies the system."""
    read_only = {"logs", "jobs", "health", "storage", "doctor", "telegram-test"}
    for command in _all_suggested_commands():
        name, *rest = shlex.split(command)[2:]
        if name == "models":
            assert rest[:1] == ["list"], command
        elif name == "scan":
            assert "--dry-run" in rest, command  # prints the report; freezes and sends nothing
        else:
            assert name in read_only and not {"--prune", "--vacuum", "--force", "--follow"} & set(rest), command


def _all_suggested_commands() -> set[str]:
    commands = {command.replace("{rest}", "collect_india") for _, command in style._HINTS}
    return commands | {style.scan_hint("pre"), style.scan_hint("final"), style.scan_hint("closing"), style.scan_hint(None),
                       "sudo monster-ai logs", "sudo monster-ai health"}
