"""Reading logs (``app.logview`` + ``cli logs``) and the diagnostics bundle (``app.diagnostics`` + ``cli diag``)."""

from __future__ import annotations

import json
import os
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import cli, logview

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
SAMPLE = [
    "2026-11-02T08:30:00.000+0530 INFO    [-] app.cli: start daemon pid=1",
    "2026-11-02T08:45:01.250+0530 WARNING [final_scan 2026-11-02#1 r7] ingestion.http: http nse_api 429, retrying in 4.0 s",
    "2026-11-02T08:45:02.000+0530 ERROR   [final_scan 2026-11-02#1 r7] automation.runner: job final_scan crashed",
    "Traceback (most recent call last):",
    '  File "x.py", line 1, in <module>',
    "ValueError: boom",
    "2026-11-02T09:00:00.000+0530 INFO    [collect_news 2026-11-02#1 r8] automation.jobs: 12 new items",
]
ERRORS = [
    "2026-11-02T08:45:01.250+0530 WARNING jobs[4242] [final_scan 2026-11-02#1 r7] ingestion.http: http nse_api 429",
    "2026-11-02T08:45:02.000+0530 ERROR   jobs[4242] [final_scan 2026-11-02#1 r7] automation.runner: job final_scan crashed",
    "Traceback (most recent call last):",
    "ValueError: boom",
]


def put(folder: Path, name: str, lines: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------- files
def test_log_files_are_listed_oldest_day_first_and_foreign_files_ignored(tmp_path):
    put(tmp_path, "jobs-2026-11-02.log", SAMPLE)
    put(tmp_path, "jobs-2026-11-02.1.log", SAMPLE)
    put(tmp_path, "daemon-2026-11-01.log", SAMPLE)
    put(tmp_path, "notes.txt", ["x"])
    put(tmp_path, "fault-daemon.log", [])
    files = logview.list_log_files(tmp_path)
    assert [(f.day, f.role, f.index) for f in files] == [("2026-11-01", "daemon", 0), ("2026-11-02", "jobs", 0), ("2026-11-02", "jobs", 1)]
    assert logview.list_log_files(tmp_path / "missing") == []
    assert "daemon-2026-11-01.log" in logview.summary(files) and logview.summary([]) == "no log files yet"


def test_select_defaults_to_the_newest_day_and_filters_by_role(tmp_path):
    for name in ("jobs-2026-11-01.log", "jobs-2026-11-02.log", "daemon-2026-11-02.log", "errors-2026-11-02.log", "cli-2026-10-30.log"):
        put(tmp_path, name, SAMPLE)
    files = logview.list_log_files(tmp_path)
    assert [f.role for f in logview.select_files(files, role="all")] == ["daemon", "errors", "jobs"]
    assert [f.day for f in logview.select_files(files, role="jobs")] == ["2026-11-02"]
    assert [f.day for f in logview.select_files(files, role="jobs", day="2026-11-01")] == ["2026-11-01"]
    assert [f.role for f in logview.select_files(files, role="cli")] == ["cli"]  # its own newest day
    assert logview.select_files(files, role="ghost") == [] and logview.select_files([], role="all") == []


# -------------------------------------------------------------------------------------------- records
def test_records_keep_their_traceback_and_parse_the_stamp_with_its_offset():
    records = list(logview.group_records(SAMPLE))
    assert [r.level for r in records] == ["INFO", "WARNING", "ERROR", "INFO"]
    crashed = records[2]
    assert crashed.lines[-1] == "ValueError: boom" and len(crashed.lines) == 4 and crashed.job == "final_scan 2026-11-02#1 r7"
    assert crashed.stamp is not None and crashed.stamp.utcoffset() == timedelta(hours=5, minutes=30)
    assert crashed.stamp.astimezone(timezone.utc) == datetime(2026, 11, 2, 3, 15, 2, tzinfo=timezone.utc)
    assert logview.parse_stamp("not a stamp") is None


def test_filters_by_level_job_text_and_time():
    records = list(logview.group_records(SAMPLE))
    assert [r.level for r in logview.filter_records(records, min_level="WARNING")] == ["WARNING", "ERROR"]
    assert len(list(logview.filter_records(records, job="collect_news"))) == 1
    assert len(list(logview.filter_records(records, grep="BOOM"))) == 1  # case-insensitive, matches inside the traceback
    now = datetime(2026, 11, 2, 3, 40, tzinfo=timezone.utc)  # 09:10 IST
    recent = list(logview.filter_records(records, since=logview.parse_since("15m", now)))
    assert [r.level for r in recent] == ["INFO"] and "12 new items" in recent[0].text
    orphan = list(logview.group_records(["  File \"x.py\", line 9", "KeyError: 'a'", *SAMPLE[:1]]))
    assert len(orphan) == 2 and [r.level for r in logview.filter_records(orphan, min_level="DEBUG")] == ["INFO"]


def test_since_accepts_durations_and_iso_text():
    now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
    assert logview.parse_since("90m", now) == now - timedelta(minutes=90)
    assert logview.parse_since("2h", now) == now - timedelta(hours=2)
    assert logview.parse_since("1d", now) == now - timedelta(days=1)
    assert logview.parse_since("2026-11-01", now) == datetime(2026, 11, 1, tzinfo=timezone.utc)
    assert logview.parse_since("2026-11-01T10:00:00+05:30", now) == datetime(2026, 11, 1, 4, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        logview.parse_since("yesterday-ish", now)


def test_tail_reads_only_the_end_of_a_big_file(tmp_path):
    path = tmp_path / "jobs-2026-11-02.log"
    path.write_text("\n".join(f"line {i}" for i in range(20_000)) + "\n", encoding="utf-8")
    assert logview.tail_lines(path, 3) == ["line 19997", "line 19998", "line 19999"]
    assert len(logview.tail_lines(path, 50_000)) == 20_000
    assert logview.tail_lines(path, 0) == []


def test_read_records_merges_files_by_time_and_bounds_memory(tmp_path):
    put(tmp_path, "jobs-2026-11-02.log", SAMPLE[1:6])  # a warning and an error with its traceback
    put(tmp_path, "daemon-2026-11-02.log", [SAMPLE[0], SAMPLE[6]])
    files = logview.list_log_files(tmp_path)
    merged = logview.read_records(files, limit=10)
    assert [r.stamp.strftime("%H:%M:%S") for r in merged] == ["08:30:00", "08:45:01", "08:45:02", "09:00:00"]
    assert merged[2].lines[-1] == "ValueError: boom"  # the traceback stays with its record
    assert len(logview.read_records(files, limit=1)) == 1 and logview.read_records(files, limit=1)[0].level == "INFO"
    big = tmp_path / "jobs-2026-11-03.log"
    big.write_text("\n".join(f"2026-11-03T08:00:{i % 60:02d}.000+0530 INFO    [-] x: n{i}" for i in range(3000)) + "\n")
    newest = logview.read_records([logview.LogFile(big, "jobs", "2026-11-03", 0, 0)], limit=5)
    assert [r.lines[0].rsplit("n", 1)[1] for r in newest] == ["2995", "2996", "2997", "2998", "2999"]


def test_an_orphaned_traceback_sorts_after_the_record_before_it(tmp_path):
    path = put(tmp_path, "jobs-2026-11-02.log", ["  File \"x.py\", line 9", "KeyError: 'a'", *SAMPLE[:2]])
    records = logview.read_records([logview.LogFile(path, "jobs", "2026-11-02", 0, 0)])
    assert [r.level for r in records] == ["", "INFO", "WARNING"]


def test_follow_prints_only_new_lines_and_picks_up_new_files(tmp_path):
    first = put(tmp_path, "jobs-2026-11-02.log", ["old line"])
    seen: list[str] = []
    step = {"n": 0}

    def appender(_seconds: float) -> None:
        step["n"] += 1
        if step["n"] == 1:
            with open(first, "a", encoding="utf-8") as handle:
                handle.write("new line 1\nnew line 2\n")
        elif step["n"] == 2:
            put(tmp_path, "daemon-2026-11-02.log", ["from a new file"])

    logview.follow(lambda: logview.list_log_files(tmp_path), seen.append, sleep=appender, max_polls=4)
    assert seen == ["new line 1", "new line 2", "from a new file"]  # "old line" was already there


# ----------------------------------------------------------------------------------------- the CLI
@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTER_ENV", "test")
    monkeypatch.setenv("MONSTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MONSTER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MONSTER_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    return tmp_path


def test_cli_logs_defaults_to_the_errors_of_the_newest_day(cli_env, capsys):
    assert cli.main(["logs"]) == 0
    assert "nothing at WARNING or above has been logged yet" in capsys.readouterr().out
    put(cli_env / "logs", "errors-2026-11-02.log", ERRORS)
    put(cli_env / "logs", "jobs-2026-11-02.log", SAMPLE)
    assert cli.main(["logs"]) == 0
    out = capsys.readouterr().out
    assert "job final_scan crashed" in out and "ValueError: boom" in out and "12 new items" not in out
    assert cli.main(["logs", "--role", "jobs", "--level", "warning", "--grep", "429"]) == 0
    out = capsys.readouterr().out
    assert "retrying in 4.0 s" in out and "crashed" not in out
    assert cli.main(["logs", "--role", "all", "-n", "2"]) == 0
    assert len([line for line in capsys.readouterr().out.splitlines() if line.startswith("2026-")]) <= 2


def test_cli_logs_list_date_job_and_empty_cases(cli_env, capsys, fake_clock):
    put(cli_env / "logs", "jobs-2026-11-02.log", SAMPLE)
    put(cli_env / "logs", "jobs-2026-11-01.log", SAMPLE[:1])
    assert cli.main(["logs", "--list"]) == 0
    assert capsys.readouterr().out.count(".log") == 2
    assert cli.main(["logs", "--role", "jobs", "--date", "2026-11-01"]) == 0 and "start daemon" in capsys.readouterr().out
    assert cli.main(["logs", "--role", "jobs", "--job", "collect_news"]) == 0
    out = capsys.readouterr().out
    assert "12 new items" in out and "crashed" not in out
    assert cli.main(["logs", "--role", "jobs", "--date", "2020-01-01"]) == 0
    assert "no 'jobs' log files for 2020-01-01" in capsys.readouterr().out
    fake_clock.set(datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc))  # 17:30 IST: the sample ended at 09:00 IST
    assert cli.main(["logs", "--role", "jobs", "--since", "5m"]) == 0
    assert capsys.readouterr().out.strip() == ""
    assert cli.main(["logs", "--role", "jobs", "--since", "9h"]) == 0
    assert "12 new items" in capsys.readouterr().out


def test_cli_logs_follow_stops_cleanly_on_interrupt(cli_env, capsys, monkeypatch):
    put(cli_env / "logs", "jobs-2026-11-02.log", SAMPLE[:1])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(logview, "follow", interrupted)
    assert cli.main(["logs", "--role", "jobs", "--follow"]) == 0
    assert "start daemon" in capsys.readouterr().out


def test_follow_line_filter_keeps_tracebacks_with_their_header():
    keep = cli._line_filter("ERROR", None)
    verdicts = [keep(line) for line in SAMPLE]
    assert verdicts == [False, False, True, True, True, True, False]
    grep = cli._line_filter(None, "nse_api")
    assert [grep(line) for line in SAMPLE[:3]] == [False, True, False]


def test_diag_bundle_is_complete_small_and_free_of_secrets(cli_env, capsys, fake_clock):
    assert cli.main(["init"]) == 0
    put(cli_env / "logs", "errors-2026-11-02.log", [*ERRORS, f"leaked https://api.telegram.org/bot{TOKEN}/sendMessage"])
    put(cli_env / "logs", "jobs-2026-11-02.log", SAMPLE)
    capsys.readouterr()
    assert cli.main(["diag"]) == 0
    out = capsys.readouterr().out.splitlines()
    bundle = Path(out[-2])
    assert bundle.exists() and bundle.name.startswith("diag-") and bundle.suffix == ".gz" and bundle.parent.name == "diag"
    with tarfile.open(bundle) as archive:
        names = sorted(m.name.split("/", 1)[1] for m in archive.getmembers())
        texts = {m.name.split("/", 1)[1]: archive.extractfile(m).read().decode("utf-8") for m in archive.getmembers()}
    assert {"summary.txt", "doctor.txt", "settings.json", "health.json", "storage.txt", "database.txt"} <= set(names)
    assert any(n.startswith("logs/errors-") for n in names) and any(n.startswith("logs/jobs-") for n in names)
    assert "monster-ai" in texts["summary.txt"] and "python" in texts["summary.txt"] and "+0530" in texts["summary.txt"]
    assert json.loads(texts["settings.json"])["telegram_bot_token"] == "***set***"
    assert json.loads(texts["health.json"])["level"] in {"ok", "warn", "crit"}
    assert "recent job runs" in texts["database.txt"] and "Retention windows" in texts["storage.txt"]
    assert all(TOKEN not in text for text in texts.values()), "the bot token leaked into the bundle"
    assert "bot<redacted>" in texts[next(n for n in names if n.startswith("logs/errors-"))]
    assert bundle.stat().st_size < 400_000


def test_diag_bundles_are_pruned_to_the_newest_few(tmp_path):
    from automation.retention import DIAG_KEEP, prune_diag_bundles

    folder = tmp_path / "diag"
    folder.mkdir()
    for i in range(DIAG_KEEP + 3):
        path = folder / f"diag-2026110{i}T000000Z.tar.gz"
        path.write_bytes(b"x")
        moment = time.time() - (DIAG_KEEP + 3 - i) * 3600
        os.utime(path, (moment, moment))
    assert prune_diag_bundles(folder, DIAG_KEEP, dry_run=True) == 3 and len(list(folder.iterdir())) == DIAG_KEEP + 3
    assert prune_diag_bundles(folder, DIAG_KEEP) == 3
    assert sorted(p.name for p in folder.iterdir())[0] == "diag-20261103T000000Z.tar.gz"  # the oldest three are gone
    assert prune_diag_bundles(tmp_path / "missing", DIAG_KEEP) == 0
