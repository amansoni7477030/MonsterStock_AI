"""Command line interface: argument plumbing, exit codes and safe behaviour of the operator commands."""

from __future__ import annotations

import json
import os
import stat

import pytest

from app import cli
from app.locks import FileLock, lock_dir
from automation import runner


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTER_ENV", "test")
    monkeypatch.setenv("MONSTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MONSTER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MONSTER_ENV_FILE", str(tmp_path / "none.env"))
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "_install_sigterm", lambda: None)
    return tmp_path


def run(argv):
    return cli.main(argv)


def test_version_and_env_example(cli_env, capsys):
    assert run(["version"]) == 0 and "monster-ai" in capsys.readouterr().out
    assert run(["env-example"]) == 0
    out = capsys.readouterr().out
    assert "TELEGRAM_BOT_TOKEN=" in out and "MONSTER_MODE=shadow" in out


def test_init_creates_database_and_calendar(cli_env, capsys):
    import re

    assert run(["init"]) == 0
    out = capsys.readouterr().out
    seeded = re.search(r"calendar seeded: (\d+) rows", out)
    assert "database ready" in out and seeded and int(seeded.group(1)) >= 20  # the 2026 list + rule-based years ahead
    assert (cli_env / "data" / "monster.db").exists()


def test_doctor_offline_reports_ready_in_a_clean_environment(cli_env, capsys):
    assert run(["doctor", "--offline", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    names = {c["name"] for c in payload["checks"]}
    assert {"python", "package numpy", "configuration", "database", "seed data"} <= names
    assert run(["doctor", "--offline"]) == 0 and "READY" in capsys.readouterr().out


def test_doctor_fails_on_invalid_configuration(cli_env, monkeypatch, capsys):
    monkeypatch.setenv("TOP_N", "99")
    assert run(["doctor", "--offline"]) == 1
    assert "TOP_N" in capsys.readouterr().out


def test_main_reports_configuration_errors_with_exit_2(cli_env, monkeypatch, capsys):
    monkeypatch.setenv("MONSTER_MODE", "turbo")
    assert run(["init"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_jobs_lists_the_timetable(cli_env, capsys):
    assert run(["jobs"]) == 0
    out = capsys.readouterr().out
    for name in ("collect_global", "prelim_scan", "final_scan", "ingest_eod", "backup", "train"):
        assert name in out


def test_health_exit_codes_and_json(cli_env, capsys):
    assert run(["health", "--json"]) == 1  # fresh install: warnings only
    payload = json.loads(capsys.readouterr().out)
    assert payload["level"] == "warn" and any(p["code"] == "telegram_not_configured" for p in payload["problems"])
    assert run(["health"]) == 1 and "health: WARN" in capsys.readouterr().out


def test_run_job_validates_names_and_runs_known_jobs(cli_env, capsys):
    assert run(["run-job", "nope"]) == 2
    assert run(["run-job", "health_check", "--key", "2026-11-02"]) == 0
    assert "health_check [2026-11-02]: success" in capsys.readouterr().out
    assert run(["run-job", "health_check", "--opt", "broken"]) == 2


def test_models_commands(cli_env, capsys):
    assert run(["models", "list"]) == 0 and "no trained models" in capsys.readouterr().out
    assert run(["models", "approve"]) == 2  # needs a version
    assert run(["models", "approve", "ghost"]) == 1
    assert run(["models", "rollback", "--stage", "pre"]) == 1  # nothing to roll back


def test_backup_verify_and_restore_guard(cli_env, capsys):
    assert run(["init"]) == 0
    assert run(["backup", "--verify"]) == 1  # no backup yet
    assert run(["backup"]) == 0 and "backup: success" in capsys.readouterr().out
    assert run(["backup", "--verify"]) == 0 and "verified" in capsys.readouterr().out
    held = FileLock(lock_dir(cli_env / "data") / "daemon.lock")
    assert held.acquire()
    try:
        assert run(["backup", "--restore-latest"]) == 3  # refuses while the scheduler runs
    finally:
        held.release()
    assert run(["backup", "--restore-latest"]) == 0


def test_scan_dry_run_prints_the_report_and_freezes_nothing(cli_env, monkeypatch, capsys):
    from automation import scan
    from automation.context import JobResult
    from ranking.types import Candidate, ReportContext

    report = ReportContext(trade_date="2026-11-02", stage="pre", snapshot_ist="2026-11-02 08:45:00 IST", model_version="baseline-pre",
                           model_status="baseline", candidates=[Candidate(symbol="ABC", rank=1, tier="WATCH", p={"hit_3": 0.3},
                                                                           primary_threshold=3.0, range={"q50": 3.1})]).to_dict()
    monkeypatch.setattr(scan, "run_scan", lambda ctx, stage: JobResult("success", "dry run", {"report": report}))
    assert run(["scan", "--stage", "pre", "--date", "2026-11-02", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ABC" in out and "pre scan 2026-11-02: success" in out


def test_telegram_setup_discovers_and_writes_the_chat_id(cli_env, monkeypatch, capsys):
    import notifications.telegram as tg

    class FakeClient:
        def __init__(self, settings, **kw):
            self.settings = settings

        def get_me(self):
            return {"username": "monster_test_bot"}

        def discover_chat_id(self, wait_seconds=0, **kw):
            return {"chat_id": "987654", "title": "Ops", "type": "private"}

    monkeypatch.setattr(tg, "TelegramClient", FakeClient)
    assert run(["telegram-setup"]) == 1  # no token anywhere
    env_file = cli_env / "monster.env"
    env_file.write_text("# keep me\nMONSTER_MODE=shadow\nTELEGRAM_CHAT_ID=old\n", encoding="utf-8")
    os.chmod(env_file, 0o640)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    assert run(["telegram-setup", "--token", token, "--wait", "1", "--write-env", str(env_file)]) == 0
    out = capsys.readouterr().out
    assert "TELEGRAM_CHAT_ID=987654" in out and token not in out
    text = env_file.read_text(encoding="utf-8")
    assert "# keep me" in text and "TELEGRAM_CHAT_ID=987654" in text and "TELEGRAM_CHAT_ID=old" not in text
    assert f"TELEGRAM_BOT_TOKEN={token}" in text
    if os.name != "nt":
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o640


def test_telegram_setup_print_only_and_no_message_yet(cli_env, monkeypatch):
    import notifications.telegram as tg

    class Silent:
        def __init__(self, *a, **k):
            pass

        def get_me(self):
            return {"username": "b"}

        def discover_chat_id(self, **kw):
            return None

    monkeypatch.setattr(tg, "TelegramClient", Silent)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    assert run(["telegram-setup", "--token", token, "--wait", "1"]) == 1


def test_alert_unit_sanitises_the_unit_name_and_redacts_logs(cli_env, monkeypatch, capsys):
    import notifications.service as svc

    sent = {}

    def fake_send(self, kind, key, text, chat_id=None, trade_date=None):
        sent.update(kind=kind, text=text, chat=chat_id)
        return svc.DeliveryResult("sent")

    monkeypatch.setattr(svc.NotificationService, "send_text", fake_send)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert run(["alert-unit", "monster-ai.service; rm -rf /"]) == 0
    assert sent["kind"] == "unit_failure" and "monster-ai.service" in sent["text"] and "rm -rf" not in sent["text"]
    assert sent["chat"] == "42"


def test_evaluate_and_train_plumbing(cli_env, monkeypatch, capsys):
    from automation import jobs
    from automation.context import JobResult

    seen = {}
    monkeypatch.setitem(jobs.FUNCTIONS, "evaluate", lambda ctx: JobResult("success", f"evaluated {ctx.run_key}"))
    monkeypatch.setitem(jobs.FUNCTIONS, "train", lambda ctx: seen.setdefault("opts", dict(ctx.options)) and JobResult("success", "ok"))
    assert run(["evaluate", "--date", "2026-11-02"]) == 0 and "evaluated 2026-11-02" in capsys.readouterr().out
    assert run(["train", "--stage", "pre"]) == 0
    assert seen["opts"] == {"stages": "pre"}


def test_bootstrap_plumbing_passes_skip_train(cli_env, monkeypatch, capsys):
    from automation import jobs
    from automation.context import JobResult

    seen = {}
    monkeypatch.setitem(jobs.FUNCTIONS, "bootstrap", lambda ctx: seen.update(opts=dict(ctx.options), force=ctx.force)
                        or JobResult("success", "done"))
    assert run(["bootstrap", "--resume", "--skip-train"]) == 0
    assert seen == {"opts": {"skip_train": "1"}, "force": False}
    assert run(["bootstrap"]) == 0 and seen["force"] is True  # without --resume the run starts over


def test_storage_report_dry_run_prune_and_busy_housekeeping(cli_env, capsys):
    from app.locks import job_lock

    assert run(["init"]) == 0
    capsys.readouterr()
    assert run(["storage"]) == 0
    out = capsys.readouterr().out
    assert "Retention windows" in out and "bars_daily" in out and "Last retention run: never" in out

    assert run(["storage", "--prune", "--dry-run"]) == 0
    assert "(dry run)" in capsys.readouterr().out
    assert run(["storage", "--json"]) == 0 and json.loads(capsys.readouterr().out)["last_run"] is None  # a dry run records nothing

    assert run(["storage", "--prune", "--vacuum"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("retention ") and "maintenance:" in out and "Last retention run: 20" in out
    assert run(["storage", "--json"]) == 0 and json.loads(capsys.readouterr().out)["last_run"]["pressure"]

    held = job_lock(cli_env / "data", "housekeeping")
    assert held.acquire()
    try:
        assert run(["storage", "--prune"]) == 3  # the nightly job is running: do not collide with it
    finally:
        held.release()
