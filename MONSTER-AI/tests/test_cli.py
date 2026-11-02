"""Command line interface: argument plumbing, exit codes and safe behaviour of the operator commands."""

from __future__ import annotations

import json
import os
import re
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
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GROQ_API_KEY", "TYPESAFE_API_KEY", "LLM_API_BASE", "LLM_MODEL", "LLM_API_KEY"):
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


def test_doctor_reports_the_ai_services_as_optional_and_names_the_ones_that_are_on(cli_env, monkeypatch):
    import app.doctor as doctor
    from app.config import load_settings

    report = doctor.DoctorReport()
    doctor._ai_config(report, load_settings())
    (row,) = report.checks
    assert row.name == "ai services" and row.status == "ok" and row.detail.startswith("off (optional)")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_groq_key_1234567890")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test_typesafe_key_1234567890")
    report = doctor.DoctorReport()
    doctor._ai_config(report, load_settings())
    detail = report.checks[0].detail
    assert "LLM via groq (openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b)" in detail and "Jev (jev-1.14.0)" in detail
    assert "gsk_test" not in detail and "ts_test" not in detail


def test_doctor_proves_the_nse_api_and_warns_with_the_reason_when_an_endpoint_is_gone(cli_env, monkeypatch):
    from types import SimpleNamespace

    import app.doctor as doctor
    import ingestion.nse as nse_mod
    from app.config import load_settings
    from app.errors import DataUnavailable

    closed = []

    class FakeNse:
        status: object = {"marketState": [], "giftnifty": {}}
        live: object = SimpleNamespace(rows=[1, 2, 3], source_ts="2026-11-02T10:10:12Z")

        def __init__(self, settings):
            self.http = SimpleNamespace(close=lambda: closed.append(True))

        @staticmethod
        def _answer(value):
            if isinstance(value, Exception):
                raise value
            return value

        def market_status(self):
            return self._answer(FakeNse.status)

        def live_equities(self, index):
            assert index == "SECURITIES IN F&O"
            return self._answer(FakeNse.live)

    def run():
        report = doctor.DoctorReport()
        doctor._nse_api(report, load_settings())
        return report

    monkeypatch.setattr(nse_mod, "NseApi", FakeNse)
    report = run()
    assert [(c.name, c.status) for c in report.checks] == [("network: NSE API", "ok"), ("network: NSE live quotes", "ok")]
    assert "3 F&O stocks" in report.checks[1].detail and closed == [True]

    FakeNse.live = DataUnavailable("nse_api", "no live-quote endpoint knows SECURITIES IN F&O: did NSE change its API?")
    report = run()
    assert [c.status for c in report.checks] == ["ok", "warn"] and "did NSE change its API" in report.checks[1].detail

    FakeNse.live = SimpleNamespace(rows=[], source_ts=None)
    assert run().checks[1].status == "warn"

    FakeNse.status = DataUnavailable("nse_api", "HTTP 403")
    report = run()
    assert [c.status for c in report.checks] == ["warn", "warn"] and not report.failed  # one failed probe never hides the other

    monkeypatch.setenv("NSE_ENABLED", "false")
    assert run().checks == []


def test_doctor_probes_only_configured_ai_services_for_free_and_never_fails_the_machine_for_them(cli_env, monkeypatch):
    import app.doctor as doctor
    from app.config import load_settings
    from news_intelligence import ai_check
    from news_intelligence.ai_check import AiCheck

    seen = []

    def fake(settings, **kwargs):
        seen.append(kwargs)
        return [AiCheck("llm", "Groq LLM", "fail", "HTTP 401: the key was rejected, check GROQ_API_KEY"),
                AiCheck("jev", "Jev (TypeSafe AI)", "off", "not configured")]

    monkeypatch.setattr(ai_check, "run_checks", fake)
    report = doctor.DoctorReport()
    doctor._ai_network(report, load_settings())
    assert seen == [{"deep": False}]  # the doctor spends no tokens
    (row,) = report.checks  # the service that is off is not listed
    assert row.name == "ai: Groq LLM" and row.status == "warn" and not report.failed

    def boom(settings, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ai_check, "run_checks", boom)
    report = doctor.DoctorReport()
    doctor._ai_network(report, load_settings())
    assert report.checks[0].name == "ai check" and report.checks[0].status == "warn" and not report.failed


def test_doctor_names_the_reason_when_the_bot_check_fails(cli_env, monkeypatch):
    from types import SimpleNamespace

    import app.doctor as doctor
    import ingestion.http as http_mod
    import notifications.telegram as tg
    from app.config import load_settings
    from app.errors import DataUnavailable

    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"

    class FakeHttp:
        @classmethod
        def from_settings(cls, *args, **kwargs):
            return cls()

        def get(self, url, **kwargs):
            return SimpleNamespace(status=200, headers={})

        def close(self):
            pass

    class BlockedBot:
        def __init__(self, settings):
            pass

        def get_me(self):
            raise DataUnavailable("telegram", f"HTTP 403: response was not Telegram JSON (bot{token})", retryable=True)

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(http_mod, "HttpClient", FakeHttp)
    monkeypatch.setattr(tg, "TelegramClient", BlockedBot)
    report = doctor.DoctorReport()
    doctor._network(report, load_settings())
    bot = next(check for check in report.checks if check.name == "telegram bot")
    assert bot.status == "fail" and "response was not Telegram JSON" in bot.detail
    assert token not in bot.detail


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
    for name in ("collect_global", "prelim_scan", "final_scan", "closing_scan", "ingest_eod", "evaluate_ideas", "backup", "train"):
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


def test_ideas_dry_run_prints_the_report_and_freezes_nothing(cli_env, monkeypatch, capsys):
    from automation import ideas
    from automation.context import JobResult
    from ranking.types import IdeasReport, TradeIdea

    idea = TradeIdea(symbol="ABC", rank=1, horizon="BTST", side="LONG", setup="strong close on volume", score=71.0, grade="B",
                     ltp=100.0, entry=100.0, stop=97.0, target1=104.0)
    report = IdeasReport(trade_date="2026-11-02", snapshot_ist="2026-11-02 14:31:00 IST", entry_by="15:15", ideas=[idea]).to_dict()
    seen = {}
    monkeypatch.setattr(ideas, "run_closing_scan", lambda ctx: seen.update(dry=ctx.dry_run, key=ctx.run_key)
                        or JobResult("success", "dry run", {"report": report}))
    assert run(["ideas", "--date", "2026-11-02", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ABC" in out and "Closing-hour trade ideas" in out and "closing scan 2026-11-02: success" in out
    assert seen == {"dry": True, "key": "2026-11-02"}


def test_ideas_runs_the_scheduler_job_and_reports_its_outcome(cli_env, monkeypatch, capsys):
    from automation import jobs
    from automation.context import JobResult

    seen = {}
    monkeypatch.setitem(jobs.FUNCTIONS, "closing_scan", lambda ctx: seen.update(force=ctx.force, key=ctx.run_key)
                        or JobResult("success", "report sent"))
    assert run(["ideas", "--date", "2026-11-02", "--force"]) == 0
    assert seen == {"force": True, "key": "2026-11-02"} and "closing scan 2026-11-02: success - report sent" in capsys.readouterr().out
    assert run(["ideas", "--date", "2026-11-02"]) == 0 and seen["force"] is False
    monkeypatch.setitem(jobs.FUNCTIONS, "closing_scan", lambda ctx: JobResult("failed", "NSE live quotes unavailable"))
    assert run(["ideas", "--date", "2026-11-02"]) == 1 and "failed - NSE live quotes unavailable" in capsys.readouterr().out


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


def _failing_client(monkeypatch, *, on):
    """A Telegram client whose ``get_me`` or ``discover_chat_id`` fails with a reason that even embeds the token."""
    import notifications.telegram as tg
    from app.errors import DataUnavailable

    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    reasons = {
        "get_me": f"[telegram] HTTP 401: Unauthorized (check TELEGRAM_BOT_TOKEN) at /bot{token}/getMe",
        "discover": "[telegram] HTTP 409: Conflict: can't use getUpdates method while webhook is active",
    }

    class Failing:
        def __init__(self, *a, **k):
            pass

        def get_me(self):
            if on == "get_me":
                raise DataUnavailable("telegram", reasons["get_me"], retryable=False)
            return {"username": "monster_test_bot"}

        def discover_chat_id(self, **kw):
            raise DataUnavailable("telegram", reasons["discover"], retryable=False)

    monkeypatch.setattr(tg, "TelegramClient", Failing)
    return token


def test_telegram_setup_says_why_the_token_check_failed_without_leaking_it(cli_env, monkeypatch, capsys):
    token = _failing_client(monkeypatch, on="get_me")
    assert run(["telegram-setup", "--token", token, "--wait", "1"]) == 1
    captured = capsys.readouterr()
    assert "HTTP 401: Unauthorized (check TELEGRAM_BOT_TOKEN)" in captured.err
    assert token not in captured.err + captured.out
    assert captured.out == ""  # the installer parses stdout: nothing may be printed there on failure


def test_telegram_setup_reports_a_failure_while_waiting_for_the_message(cli_env, monkeypatch, capsys):
    token = _failing_client(monkeypatch, on="discover")
    assert run(["telegram-setup", "--token", token, "--wait", "1"]) == 1
    captured = capsys.readouterr()
    assert "while waiting for your message" in captured.err and "webhook is active" in captured.err
    assert "TELEGRAM_CHAT_ID=" not in captured.out and "Traceback" not in captured.err


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
    assert sent["text"].startswith("🚨 <b>Service failed</b> · <code>monster-ai.servicerm-rf</code>")  # the sanitised name
    assert "📜 <b>Last journal lines</b>" in sent["text"] and " IST</i>" in sent["text"]


def _preview_env(monkeypatch, *, alert_chat="77"):
    """Telegram settings plus a fake client that records what the preview sends: ``sent`` the texts, ``sent.photos`` the images."""
    import notifications.telegram as tg

    class Sent(list):
        photos: list

    sent = Sent()
    sent.photos = []

    class RecordingClient:
        configured = True

        def __init__(self, settings, **kwargs):
            pass

        def send_message(self, chat_id, text, **kwargs):
            sent.append((chat_id, text))
            return [len(sent)]

        def send_photo(self, chat_id, photo, *, caption=None, filename="chart.png", **kwargs):
            sent.photos.append((chat_id, photo, caption, filename))
            return len(sent.photos)

    monkeypatch.setattr(tg, "TelegramClient", RecordingClient)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("TELEGRAM_ALERT_CHAT_ID", alert_chat)
    return sent


def test_telegram_preview_lists_the_kinds_without_needing_telegram(cli_env, capsys):
    assert run(["telegram-preview", "--list"]) == 0
    out = capsys.readouterr().out
    for kind in ("final", "prelim", "no-setup", "ideas", "evening", "alert", "digest", "failure", "unit", "online"):
        assert any(line.startswith(kind) for line in out.splitlines()), kind


def test_telegram_preview_needs_a_configured_bot_and_a_known_kind(cli_env, capsys):
    assert run(["telegram-preview"]) == 1 and "not configured" in capsys.readouterr().err
    assert run(["telegram-preview", "--only", "bogus"]) == 2 and "unknown kind(s) bogus" in capsys.readouterr().err


def test_telegram_preview_sends_marked_samples_and_routes_alerts_to_the_alert_chat(cli_env, monkeypatch, capsys):
    sent = _preview_env(monkeypatch)
    assert run(["telegram-preview", "--delay", "0", "--only", "final", "alert", "unit"]) == 0
    assert all(text.startswith("🧪 <b>SAMPLE PREVIEW</b>") for _, text in sent)
    chats = [chat for chat, _ in sent]
    assert chats[0] == "42" and set(chats[1:]) == {"77"}  # the report to the report chat, 4 alerts + the unit failure to the alert chat
    assert len(sent) >= 6 and "made-up data" in capsys.readouterr().out
    ((chat, png, caption, filename),) = sent.photos  # the final report's chart: to the report chat, marked as a sample
    assert chat == "42" and png.startswith(b"\x89PNG") and caption.startswith("🧪 <b>SAMPLE PREVIEW</b>") and filename == "candidates.png"


def test_telegram_preview_sends_everything_by_default(cli_env, monkeypatch, capsys):
    from notifications import samples

    sent = _preview_env(monkeypatch)
    assert run(["telegram-preview", "--delay", "0"]) == 0
    assert len(sent) >= len(samples.KINDS)
    assert [name for _, _, _, name in sent.photos] == ["candidates.png", "candidates.png", "ideas-board.png", "ideas-plan.png", "evening.png"]
    assert all(photo.startswith(b"\x89PNG") and chat == "42" for chat, photo, _, _ in sent.photos)
    assert "with 2 chart(s)" in capsys.readouterr().out


def test_telegram_preview_sends_no_images_when_charts_are_off(cli_env, monkeypatch):
    sent = _preview_env(monkeypatch)
    monkeypatch.setenv("SEND_CHARTS", "false")
    assert run(["telegram-preview", "--delay", "0", "--only", "ideas"]) == 0
    assert sent and sent.photos == []


def test_telegram_preview_reports_the_reason_when_telegram_fails_midway(cli_env, monkeypatch, capsys):
    from app.errors import DataUnavailable

    sent = _preview_env(monkeypatch)
    import notifications.telegram as tg

    real = tg.TelegramClient

    class Flaky(real):
        def send_message(self, chat_id, text, **kwargs):
            if len(sent) >= 2:
                raise DataUnavailable("telegram", "HTTP 429: rate limited by Telegram", retryable=False)
            return super().send_message(chat_id, text, **kwargs)

    monkeypatch.setattr(tg, "TelegramClient", Flaky)
    assert run(["telegram-preview", "--delay", "0"]) == 1
    err = capsys.readouterr().err
    assert re.search(r"after [1-9]\d* message\(s\): HTTP 429: rate limited by Telegram", err)


def _ai_checks(monkeypatch, *checks):
    """``run_checks`` answers with ``checks`` (filtered like the real one) and records how the command called it."""
    from news_intelligence import ai_check

    calls = []

    def fake(settings, **kwargs):
        calls.append(kwargs)
        return [c for c in checks if kwargs.get("only") is None or c.service in kwargs["only"]]

    monkeypatch.setattr(ai_check, "run_checks", fake)
    return calls


def test_ai_test_without_any_key_says_what_to_add_and_fails(cli_env, capsys):
    assert run(["ai-test"]) == 1
    captured = capsys.readouterr()
    assert captured.out.count("OFF") == 3  # the LLM, its web check and Jev
    assert "GROQ_API_KEY" in captured.err and "TYPESAFE_API_KEY" in captured.err


def test_ai_test_prints_a_line_per_service_and_succeeds_when_every_configured_one_answers(cli_env, monkeypatch, capsys):
    from news_intelligence.ai_check import AiCheck

    calls = _ai_checks(monkeypatch, AiCheck("llm", "Groq LLM", "ok", "model m: order_win", 1.5),
                       AiCheck("jev", "Jev (TypeSafe AI)", "off", "not configured"))
    assert run(["ai-test"]) == 0
    out = capsys.readouterr().out
    assert "Groq LLM" in out and "OK" in out and "1.5s" in out and "OFF" in out
    assert calls[0]["deep"] is True and calls[0]["resolver"] is not None  # the real database gives the LLM's companies a meaning


def test_ai_test_fails_when_a_configured_service_does_not_answer_and_a_warning_still_passes(cli_env, monkeypatch, capsys):
    from news_intelligence.ai_check import AiCheck

    _ai_checks(monkeypatch, AiCheck("llm", "Groq LLM", "warn", "model m: order_win; not listed for this key: x", 1.0),
               AiCheck("jev", "Jev (TypeSafe AI)", "fail", "HTTP 401: the key was rejected", 0.2))
    assert run(["ai-test"]) == 1 and "FAIL" in capsys.readouterr().out
    assert run(["ai-test", "--service", "llm"]) == 0  # only the warning remains


def test_ai_test_options_json_service_and_list_only(cli_env, monkeypatch, capsys):
    from news_intelligence.ai_check import AiCheck

    calls = _ai_checks(monkeypatch, AiCheck("llm", "Groq LLM", "ok", "key accepted"),
                       AiCheck("web", "Groq web check", "ok", "enabled"), AiCheck("jev", "Jev (TypeSafe AI)", "ok", "key accepted"))
    assert run(["ai-test", "--json", "--service", "jev", "--list-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["service"] for item in payload] == ["jev"] and payload[0]["status"] == "ok"
    assert calls[0]["only"] == ["jev"] and calls[0]["deep"] is False
    assert run(["ai-test", "--json", "--service", "web"]) == 0
    assert [item["service"] for item in json.loads(capsys.readouterr().out)] == ["web"] and calls[1]["only"] == ["web"]


def test_ai_test_still_works_when_the_database_cannot_be_opened(cli_env, monkeypatch, capsys):
    from news_intelligence.ai_check import AiCheck

    calls = _ai_checks(monkeypatch, AiCheck("llm", "Groq LLM", "ok", "fine"))
    monkeypatch.setattr(cli, "_open_store", lambda settings: (_ for _ in ()).throw(OSError("disk is read-only")))
    assert run(["ai-test"]) == 0 and calls[0]["resolver"] is None


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
