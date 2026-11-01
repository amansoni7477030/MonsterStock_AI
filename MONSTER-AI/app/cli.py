"""Command line interface: ``python -m app.cli <command>``.

Exit codes: 0 success, 1 failure, 2 usage / health critical, 3 another instance is running.
Run as the service user (``monster``) on the server, never as root.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import platform
import shutil
import subprocess  # noqa: S404 - only journalctl with a fixed argument vector
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from app import __version__
from app.clock import get_clock, ist_date, pin_process_timezone
from app.config import ConfigError, Settings, env_example_text, load_settings
from app.logging_setup import configure_from_settings, redact, register_secrets
from app.release import release_id

log = logging.getLogger("app.cli")

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_BUSY = 0, 1, 2, 3


# ---------------------------------------------------------------------------------------------- helpers
def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(env_file=args.env_file)
    register_secrets(settings.secrets())
    return settings


def _logging(settings: Settings, filename: str) -> None:
    configure_from_settings(settings, filename)
    log.info("start %s: monster-ai %s pid=%d python=%s on %s; env=%s mode=%s; logs in %s (display zone %s, server zone pinned "
             "to UTC and never used)", Path(filename).stem, release_id(), os.getpid(), platform.python_version(),
             platform.system(), settings.env, settings.mode, settings.log_dir, settings.timezone)


def _open_store(settings: Settings) -> Any:
    from database.store import Store

    settings.ensure_dirs()
    return Store.open(settings.db_path)


def _parse_day(value: str | None) -> date:
    return date.fromisoformat(value) if value else ist_date(get_clock().now())


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


def _why(exc: BaseException) -> str:
    """One redacted line saying why a call failed (the Telegram client's own reasons never carry the token)."""
    return " ".join(redact(str(exc)).split())[:200] or type(exc).__name__


# --------------------------------------------------------------------------------------------- commands
def cmd_version(_args: argparse.Namespace) -> int:
    print(f"monster-ai {__version__}")  # the installer reads only this first line
    print(f"release {release_id()}")
    return EXIT_OK


def cmd_env_example(_args: argparse.Namespace) -> int:
    sys.stdout.write(env_example_text())
    return EXIT_OK


def cmd_init(args: argparse.Namespace) -> int:
    from automation.calendar import seed_calendar
    from ingestion.registry import register_all

    settings = _settings(args)
    _logging(settings, "cli.log")
    store = _open_store(settings)
    try:
        written = seed_calendar(store, settings)
        sources = register_all(store)
        print(f"database ready: {settings.db_path}")
        print(f"calendar seeded: {written} rows; {sources} data sources registered; models dir: {settings.models_dir}")
    finally:
        store.close()
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    from app.doctor import render, run_doctor

    report = run_doctor(offline=args.offline, env_file=args.env_file)
    if args.json:
        _print_json(report.to_dict())
    else:
        print(render(report))
    return EXIT_FAIL if report.failed else EXIT_OK


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from automation.runner import execute_job

    settings = _settings(args)
    _logging(settings, "bootstrap.log")
    settings.ensure_dirs()
    result = execute_job(settings, "bootstrap", run_key=ist_date(get_clock().now()).isoformat(), force=not args.resume,
                         install_signal=True, options={"skip_train": "1"} if args.skip_train else None)
    print(f"bootstrap: {result.status} - {result.message}")
    return EXIT_OK if result.status in {"success", "skipped"} else EXIT_FAIL  # skipped: another bootstrap holds the lock


def cmd_daemon(args: argparse.Namespace) -> int:
    from automation.scheduler import run_daemon

    settings = _settings(args)
    _logging(settings, "daemon.log")
    settings.ensure_dirs()
    return run_daemon(settings)


def cmd_run_job(args: argparse.Namespace) -> int:
    from automation.jobspec import JOBS
    from automation.runner import execute_job

    if args.job not in JOBS:
        print(f"unknown job {args.job!r}; choose from: {', '.join(sorted(JOBS))}", file=sys.stderr)
        return EXIT_USAGE
    settings = _settings(args)
    _logging(settings, "jobs.log")
    settings.ensure_dirs()
    try:
        options = dict(item.split("=", 1) for item in args.opt)
    except ValueError:
        print("--opt expects KEY=VALUE", file=sys.stderr)
        return EXIT_USAGE
    key = args.key or ist_date(get_clock().now()).isoformat()
    result = execute_job(settings, args.job, key, attempt=args.attempt, max_attempts=args.max_attempts,
                         last_attempt=args.last_attempt or args.attempt >= args.max_attempts, force=args.force,
                         dry_run=args.dry_run, timeout_min=args.timeout_min, install_signal=True, options=options)
    print(f"{args.job} [{key}]: {result.status} - {result.message}")
    return EXIT_OK if result.status in {"success", "skipped"} else EXIT_FAIL


def cmd_jobs(args: argparse.Namespace) -> int:
    from automation.jobspec import JOBS
    from automation.schedule import SCHEDULE

    settings = _settings(args)
    store = _open_store(settings)
    try:
        print(f"{'time (IST)':<18}{'job':<22}{'days':<9}{'tries':<6}description")
        for e in sorted(SCHEDULE, key=lambda x: x.at):
            window = f"{e.at[:5]}-{e.until[:5]}"
            when = "weekly" if e.weekly_dow is not None else e.days
            print(f"{window:<18}{e.label:<22}{when:<9}{e.max_attempts:<6}{JOBS[e.job].description}")
        print("\nrecent runs:")
        for row in store.recent_jobs(15):
            print(f"  {row['started_at'] or '-':<22}{row['job_name']:<18}{row['run_key']:<16}{row['status']:<9}{(row['message'] or '')[:70]}")
    finally:
        store.close()
    return EXIT_OK


def cmd_scan(args: argparse.Namespace) -> int:
    from automation import scan
    from automation.context import JobContext
    from automation.runner import execute_job
    from notifications.telegram import to_plain_text
    from ranking.types import ReportContext

    settings = _settings(args)
    _logging(settings, "cli.log")
    day = _parse_day(args.date).isoformat()
    if not args.dry_run:
        name = "prelim_scan" if args.stage == "pre" else "final_scan"
        result = execute_job(settings, name, day, force=args.force, install_signal=True)
    else:
        store = _open_store(settings)
        try:
            ctx = JobContext.create(settings, store, run_key=day, dry_run=True, force=args.force)
            result = scan.run_scan(ctx, args.stage)
        finally:
            store.close()
        report = result.metrics.get("report")
        if report:
            from notifications import report as renderer

            ctx_report = ReportContext.from_dict(report)
            text = renderer.render_final_report(ctx_report) if args.stage == "final" else renderer.render_prelim_report(ctx_report)
            print("\n\n".join(to_plain_text(part) for part in text))
    print(f"{args.stage} scan {day}: {result.status} - {result.message}")
    return EXIT_OK if result.status in {"success", "skipped"} else EXIT_FAIL


def cmd_evaluate(args: argparse.Namespace) -> int:
    from automation.runner import execute_job

    settings = _settings(args)
    _logging(settings, "cli.log")
    day = _parse_day(args.date).isoformat()
    result = execute_job(settings, "evaluate", day, force=True, install_signal=True)
    print(f"evaluate {day}: {result.status} - {result.message}")
    return EXIT_OK if result.status in {"success", "skipped"} else EXIT_FAIL


def cmd_train(args: argparse.Namespace) -> int:
    from automation.runner import execute_job

    settings = _settings(args)
    _logging(settings, "train.log")
    settings.ensure_dirs()
    if args.publish:
        from models.external import train_and_publish

        return train_and_publish(settings, stage=args.stage)
    result = execute_job(settings, "train", ist_date(get_clock().now()).isoformat(), force=True, install_signal=True,
                         timeout_min=240, options={"stages": "pre,final" if args.stage == "both" else args.stage})
    print(f"train: {result.status} - {result.message}")
    return EXIT_OK if result.status in {"success", "skipped"} else EXIT_FAIL


def cmd_backtest(args: argparse.Namespace) -> int:
    from backtesting.walkforward import run_backtest

    settings = _settings(args)
    _logging(settings, "cli.log")
    store = _open_store(settings)
    try:
        return run_backtest(settings, store, stages=("pre", "final") if args.stage == "both" else (args.stage,), years=args.years,
                            out_dir=Path(args.out) if args.out else None)
    finally:
        store.close()


def cmd_models(args: argparse.Namespace) -> int:
    from models.registry import ModelRegistry

    settings = _settings(args)
    store = _open_store(settings)
    try:
        registry = ModelRegistry(store, settings)
        if args.action == "list":
            rows = registry.summary()
            if not rows:
                print("no trained models yet (the heuristic baseline is in use)")
            for r in rows:
                print(f"{r['version']:<28}{r['stage']:<7}{r['status']:<10}{'ACTIVE' if r['active'] else '':<8}{r['created_at'] or ''}")
            return EXIT_OK
        if args.action == "approve":
            registry.approve(args.version, args.status)
        elif args.action == "activate":
            registry.activate(args.version)
        elif args.action == "rollback":
            previous = registry.rollback(args.stage)
            print(f"rolled back to {previous}" if previous else "no previous model to roll back to (baseline in use)")
            return EXIT_OK if previous else EXIT_FAIL
        print("ok")
        return EXIT_OK
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAIL
    finally:
        store.close()


def cmd_backup(args: argparse.Namespace) -> int:
    from app.locks import FileLock, lock_dir
    from automation.runner import execute_job
    from database.backup import list_backups, restore_latest_valid, verify_backup

    settings = _settings(args)
    _logging(settings, "cli.log")
    settings.ensure_dirs()
    if args.verify:
        backups = list_backups(settings.backup_dir)
        if not backups:
            print("no backups found")
            return EXIT_FAIL
        ok = verify_backup(backups[-1], settings.backup_passphrase)
        print(f"{backups[-1].name}: {'verified' if ok else 'FAILED verification'}")
        return EXIT_OK if ok else EXIT_FAIL
    if args.restore_latest:
        daemon = FileLock(lock_dir(settings.data_dir) / "daemon.lock")
        if not daemon.acquire(blocking=False):
            print("refusing to restore while the scheduler is running (systemctl stop monster-ai)", file=sys.stderr)
            return EXIT_BUSY
        try:
            if args.remote:
                from database.object_store import build_object_store
                from database.remote_backup import fetch_latest_backup

                remote = build_object_store(settings)
                if remote is None:
                    print("off-box storage is not configured (S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY)", file=sys.stderr)
                    return EXIT_FAIL
                fetched = fetch_latest_backup(remote, settings.backup_dir, settings.backup_passphrase)
                print(f"downloaded {fetched.name}" if fetched else "no backups found in object storage")
            restored = restore_latest_valid(settings.db_path, settings.backup_dir, settings.backup_passphrase)
        finally:
            daemon.release()
        print(f"restored from {restored}" if restored else "no valid backup found")
        return EXIT_OK if restored else EXIT_FAIL
    result = execute_job(settings, "backup", ist_date(get_clock().now()).isoformat(), force=True)
    print(f"backup: {result.status} - {result.message}")
    return EXIT_OK if result.status == "success" else EXIT_FAIL


def cmd_storage(args: argparse.Namespace) -> int:
    """Where the disk space goes, the retention windows, and (with --prune) a retention run right now."""
    from app.locks import job_lock
    from automation.retention import run_retention, storage_json, storage_report, storage_text

    settings = _settings(args)
    _logging(settings, "cli.log")
    store = _open_store(settings)
    try:
        if args.prune or args.dry_run:
            lock = job_lock(settings.data_dir, "housekeeping")
            if not lock.acquire(blocking=False):
                print("the nightly housekeeping is running right now; try again in a few minutes", file=sys.stderr)
                return EXIT_BUSY
            try:
                report = run_retention(store, settings, dry_run=args.dry_run, force_vacuum=args.vacuum)
            finally:
                lock.release()
            print(report.summary())
            for note in report.notes:
                print(f"  ! {note}")
            if args.vacuum and not args.dry_run:
                print(f"  maintenance: {report.maintenance}")
            print()
        print(storage_json(storage_report(store, settings)) if args.json else storage_text(storage_report(store, settings)))
        return EXIT_OK
    finally:
        store.close()


def cmd_logs(args: argparse.Namespace) -> int:
    """Read the daily log files without knowing where they are or how they are named."""
    from app import logview

    settings = _settings(args)  # no logging setup: reading must not create files
    files = logview.list_log_files(settings.log_dir)
    if args.list:
        print(logview.summary(files))
        return EXIT_OK
    selected = logview.select_files(files, role=args.role, day=args.date)
    if not selected:
        print(f"no '{args.role}' log files" + (f" for {args.date}" if args.date else "") + f" in {settings.log_dir}"
              + (" (nothing at WARNING or above has been logged yet - good)" if args.role == logview.ERRORS_ROLE else ""))
        return EXIT_OK
    since = logview.parse_since(args.since, get_clock().now()) if args.since else None
    filters = {"min_level": args.level, "job": args.job, "grep": args.grep, "since": since}
    for record in logview.read_records(selected, limit=args.lines, **filters):
        print(record.text)
    if args.follow:
        keep = _line_filter(args.level, args.grep)
        with contextlib.suppress(KeyboardInterrupt):
            logview.follow(lambda: logview.select_files(logview.list_log_files(settings.log_dir), role=args.role, day=args.date),
                           lambda line: print(line) if keep(line) else None)
    return EXIT_OK


def _line_filter(level: str | None, grep: str | None) -> Any:
    """Line-level filter for ``--follow``: a continuation line (traceback) follows the decision made for its header."""
    from app import logview

    floor = logview.LEVELS.index(level.upper()) if level else 0
    needle = grep.lower() if grep else None
    state = {"show": True}

    def keep(line: str) -> bool:
        match = logview.HEADER.match(line)
        if match:
            state["show"] = (match["level"] in logview.LEVELS and logview.LEVELS.index(match["level"]) >= floor
                             and (needle is None or needle in line.lower()))
        return state["show"]

    return keep


def cmd_diag(args: argparse.Namespace) -> int:
    """One redacted archive with everything needed to understand a failure (versions, doctor, health, jobs, logs)."""
    from app.diagnostics import build_bundle

    settings = _settings(args)
    _logging(settings, "cli.log")
    store = _open_store(settings)
    try:
        path = build_bundle(settings, store)
    finally:
        store.close()
    print(path)
    print("attach this file to the bug report; every secret in it is masked")
    return EXIT_OK


def cmd_health(args: argparse.Namespace) -> int:
    from automation.health import HealthReport, evaluate_health

    settings = _settings(args)
    _logging(settings, "health.log")
    store = _open_store(settings)
    try:
        report: HealthReport = evaluate_health(store, settings)
        if args.notify:
            _notify_health(settings, store, report)
        if args.json:
            _print_json(report.to_dict())
        else:
            print(f"health: {report.level.upper()}")
            for p in report.problems:
                print(f"  [{p.level}] {p.code}: {p.message}")
        return report.exit_code
    finally:
        store.close()


def _notify_health(settings: Settings, store: Any, report: Any) -> None:
    """Queue alerts for problems and announce a recovery once the system returns to ok."""
    from automation.health import raise_alerts
    from notifications.service import NotificationService

    previous = store.kv_get("health.last_level", "ok")
    raise_alerts(store, report)
    if report.level == "ok" and previous != "ok":
        store.add_alert("info", "health_recovered", "System health is back to OK.", dedupe_key="health-recovered",
                        dedupe_minutes=30)
    store.kv_set("health.last_level", report.level)
    NotificationService(settings, store).flush_alerts()


def cmd_alert_unit(args: argparse.Namespace) -> int:
    """Called by ``monster-ai-alert@.service`` when a unit fails: Telegram message with the last journal lines."""
    settings = _settings(args)
    _logging(settings, "cli.log")
    unit = "".join(ch for ch in args.unit if ch.isalnum() or ch in "-_.@:")
    tail = "(journal not readable)"
    journalctl = shutil.which("journalctl")
    if journalctl:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, unit name sanitised above
                [journalctl, "-u", unit, "-n", "12", "--no-pager", "-o", "cat"], capture_output=True, text=True, timeout=15,
                check=False)
            tail = redact(done.stdout.strip())[-1200:]
        except (OSError, subprocess.SubprocessError):
            pass
    from notifications import report
    from notifications.service import NotificationService
    from notifications.telegram import TelegramClient

    text = report.render_unit_failure(unit, tail, get_clock().now())

    try:
        store = _open_store(settings)
        try:
            outcome = NotificationService(settings, store).send_text(
                "unit_failure", f"unit-failure:{unit}:{get_clock().now():%Y%m%d%H}", text, chat_id=settings.alert_chat_id)
            return EXIT_OK if outcome.ok or outcome.status == "withheld" else EXIT_FAIL
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 - database problems must not stop the alert
        log.warning("alert-unit %s: the database path failed (%s: %s); sending the alert directly", unit,
                    type(exc).__name__, redact(str(exc))[:160])
        client = TelegramClient(settings)
        if client.configured and settings.alert_chat_id:
            client.send_message(settings.alert_chat_id, text)
            return EXIT_OK
        return EXIT_FAIL


def cmd_telegram_setup(args: argparse.Namespace) -> int:
    """Find the chat id of the person who messaged the bot (no manual API calls needed)."""
    from app.envfile import upsert_env
    from notifications.telegram import TelegramClient

    settings = _settings(args)
    token = args.token or settings.telegram_bot_token
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and put the token in the env file.", file=sys.stderr)
        return EXIT_FAIL
    register_secrets([token])
    client = TelegramClient(settings.__class__(**{**settings.__dict__, "telegram_bot_token": token}))
    try:
        me = client.get_me()
    except Exception as exc:  # noqa: BLE001
        print(f"Telegram call failed: {_why(exc)}", file=sys.stderr)
        return EXIT_FAIL
    print(f"bot: @{me.get('username', '?')}")
    print(f"Open Telegram, find @{me.get('username', 'your bot')} and send it any message (e.g. /start).", file=sys.stderr)
    try:
        found = client.discover_chat_id(wait_seconds=float(args.wait))
    except Exception as exc:  # noqa: BLE001
        print(f"Telegram call failed while waiting for your message: {_why(exc)}", file=sys.stderr)
        return EXIT_FAIL
    if not found:
        print("no message received yet; send the bot a message and run this command again", file=sys.stderr)
        return EXIT_FAIL
    print(f"chat: {found['title']} ({found['type']})")
    print(f"TELEGRAM_CHAT_ID={found['chat_id']}")
    if args.write_env and not args.print_only:
        updates = {"TELEGRAM_CHAT_ID": found["chat_id"]}
        if args.token:
            updates["TELEGRAM_BOT_TOKEN"] = args.token
        upsert_env(args.write_env, updates)
        print(f"written to {args.write_env}")
    return EXIT_OK


def cmd_telegram_test(args: argparse.Namespace) -> int:
    from notifications.service import NotificationService

    settings = _settings(args)
    _logging(settings, "cli.log")
    store = _open_store(settings)
    try:
        outcome = NotificationService(settings, store).send_test()
    finally:
        store.close()
    print(f"telegram test: {outcome.status}" + (f" ({outcome.reason})" if outcome.reason else ""))
    return EXIT_OK if outcome.ok else EXIT_FAIL


def cmd_telegram_preview(args: argparse.Namespace) -> int:
    """Send one sample of every kind of Telegram message (made-up data, each one marked as a preview) to see how they look."""
    from notifications import samples
    from notifications.telegram import TelegramClient

    if args.list:
        for kind, what in samples.KINDS.items():
            print(f"{kind:9} {what}")
        return EXIT_OK
    unknown = sorted(set(args.only or ()) - set(samples.KINDS))
    if unknown:
        print(f"unknown kind(s) {', '.join(unknown)}; choose from: {', '.join(samples.KINDS)}", file=sys.stderr)
        return EXIT_USAGE
    settings = _settings(args)
    _logging(settings, "cli.log")
    client = TelegramClient(settings)
    if not client.configured or not settings.telegram_chat_id:
        print("Telegram is not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see telegram-setup).", file=sys.stderr)
        return EXIT_FAIL
    wanted = set(args.only or samples.KINDS)
    sent = 0
    try:
        for kind, messages in samples.build_samples(settings, get_clock().now()):
            if kind not in wanted:
                continue
            chat = settings.alert_chat_id if kind in samples.ALERT_KINDS else settings.telegram_chat_id
            for text in samples.marked(messages):
                if sent:
                    get_clock().sleep(args.delay)  # Telegram allows about one message per second to a chat
                client.send_message(chat, text)
                sent += 1
            print(f"preview {kind}: sent")
    except Exception as exc:  # noqa: BLE001
        print(f"Telegram call failed after {sent} message(s): {_why(exc)}", file=sys.stderr)
        return EXIT_FAIL
    print(f"preview: {sent} sample message(s) sent; every one is made-up data, not a market signal")
    return EXIT_OK


def cmd_ai_test(args: argparse.Namespace) -> int:
    """Send one made-up news item to Groq and Jev through the production code and say whether each answered (or why not)."""
    from news_intelligence.ai_check import run_checks

    settings = _settings(args)
    _logging(settings, "cli.log")
    store = resolver = None
    try:  # the entity resolver makes the LLM's companies resolvable; without a database the check still works
        from news_intelligence.entities import EntityResolver

        store = _open_store(settings)
        resolver = EntityResolver(store)
    except Exception as exc:  # noqa: BLE001
        log.warning("ai-test runs without entity resolution: %s", _why(exc))
    try:
        checks = run_checks(settings, only=None if args.service == "all" else [args.service], deep=not args.list_only,
                            resolver=resolver)
    finally:
        if store is not None:
            store.close()
    if args.json:
        _print_json([check.to_dict() for check in checks])
    else:
        for check in checks:
            took = "" if check.seconds is None else f" {check.seconds:.1f}s"
            print(f"{check.label:<22} {check.status.upper():<5}{took}  {check.detail}")
    active = [check for check in checks if check.status != "off"]
    if not active:
        print("nothing to test: the selected service is not configured (see the line above); set GROQ_API_KEY and/or "
              "TYPESAFE_API_KEY in the env file to switch the optional AI services on", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK if all(check.usable for check in active) else EXIT_FAIL


def cmd_dashboard(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _logging(settings, "dashboard.log")
    try:
        from dashboard import serve
    except ImportError:
        print("dashboard dependencies are not installed: pip install -r requirements-dashboard.txt", file=sys.stderr)
        return EXIT_FAIL
    return serve(settings)


# ------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monster", description="MONSTER AI - NSE large-move intelligence (shadow mode).")
    parser.add_argument("--env-file", help="dotenv file (default: $MONSTER_ENV_FILE or ./.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Any, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        return p

    add("version", cmd_version, "print the version")
    add("env-example", cmd_env_example, "print a documented .env template")
    add("init", cmd_init, "create directories, migrate the database, seed the calendar")
    p = add("doctor", cmd_doctor, "verify this machine can run the system")
    p.add_argument("--offline", action="store_true", help="skip network checks")
    p.add_argument("--json", action="store_true")
    p = add("bootstrap", cmd_bootstrap, "first-run setup (resumable)")
    p.add_argument("--resume", action="store_true", help="continue from the last completed step")
    p.add_argument("--skip-train", action="store_true", help="do not train models during bootstrap")
    add("daemon", cmd_daemon, "run the scheduler")
    p = add("run-job", cmd_run_job, "run one job now (used by the scheduler)")
    p.add_argument("job")
    p.add_argument("--key", help="run key (default: today's IST date)")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--timeout-min", type=int)
    p.add_argument("--last-attempt", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="job option (repeatable)")
    add("jobs", cmd_jobs, "show the timetable and recent runs")
    p = add("scan", cmd_scan, "run a scan now")
    p.add_argument("--stage", choices=("pre", "final"), default="final")
    p.add_argument("--date")
    p.add_argument("--dry-run", action="store_true", help="print the report; freeze and send nothing")
    p.add_argument("--force", action="store_true")
    p = add("evaluate", cmd_evaluate, "evaluate a day's predictions")
    p.add_argument("--date")
    p = add("train", cmd_train, "train the models now")
    p.add_argument("--stage", choices=("pre", "final", "both"), default="both")
    p.add_argument("--publish", action="store_true", help="CI mode: train, sign and upload the artifact bundle")
    p = add("backtest", cmd_backtest, "walk-forward evaluation vs baselines (writes a report)")
    p.add_argument("--stage", choices=("pre", "final", "both"), default="final")
    p.add_argument("--years", type=int, default=6)
    p.add_argument("--out")
    p = add("models", cmd_models, "inspect and govern models")
    p.add_argument("action", choices=("list", "approve", "activate", "rollback"))
    p.add_argument("version", nargs="?")
    p.add_argument("--status", choices=("shadow", "validated"), default="shadow")
    p.add_argument("--stage", choices=("pre", "final"), default="final")
    p = add("backup", cmd_backup, "back up the database (or verify/restore)")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--restore-latest", action="store_true")
    p.add_argument("--remote", action="store_true", help="with --restore-latest: first download the newest off-box backup")
    p = add("health", cmd_health, "system health (exit 0 ok, 1 warnings, 2 critical)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--notify", action="store_true", help="queue and send alerts for problems")
    p = add("storage", cmd_storage, "disk usage, row counts, retention windows (and --prune to apply them now)")
    p.add_argument("--prune", action="store_true", help="apply the retention rules now (the nightly housekeeping does this too)")
    p.add_argument("--dry-run", action="store_true", help="with --prune: only count what would be removed")
    p.add_argument("--vacuum", action="store_true", help="with --prune: also VACUUM the database to return freed pages")
    p.add_argument("--json", action="store_true")
    p = add("logs", cmd_logs, "read the daily logs (default: today's warnings and errors of every process)")
    p.add_argument("--role", default="errors", help="errors (default), all, daemon, jobs, cli, health, train, bootstrap, dashboard")
    p.add_argument("--date", help="YYYY-MM-DD in the display zone (default: the newest day)")
    p.add_argument("-n", "--lines", type=int, default=60, help="how many records (default 60)")
    p.add_argument("--level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), type=str.upper)
    p.add_argument("--job", help="only records of this job (substring of the job tag)")
    p.add_argument("--grep", help="only records containing this text (case-insensitive)")
    p.add_argument("--since", help="only records newer than e.g. 90m, 2h, 1d or an ISO date")
    p.add_argument("--follow", "-f", action="store_true", help="keep printing new lines (Ctrl-C to stop)")
    p.add_argument("--list", action="store_true", help="list the log files and their sizes")
    add("diag", cmd_diag, "write a redacted diagnostics bundle for a bug report")
    p = add("alert-unit", cmd_alert_unit, "send a Telegram alert for a failed systemd unit")
    p.add_argument("unit")
    p = add("telegram-setup", cmd_telegram_setup, "discover the Telegram chat id")
    p.add_argument("--token")
    p.add_argument("--wait", type=int, default=120)
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--write-env")
    add("telegram-test", cmd_telegram_test, "send a test message")
    p = add("telegram-preview", cmd_telegram_preview, "send one sample of every Telegram message type (made-up data)")
    p.add_argument("--only", nargs="+", metavar="KIND", help="only these kinds (see --list)")
    p.add_argument("--list", action="store_true", help="list the kinds of sample message and exit")
    p.add_argument("--delay", type=float, default=1.2, help="seconds between messages (default 1.2)")
    p = add("ai-test", cmd_ai_test, "check the optional Groq / Jev AI keys with one made-up news item (and one real web search)")
    p.add_argument("--service", choices=("all", "llm", "web", "jev"), default="all")
    p.add_argument("--list-only", action="store_true", help="only list the models the key can use (spends no tokens, no web search)")
    p.add_argument("--json", action="store_true")
    add("dashboard", cmd_dashboard, "run the read-only dashboard")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    pin_process_timezone()  # the code never reads the server's zone; this makes any library that does deterministic too
    for stream in (sys.stdout, sys.stderr):  # a legacy console codepage must never crash a command on a "₹" in a headline
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.command in {"models"} and args.action in {"approve", "activate"} and not args.version:
        print(f"models {args.action} needs a VERSION", file=sys.stderr)
        return EXIT_USAGE
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
