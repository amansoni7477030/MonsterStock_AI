"""External training mode (signed bundles pulled by the server), the CI entry point ``train --publish`` and the off-box
backup helpers. A small model is really trained once; everything else runs against a local-directory object store."""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.clock import FakeClock
from app.config import load_settings
from app.errors import DataUnavailable, ModelArtifactError
from automation import jobs
from automation.context import JobResult
from database.backup import backup_database, retained_names
from database.object_store import LocalDirStore
from database.remote_backup import fetch_latest_backup, prune_remote_backups, restore_from_remote
from database.store import Store
from models import external, training
from models.bundle import publish_bundle
from models.inference import ModelBundle
from models.registry import ModelRegistry
from tests.seed_data import SMALL_TRAIN, make_job_context, panel_ending, populate_store

ROOT = Path(__file__).resolve().parent.parent
KEY = "unit-test-signing-key-0123456789abcdef"
NOW = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)


def make_settings(root: Path, **overrides):
    env = {"MONSTER_ENV": "test", "MONSTER_HOME": str(ROOT), "MONSTER_DATA_DIR": str(root / "data"),
           "MONSTER_LOG_DIR": str(root / "logs"), "MODEL_SIGNING_KEY": KEY}
    cfg = dataclasses.replace(load_settings(env, env_file=root / "missing.env"), **overrides)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One really trained (small) 'pre' model in a worker workspace."""
    root = tmp_path_factory.mktemp("worker")
    clock = FakeClock(NOW)
    cfg = make_settings(root, gate_max_ece=0.5, gate_min_brier_skill=-1.0)
    store = Store.open(cfg.db_path, clock=clock)
    populate_store(store, panel_ending(n_dates=470, n_symbols=40, seed=21))
    real = training.TrainConfig
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, "TrainConfig", lambda **kw: real(**{**kw, **SMALL_TRAIN}))
        ctx = make_job_context(cfg, store, clock, run_key="2026-W44")
        ctx.options = {"stages": "pre"}
        result = jobs.train(ctx)
    store.close()
    assert result.status == "success", result.message
    return SimpleNamespace(settings=cfg, version=result.metrics["pre"]["version"], status=result.metrics["pre"]["status"])


@pytest.fixture()
def remote(tmp_path):
    return LocalDirStore(tmp_path / "remote")


@pytest.fixture()
def server(tmp_path, fake_clock):
    cfg = make_settings(tmp_path / "server")
    store = Store.open(cfg.db_path, clock=fake_clock)
    yield SimpleNamespace(settings=cfg, store=store, registry=ModelRegistry(store, cfg), clock=fake_clock)
    store.close()


def clone_model(trained, models_dir: Path, name: str, **meta_overrides) -> str:
    """A copy of the trained model under a new version with a doctored meta.json (the draft manifest is dropped, so
    publishing signs whatever is in the directory)."""
    target = models_dir / name
    shutil.copytree(trained.settings.models_dir / trained.version, target)
    (target / "manifest.json").unlink()
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    meta.update(version=name, **meta_overrides)
    (target / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return name


# ------------------------------------------------------------------------------------------ publish + sync
def test_a_published_model_is_pulled_verified_registered_and_loadable(trained, remote, server):
    keys = external.publish_models(trained.settings, {"pre": trained.version}, sink=remote, clock=server.clock)
    assert keys == {"pre": f"models/pre/{trained.version}.zip"}

    outcome = external.sync_models(server.settings, server.store, server.registry, source=remote)

    assert outcome["synced"] == [f"pre:{trained.version} ({trained.status}, activated)"]
    assert len(outcome["failed"]) == 1 and outcome["failed"][0].startswith("final: no published model bundle")
    record = server.registry.active_record("pre")
    assert record["model_version"] == trained.version and record["status"] == trained.status
    bundle = server.registry.active_bundle("pre")  # re-verifies every file and the checksum recorded at registration
    assert isinstance(bundle, ModelBundle) and bundle.version == trained.version
    assert [a["code"] for a in server.store.pending_alerts(10)] == ["model_sync_failed"]


def test_a_second_sync_changes_nothing(trained, remote, server):
    external.publish_models(trained.settings, {"pre": trained.version}, sink=remote, clock=server.clock)
    external.sync_models(server.settings, server.store, server.registry, source=remote)
    again = external.sync_models(server.settings, server.store, server.registry, source=remote)
    assert again["synced"] == [] and again["current"] == [f"pre:{trained.version}"]
    assert len(server.store.list_models("pre")) == 1


def test_a_bundle_signed_with_another_key_is_never_installed(trained, remote, server):
    external.publish_models(trained.settings, {"pre": trained.version}, sink=remote, clock=server.clock)
    wrong = dataclasses.replace(server.settings, model_signing_key="a-different-key-" + "x" * 30)
    outcome = external.sync_models(wrong, server.store, ModelRegistry(server.store, wrong), source=remote)
    assert outcome["synced"] == [] and "signature mismatch" in outcome["failed"][0]
    assert server.store.list_models() == [] and not (server.settings.models_dir / trained.version).exists()


def test_a_tampered_archive_is_rejected_by_the_signed_pointer(trained, remote, server):
    external.publish_models(trained.settings, {"pre": trained.version}, sink=remote, clock=server.clock)
    archive = remote.root / "models" / "pre" / f"{trained.version}.zip"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    outcome = external.sync_models(server.settings, server.store, server.registry, source=remote)
    assert outcome["synced"] == [] and "pointer" in outcome["failed"][0]
    assert server.store.list_models() == []


@pytest.mark.parametrize("override, expected", [
    ({"status": "rejected"}, "not publishable"),
    ({"feature_version": "fv0"}, "feature version"),
    ({"stage": "final"}, "does not describe"),
])
def test_a_signed_but_unusable_bundle_is_refused(trained, remote, server, tmp_path, override, expected):
    worker = dataclasses.replace(trained.settings, models_dir=tmp_path / "worker-models")
    name = clone_model(trained, worker.models_dir, "m-doctored-pre", **override)
    external.publish_models(worker, {"pre": name}, sink=remote, clock=server.clock)
    outcome = external.sync_models(server.settings, server.store, server.registry, source=remote)
    assert outcome["synced"] == [] and expected in outcome["failed"][0]
    assert server.store.list_models() == []


def test_sync_needs_a_source_and_a_key(server, remote):
    assert external.sync_models(server.settings, server.store, server.registry)["message"].startswith("no artifact source")
    keyless = dataclasses.replace(server.settings, model_signing_key=None)
    with pytest.raises(ModelArtifactError, match="MODEL_SIGNING_KEY"):
        external.sync_models(keyless, server.store, server.registry, source=remote)


def test_a_bundle_without_metadata_is_refused(remote, server, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "model.txt").write_text("not a model", encoding="utf-8")
    publish_bundle(remote, "m-bare-pre", bare, KEY, stage="pre", clock=server.clock)
    outcome = external.sync_models(server.settings, server.store, server.registry, source=remote)
    assert outcome["synced"] == [] and "meta.json is unreadable" in outcome["failed"][0]


def test_a_pointer_that_moves_between_the_two_reads_registers_nothing_twice(trained, server, tmp_path):
    class Flapping(LocalDirStore):
        """Serves a scripted sequence of latest.json answers (a publish racing the sync)."""

        def __init__(self, root, pointers):
            super().__init__(root)
            self.pointers = list(pointers)

        def get(self, key, **kwargs):
            if key == "models/pre/latest.json" and self.pointers:
                return self.pointers.pop(0)
            return super().get(key, **kwargs)

    remote = Flapping(tmp_path / "remote", [])
    external.publish_models(trained.settings, {"pre": trained.version}, sink=remote, clock=server.clock)
    old_pointer = (remote.root / "models" / "pre" / "latest.json").read_bytes()
    external.sync_models(server.settings, server.store, server.registry, source=remote)

    worker = dataclasses.replace(trained.settings, models_dir=tmp_path / "worker-models")
    newer = clone_model(trained, worker.models_dir, "m-newer-pre")
    external.publish_models(worker, {"pre": newer}, sink=remote, clock=server.clock)
    new_pointer = (remote.root / "models" / "pre" / "latest.json").read_bytes()
    remote.pointers = [new_pointer, old_pointer]  # first read: the unseen version; second read: the known one again
    outcome = external.sync_models(server.settings, server.store, server.registry, source=remote)
    assert outcome["synced"] == [] and f"pre:{trained.version}" in outcome["current"]
    assert [m["model_version"] for m in server.store.list_models("pre")] == [trained.version]


def test_publishing_needs_storage_a_key_and_readable_metadata(trained, remote, tmp_path):
    with pytest.raises(ModelArtifactError, match="no S3 object store"):
        external.publish_models(trained.settings, {"pre": trained.version})
    keyless = dataclasses.replace(trained.settings, model_signing_key=None)
    with pytest.raises(ModelArtifactError, match="MODEL_SIGNING_KEY"):
        external.publish_models(keyless, {"pre": trained.version}, sink=remote)
    worker = dataclasses.replace(trained.settings, models_dir=tmp_path / "empty")
    with pytest.raises(ModelArtifactError, match="meta.json is unreadable"):
        external.publish_models(worker, {"pre": "m-missing-pre"}, sink=remote)


def test_the_model_sync_job_reports_failure_only_when_nothing_could_be_synced(trained, remote, server, monkeypatch):
    external_cfg = dataclasses.replace(server.settings, training_mode="external", model_artifact_url="https://models.example.com/x")
    monkeypatch.setattr("models.external.build_artifact_source", lambda settings: remote)
    ctx = make_job_context(external_cfg, server.store, server.clock)
    nothing = jobs.model_sync(ctx)
    assert nothing.status == "failed" and "no published model bundle" in nothing.message

    external.publish_models(trained.settings, {"pre": trained.version}, sink=remote, clock=server.clock)
    partial = jobs.model_sync(make_job_context(external_cfg, server.store, server.clock))
    assert partial.status == "success" and f"synced pre:{trained.version}" in partial.message


# ------------------------------------------------------------------------------- train --publish (CI worker)
@pytest.fixture()
def ci(trained, remote, tmp_path, monkeypatch):
    """The CI entry point with the training job faked: tests choose the outcomes it reports."""
    models = tmp_path / "ci-models"
    cfg = dataclasses.replace(trained.settings, models_dir=models)
    pre = clone_model(trained, models, "m-ci-pre")
    final = clone_model(trained, models, "m-ci-final", stage="final")
    state = SimpleNamespace(calls=[], outcomes={}, status="success", settings=cfg, remote=remote, pre=pre, final=final)

    def fake_execute_job(settings, name, run_key, **kwargs):
        state.calls.append((name, kwargs.get("options"), settings.training_mode))
        return JobResult(state.status, "fake run", state.outcomes if name == "train" else {})

    monkeypatch.setattr("automation.runner.execute_job", fake_execute_job)
    monkeypatch.setattr(external, "_has_market_data", lambda settings: True)
    return state


def run_ci(ci, stage="both", sink="default"):
    return external.train_and_publish(ci.settings, stage=stage, sink=ci.remote if sink == "default" else sink)


def test_ci_publishes_exactly_the_models_it_trained_and_trains_whatever_the_server_mode(ci, capsys):
    ci.settings = dataclasses.replace(ci.settings, training_mode="external")
    ci.outcomes = {"pre": {"version": ci.pre, "status": "validated"}, "final": {"version": ci.final, "status": "shadow"}}
    assert run_ci(ci) == 0
    assert ci.calls == [("train", {"stages": "pre,final"}, "local-light")]
    assert {o.key for o in ci.remote.list("models/")} == {
        f"models/pre/{ci.pre}.zip", "models/pre/latest.json", f"models/final/{ci.final}.zip", "models/final/latest.json"}
    assert "published pre" in capsys.readouterr().out


def test_ci_does_not_publish_or_fail_for_models_that_failed_their_gates(ci, capsys):
    ci.outcomes = {"pre": {"version": ci.pre, "status": "rejected", "reason": "gates failed: not calibrated"}}
    assert run_ci(ci, stage="pre") == 0
    assert ci.remote.list("models/") == []
    assert "pre: not published (gates failed: not calibrated)" in capsys.readouterr().out


def test_ci_fails_when_a_stage_could_not_be_trained_but_still_publishes_the_other(ci, capsys):
    ci.outcomes = {"pre": {"version": ci.pre, "status": "validated"}, "final": {"error": "insufficient history"}}
    assert run_ci(ci) == 1
    assert [o.key for o in ci.remote.list("models/pre/")] != [] and ci.remote.list("models/final/") == []
    assert "final: not published (insufficient history)" in capsys.readouterr().out


def test_ci_fails_when_training_fails_or_publishing_is_impossible(ci, capsys):
    ci.status = "failed"
    assert run_ci(ci) == 1 and ci.remote.list("models/") == []

    class Down(LocalDirStore):
        def put(self, key, data, **kwargs):
            raise DataUnavailable("object_store", "storage is down")

    ci.status, ci.outcomes = "success", {"pre": {"version": ci.pre, "status": "validated"}}
    assert run_ci(ci, stage="pre", sink=Down(ci.remote.root)) == 1
    assert "publish failed" in capsys.readouterr().out

    keyless = dataclasses.replace(ci.settings, model_signing_key=None)
    assert external.train_and_publish(keyless, stage="pre", sink=ci.remote) == 1
    assert external.train_and_publish(ci.settings, stage="pre") == 1  # no S3 object store configured either
    assert "publishing needs" in capsys.readouterr().out


@pytest.mark.parametrize("restored", [True, False])
def test_ci_collects_market_data_when_the_workspace_is_empty(ci, monkeypatch, restored):
    monkeypatch.setattr(external, "_has_market_data", lambda settings: False)
    monkeypatch.setattr("database.remote_backup.restore_from_remote", lambda settings, store: Path("b") if restored else None)
    ci.outcomes = {"pre": {"version": ci.pre, "status": "validated"}}
    assert run_ci(ci, stage="pre") == 0
    jobs_run = [call[0] for call in ci.calls]
    assert jobs_run == (["train"] if restored else ["bootstrap", "train"])
    if not restored:
        assert ci.calls[0][1] == {"data_only": "1"}  # no training, smoke scan or Telegram test on the worker


def test_market_data_detection_counts_symbols_with_history(settings, store, fake_clock):
    assert external._has_market_data(settings) is False
    populate_store(store, panel_ending(n_dates=40, n_symbols=external.MIN_SYMBOLS_WITH_HISTORY, seed=3))
    assert external._has_market_data(settings) is True


# ---------------------------------------------------------------------------------------- off-box backups
def make_backup(tmp_path: Path, when: datetime, marker: str) -> Path:
    db = tmp_path / f"db-{marker}" / "monster.db"
    db.parent.mkdir(parents=True)
    store = Store.open(db)
    store.kv_set("marker", marker)
    store.conn.commit()
    store.close()
    return backup_database(db, tmp_path / "local-backups", clock=FakeClock(when)).path


def upload(remote, path: Path) -> str:
    key = f"backups/{path.name}"
    remote.put_file(key, path)
    return key


def test_the_newest_off_box_backup_is_fetched_and_restored(tmp_path, remote):
    for days, marker in ((2, "old"), (1, "middle"), (0, "newest")):
        upload(remote, make_backup(tmp_path, NOW - timedelta(days=days), marker))
    remote.put("backups/readme.txt", b"not a backup")
    remote.put("backups/monster-garbage.db.gz", b"not a backup either")
    assert fetch_latest_backup(remote, tmp_path / "dl").name == "monster-20261102T030000Z.db.gz"

    cfg = make_settings(tmp_path / "restored")
    restored = restore_from_remote(cfg, remote)
    assert restored is not None
    fresh = Store.open(cfg.db_path)
    try:
        assert fresh.kv_get("marker") == "newest"
    finally:
        fresh.close()


def test_a_corrupt_newest_backup_falls_back_to_the_previous_one(tmp_path, remote):
    upload(remote, make_backup(tmp_path, NOW - timedelta(days=1), "good"))
    remote.put("backups/monster-20261102T030000Z.db.gz", b"truncated upload")
    cfg = make_settings(tmp_path / "restored")
    assert restore_from_remote(cfg, remote).name == "monster-20261101T030000Z.db.gz"
    assert not (cfg.backup_dir / "monster-20261102T030000Z.db.gz").exists()


def test_a_download_failure_falls_back_to_the_previous_backup(tmp_path, remote):
    upload(remote, make_backup(tmp_path, NOW - timedelta(days=1), "good"))
    upload(remote, make_backup(tmp_path, NOW, "unreachable"))

    class Flaky(LocalDirStore):
        def get_to_file(self, key, dest, **kwargs):
            if key.endswith("20261102T030000Z.db.gz"):
                raise DataUnavailable("object_store", "connection reset")
            return super().get_to_file(key, dest, **kwargs)

    flaky = Flaky(remote.root)
    assert fetch_latest_backup(flaky, tmp_path / "dl").name == "monster-20261101T030000Z.db.gz"


def test_no_usable_off_box_backup_returns_none(tmp_path, remote):
    cfg = make_settings(tmp_path / "restored")
    assert restore_from_remote(cfg, remote) is None
    remote.put("backups/monster-20261102T030000Z.db.gz", b"junk")
    assert restore_from_remote(cfg, remote) is None


def test_off_box_retention_matches_the_local_policy_and_spares_foreign_objects(tmp_path, remote):
    start = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)
    names = []
    for day in range(40):
        name = f"monster-{(start + timedelta(days=day)).strftime('%Y%m%dT%H%M%SZ')}.db.gz"
        remote.put(f"backups/{name}", b"x")
        names.append(name)
    remote.put("backups/keep-me.txt", b"foreign")
    removed = prune_remote_backups(remote, keep_daily=7, keep_weekly=4)
    kept = {o.key.rsplit("/", 1)[-1] for o in remote.list("backups/")}
    assert kept == retained_names(names, 7, 4) | {"keep-me.txt"}
    assert set(names[-7:]) <= kept and 7 <= len(kept) - 1 <= 11  # the last week in full plus a few weekly copies
    assert len(removed) == 40 - (len(kept) - 1)
    assert prune_remote_backups(remote, keep_daily=7, keep_weekly=4) == []
