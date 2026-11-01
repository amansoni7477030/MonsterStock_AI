"""The code fingerprint that is stamped on snapshots and models."""

from __future__ import annotations

from pathlib import Path

from app import __version__, release


def tree(tmp_path: Path) -> Path:
    for name, text in (("app/a.py", "x = 1\n"), ("automation/b.py", "y = 2\n"), ("database/migrations/0001.sql", "select 1;\n"),
                       ("data/seeds/s.json", "{}"), ("tests/t.py", "ignored\n"), ("app/__pycache__/c.pyc", "ignored")):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_the_fingerprint_is_stable_and_changes_with_any_source_or_seed_file(tmp_path):
    root = tree(tmp_path)
    first = release.code_fingerprint(root)
    assert len(first) == 8 and release.code_fingerprint(root) == first
    (root / "app" / "a.py").write_text("x = 2\n", encoding="utf-8")
    second = release.code_fingerprint(root)
    assert second != first
    (root / "data" / "seeds" / "s.json").write_text('{"a": 1}', encoding="utf-8")
    assert release.code_fingerprint(root) not in {first, second}


def test_tests_caches_and_foreign_files_do_not_move_the_fingerprint(tmp_path):
    root = tree(tmp_path)
    before = release.code_fingerprint(root)
    (root / "tests" / "t.py").write_text("changed\n", encoding="utf-8")
    (root / "app" / "__pycache__" / "c.pyc").write_text("changed", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "x.md").write_text("changed", encoding="utf-8")
    assert release.code_fingerprint(root) == before


def test_the_same_files_in_another_place_give_the_same_id(tmp_path):
    one, two = tree(tmp_path / "one"), tree(tmp_path / "two")
    assert release.code_fingerprint(one) == release.code_fingerprint(two)


def test_the_running_code_has_an_id_and_the_environment_can_override_it(monkeypatch):
    release.release_id.cache_clear()
    monkeypatch.delenv("MONSTER_RELEASE", raising=False)
    assert release.release_id().startswith(f"{__version__}+") and len(release.release_id()) == len(__version__) + 9
    release.release_id.cache_clear()
    monkeypatch.setenv("MONSTER_RELEASE", "abc1234")
    assert release.release_id() == f"{__version__}+abc1234"
    release.release_id.cache_clear()


def test_every_code_package_of_the_repository_is_fingerprinted():
    packages = {p.name for p in release.ROOT.iterdir() if p.is_dir() and (p / "__init__.py").is_file() and p.name != "tests"}
    assert packages <= set(release.CODE_DIRS), f"add to app.release.CODE_DIRS: {sorted(packages - set(release.CODE_DIRS))}"


def test_an_unreadable_tree_never_stops_a_freeze(monkeypatch):
    release.release_id.cache_clear()
    monkeypatch.delenv("MONSTER_RELEASE", raising=False)

    def broken(root=release.ROOT):
        raise PermissionError("denied")

    monkeypatch.setattr(release, "code_fingerprint", broken)
    assert release.release_id() == f"{__version__}+unknown"
    release.release_id.cache_clear()
