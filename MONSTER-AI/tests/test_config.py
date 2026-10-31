from __future__ import annotations

from pathlib import Path

import pytest

from app.config import VARS, ConfigError, Settings, env_example_text, load_settings

ROOT = Path(__file__).resolve().parent.parent


def _load(tmp_path: Path, **env: str) -> Settings:
    return load_settings({"MONSTER_ENV": "test", **env}, env_file=tmp_path / "none.env")


def test_defaults_are_valid_and_safe(tmp_path):
    s = _load(tmp_path)
    assert s.mode == "shadow"
    assert s.top_n == 5
    assert s.range_thresholds == (1.0, 2.0, 3.0, 5.0)
    assert s.primary_threshold == 3.0
    assert s.report_deadline_time < "09:15:00"
    assert s.final_scan_time == "09:08:45"
    assert s.prelim_scan_time == "08:45:00"  # normalised to HH:MM:SS
    assert s.dashboard_enabled is False and s.dashboard_host == "127.0.0.1"
    assert s.db_path.name == "monster.db"
    assert s.telegram_configured is False


def test_env_overrides_and_paths_follow_data_dir(tmp_path):
    s = _load(tmp_path, MONSTER_DATA_DIR=str(tmp_path / "d"), TOP_N="7", SEND_PRELIM_REPORT="yes")
    assert s.db_path == tmp_path / "d" / "monster.db"
    assert s.models_dir == tmp_path / "d" / "models"
    assert s.top_n == 7 and s.send_prelim_report is True


def test_all_errors_are_reported_together(tmp_path):
    with pytest.raises(ConfigError) as err:
        _load(tmp_path, TOP_N="abc", SEND_PRELIM_REPORT="maybe", PRELIM_SCAN_TIME="25:00")
    message = str(err.value)
    assert "TOP_N" in message and "SEND_PRELIM_REPORT" in message and "PRELIM_SCAN_TIME" in message


@pytest.mark.parametrize(
    "env, needle",
    [
        ({"RANGE_THRESHOLDS": "3,2,1"}, "ascending"),
        ({"PRIMARY_THRESHOLD": "4"}, "PRIMARY_THRESHOLD"),
        ({"TOP_N": "0"}, "TOP_N"),
        ({"TIER_WATCH_P": "0.6"}, "TIER_WATCH_P"),
        ({"UNUSUAL_K": "1.0"}, "UNUSUAL_K"),
        ({"UNIVERSE_MODE": "custom"}, "UNIVERSE_SYMBOLS"),
        ({"MIN_FREE_DISK_MB_CRIT": "2000"}, "MIN_FREE_DISK_MB_CRIT"),
        ({"REPORT_DEADLINE_TIME": "09:16:00"}, "09:15"),
        ({"DASHBOARD_ENABLED": "true", "DASHBOARD_HOST": "0.0.0.0"}, "DASHBOARD_TOKEN"),
        ({"LLM_API_BASE": "https://x.example/v1"}, "LLM_API_BASE and LLM_MODEL"),
        ({"LLM_API_BASE": "http://evil.example/v1", "LLM_MODEL": "m"}, "https"),
        ({"TRAINING_MODE": "external"}, "MODEL_ARTIFACT_URL"),
        ({"S3_BUCKET": "b"}, "S3_BUCKET"),
        ({"TELEGRAM_BOT_TOKEN": "not-a-token"}, "BotFather"),
        ({"MONSTER_TIMEZONE": "Mars/Base"}, "MONSTER_TIMEZONE"),
        ({"HISTORY_YEARS": "3", "TRAIN_YEARS": "5"}, "HISTORY_YEARS"),
    ],
)
def test_cross_field_validation(tmp_path, env, needle):
    with pytest.raises(ConfigError) as err:
        _load(tmp_path, **env)
    assert needle in str(err.value)


def test_valid_token_and_redaction(tmp_path):
    token = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"
    s = _load(tmp_path, TELEGRAM_BOT_TOKEN=token, TELEGRAM_CHAT_ID="42", FRED_API_KEY="abcdef123456")
    assert s.telegram_configured and s.alert_chat_id == "42"
    shown = s.redacted()
    assert shown["telegram_bot_token"] == "***set***" and shown["fred_api_key"] == "***set***"
    assert token not in str(shown)
    assert token in s.secrets()


def test_env_file_parsing(tmp_path):
    env_file = tmp_path / "x.env"
    env_file.write_text(
        "# comment\nTOP_N=6\nexport RANGE_THRESHOLDS=\"1,2,3\"\nPRIMARY_THRESHOLD=2 # inline\nlowercase=ignored\nBAD LINE\n"
    )
    s = load_settings({"MONSTER_ENV": "test"}, env_file=env_file)
    assert s.top_n == 6 and s.range_thresholds == (1.0, 2.0, 3.0) and s.primary_threshold == 2.0


def test_real_environment_beats_env_file(tmp_path):
    env_file = tmp_path / "x.env"
    env_file.write_text("TOP_N=9\n")
    s = load_settings({"MONSTER_ENV": "test", "TOP_N": "4"}, env_file=env_file)
    assert s.top_n == 4


def test_empty_value_falls_back_to_default(tmp_path):
    assert _load(tmp_path, TOP_N="").top_n == 5


def test_env_example_is_complete_and_current():
    text = env_example_text()
    for var in VARS:
        assert f"\n{var.env}=" in "\n" + text, var.env
    committed = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert committed == text, "regenerate with: python -m app.cli env-example > .env.example"


def test_every_var_maps_to_a_settings_field():
    fields = set(Settings.__dataclass_fields__)
    assert {v.attr for v in VARS} <= fields
    assert len({v.env for v in VARS}) == len(VARS)


def test_ensure_dirs(settings):
    assert settings.data_dir.exists() and settings.models_dir.exists() and settings.tmp_dir.exists()
