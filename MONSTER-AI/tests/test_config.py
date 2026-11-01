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
        ({"LLM_MODEL": "m"}, "LLM_API_BASE and LLM_MODEL"),
        ({"LLM_API_BASE": "https://api.groq.com/openai/v1", "LLM_MODEL": "openai/gpt-oss-20b"}, "neither GROQ_API_KEY nor LLM_API_KEY"),
        ({"LLM_MAX_SECONDS_PER_RUN": "5"}, "LLM_MAX_SECONDS_PER_RUN"),
        ({"LLM_REASONING_EFFORT": "extreme"}, "LLM_REASONING_EFFORT"),
        ({"JEV_MIN_CONFIDENCE": "1.5"}, "JEV_MIN_CONFIDENCE"),
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
    s = _load(tmp_path, TELEGRAM_BOT_TOKEN=token, TELEGRAM_CHAT_ID="42", BACKUP_PASSPHRASE="abcdef123456")
    assert s.telegram_configured and s.alert_chat_id == "42"
    shown = s.redacted()
    assert shown["telegram_bot_token"] == "***set***" and shown["backup_passphrase"] == "***set***"
    assert token not in str(shown)
    assert token in s.secrets()
    assert token not in repr(s) and "abcdef123456" not in repr(s) and "***set***" in repr(s)  # logs and tracebacks stay clean


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


# -------------------------------------------------------------------------------------- the AI services
GROQ = "gsk_test_groq_key_1234567890"
TYPESAFE = "ts_test_typesafe_key_1234567890"


def test_the_ai_services_are_off_by_default_and_the_secrets_are_secret(tmp_path):
    s = _load(tmp_path)
    assert (s.llm_configured, s.jev_configured, s.llm_provider) == (False, False, None)
    assert s.llm_base_url is None and s.llm_key is None and s.llm_model_name is None and s.llm_fallback_name is None
    assert (s.jev_model, s.llm_reasoning_effort, s.jev_min_confidence) == ("jev-1.14.0", "low", 0.5)
    secret_vars = {v.env for v in VARS if v.secret}
    assert {"GROQ_API_KEY", "TYPESAFE_API_KEY", "LLM_API_KEY"} <= secret_vars


def test_a_groq_key_alone_is_enough_and_selects_the_best_model_with_a_fallback(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ)
    assert s.llm_configured and s.llm_provider == "groq" and s.llm_key == GROQ
    assert s.llm_base_url == "https://api.groq.com/openai/v1"
    assert (s.llm_model_name, s.llm_fallback_name) == ("openai/gpt-oss-120b", "openai/gpt-oss-20b")


def test_the_models_can_be_overridden_and_a_fallback_never_equals_the_primary(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_MODEL="openai/gpt-oss-20b")
    assert s.llm_model_name == "openai/gpt-oss-20b" and s.llm_fallback_name is None  # the default fallback is now the primary
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_MODEL="m1", LLM_FALLBACK_MODEL="m2")
    assert (s.llm_model_name, s.llm_fallback_name) == ("m1", "m2")


def test_the_groq_key_is_never_used_for_another_providers_endpoint(tmp_path):
    other = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE="https://api.openai.com/v1", LLM_MODEL="gpt-x")
    assert other.llm_provider == "custom" and other.llm_key is None  # not sent to openai.com
    assert other.llm_fallback_name is None  # Groq's fallback model is meaningless elsewhere
    own = _load(tmp_path, LLM_API_BASE="https://api.openai.com/v1", LLM_MODEL="gpt-x", LLM_API_KEY="sk-own-key-1234567890")
    assert own.llm_key == "sk-own-key-1234567890" and own.llm_configured
    on_groq = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE="https://api.groq.com/openai/v1", LLM_MODEL="openai/gpt-oss-20b")
    assert on_groq.llm_provider == "groq" and on_groq.llm_key == GROQ


def test_an_explicit_llm_key_wins_over_the_groq_key(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_KEY="gsk_other_groq_key_0987654321")
    assert s.llm_key == "gsk_other_groq_key_0987654321" and s.llm_provider == "groq"


def test_a_typesafe_key_alone_enables_jev_and_never_the_llm(tmp_path):
    s = _load(tmp_path, TYPESAFE_API_KEY=TYPESAFE)
    assert s.jev_configured and not s.llm_configured


def test_the_new_keys_are_redacted_and_registered_for_log_scrubbing(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ, TYPESAFE_API_KEY=TYPESAFE)
    shown = s.redacted()
    assert shown["groq_api_key"] == "***set***" and shown["typesafe_api_key"] == "***set***"
    assert {GROQ, TYPESAFE} <= set(s.secrets())
    assert GROQ not in repr(s) and TYPESAFE not in repr(s) and GROQ not in str(shown) and TYPESAFE not in str(shown)
