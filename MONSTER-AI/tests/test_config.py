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
        ({"LLM_WEB_MAX_PER_RUN": "-1"}, "LLM_WEB_MAX_PER_RUN"),
        ({"LLM_WEB_MAX_PER_DAY": "-1"}, "LLM_WEB_MAX_PER_DAY"),
        ({"LLM_REASONING_EFFORT": "extreme"}, "LLM_REASONING_EFFORT"),
        ({"JEV_MIN_CONFIDENCE": "1.5"}, "JEV_MIN_CONFIDENCE"),
        ({"TRAINING_MODE": "external"}, "MODEL_ARTIFACT_URL"),
        ({"S3_BUCKET": "b"}, "S3_BUCKET"),
        ({"TELEGRAM_BOT_TOKEN": "not-a-token"}, "BotFather"),
        ({"MONSTER_TIMEZONE": "Mars/Base"}, "MONSTER_TIMEZONE"),
        ({"HISTORY_YEARS": "3", "TRAIN_YEARS": "5"}, "HISTORY_YEARS"),
        ({"IDEAS_ENABLED": "maybe"}, "IDEAS_ENABLED"),
        ({"IDEAS_REPORT_DEADLINE_TIME": "14:30"}, "IDEAS_REPORT_DEADLINE_TIME"),  # not after the 14:30 scan
        ({"IDEAS_REPORT_DEADLINE_TIME": "15:20"}, "IDEAS_REPORT_DEADLINE_TIME"),  # after the entry-by time
        ({"IDEAS_ENTRY_BY_TIME": "15:30"}, "IDEAS_ENTRY_BY_TIME"),  # not before the close
        ({"IDEAS_TOP_N": "0"}, "IDEAS_TOP_N"),
        ({"IDEAS_TOP_N": "11"}, "IDEAS_TOP_N"),
        ({"IDEAS_MIN_SCORE": "101"}, "IDEAS_MIN_SCORE"),
        ({"IDEAS_MIN_SCORE": "-1"}, "IDEAS_MIN_SCORE"),
        ({"IDEAS_MAX_RISK_PCT": "0"}, "IDEAS_MAX_RISK_PCT"),
        ({"IDEAS_MAX_RISK_PCT": "16"}, "IDEAS_MAX_RISK_PCT"),
    ],
)
def test_cross_field_validation(tmp_path, env, needle):
    with pytest.raises(ConfigError) as err:
        _load(tmp_path, **env)
    assert needle in str(err.value)


def test_the_trade_idea_settings_default_to_a_closing_hour_report_with_shorts_for_f_and_o(tmp_path):
    s = _load(tmp_path)
    assert s.ideas_enabled and s.send_ideas_report and s.ideas_include_shorts
    assert (s.ideas_report_deadline_time, s.ideas_entry_by_time) == ("15:05:00", "15:15:00")  # normalised to HH:MM:SS
    assert (s.ideas_top_n, s.ideas_min_score, s.ideas_max_risk_pct) == (5, 60.0, 4.0)
    off = _load(tmp_path, IDEAS_ENABLED="false", SEND_IDEAS_REPORT="no", IDEAS_INCLUDE_SHORTS="0", IDEAS_TOP_N="3")
    assert not off.ideas_enabled and not off.send_ideas_report and not off.ideas_include_shorts and off.ideas_top_n == 3


def test_the_ideas_deadline_may_equal_the_entry_by_time_and_the_error_names_the_pair(tmp_path):
    assert _load(tmp_path, IDEAS_REPORT_DEADLINE_TIME="15:15").ideas_report_deadline_time == "15:15:00"
    with pytest.raises(ConfigError) as err:
        _load(tmp_path, IDEAS_REPORT_DEADLINE_TIME="15:20")
    assert "IDEAS_ENTRY_BY_TIME" in str(err.value)


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


def test_a_groq_key_alone_gets_a_chain_of_three_models_each_with_its_own_allowance(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ)
    assert s.llm_fallback_models is None and s.llm_extra_fallback_names == ("qwen/qwen3.8-27b",)
    assert s.llm_model_chain == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


def test_the_further_models_are_set_in_order_and_none_switches_them_off(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_FALLBACK_MODELS="meta-models/muse-spark-open, qwen/qwen3.8-27b")
    assert s.llm_model_chain == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-models/muse-spark-open", "qwen/qwen3.8-27b"]
    assert _load(tmp_path, GROQ_API_KEY=GROQ, LLM_FALLBACK_MODELS="NONE").llm_model_chain == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert _load(tmp_path, GROQ_API_KEY=GROQ, LLM_FALLBACK_MODELS="").llm_extra_fallback_names == ("qwen/qwen3.8-27b",)  # empty = default
    # a name that is already in the chain is never repeated, however it got there
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_MODEL="qwen/qwen3.8-27b", LLM_FALLBACK_MODELS="qwen/qwen3.8-27b openai/gpt-oss-20b x/y x/y")
    assert s.llm_model_chain == ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "x/y"]


def test_another_provider_gets_only_the_models_it_is_told_about(tmp_path):
    base = {"LLM_API_BASE": "https://api.openai.com/v1", "LLM_MODEL": "gpt-x", "LLM_API_KEY": "sk-own-key-1234567890"}
    assert _load(tmp_path, **base).llm_model_chain == ["gpt-x"]  # Groq's models mean nothing elsewhere
    assert _load(tmp_path, **base, LLM_FALLBACK_MODELS="gpt-y,gpt-z").llm_model_chain == ["gpt-x", "gpt-y", "gpt-z"]
    assert _load(tmp_path).llm_model_chain == []  # nothing is configured without a key


@pytest.mark.parametrize("value", [",", "-x/y", "model;rm", "x/y $(id)", "a" * 101])
def test_a_malformed_model_list_is_refused_with_the_variable_named(tmp_path, value):
    with pytest.raises(ConfigError, match="LLM_FALLBACK_MODELS"):
        _load(tmp_path, GROQ_API_KEY=GROQ, LLM_FALLBACK_MODELS=value)


def test_a_model_list_may_be_separated_by_commas_or_spaces_and_tolerates_stray_commas(tmp_path):
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_FALLBACK_MODELS="a/b  c/d,,e/f,")
    assert s.llm_fallback_models == ("a/b", "c/d", "e/f")


def test_the_web_check_is_on_by_default_but_only_ever_exists_with_groq_and_a_gpt_oss_model(tmp_path):
    off = _load(tmp_path)
    assert (off.llm_web_search, off.llm_web_max_per_run, off.llm_web_max_per_day) == (True, 8, 16)
    assert (off.llm_web_configured, off.llm_web_models) == (False, [])
    on = _load(tmp_path, GROQ_API_KEY=GROQ)
    assert on.llm_web_configured and on.llm_web_models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert not _load(tmp_path, GROQ_API_KEY=GROQ, LLM_WEB_SEARCH="false").llm_web_configured
    assert not _load(tmp_path, GROQ_API_KEY=GROQ, LLM_WEB_MAX_PER_RUN="0").llm_web_configured
    assert not _load(tmp_path, GROQ_API_KEY=GROQ, LLM_WEB_MAX_PER_DAY="0").llm_web_configured
    assert _load(tmp_path, GROQ_API_KEY=GROQ, LLM_WEB_MAX_PER_RUN="25").llm_web_max_per_run == 25
    assert _load(tmp_path, GROQ_API_KEY=GROQ, LLM_WEB_MAX_PER_DAY="40").llm_web_max_per_day == 40
    other = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE="https://api.openai.com/v1", LLM_MODEL="gpt-x")
    assert other.llm_configured and not other.llm_web_configured and other.llm_web_models == []  # browser_search is Groq's
    partial = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_MODEL="qwen/qwen3.8-27b")
    assert partial.llm_web_models == ["openai/gpt-oss-20b"] and partial.llm_web_configured  # the fallback can still search
    no_search_models = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_MODEL="qwen/qwen3.8-27b", LLM_FALLBACK_MODEL="x/y", LLM_FALLBACK_MODELS="none")
    assert no_search_models.llm_web_models == [] and not no_search_models.llm_web_configured
    later = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_MODEL="qwen/qwen3.8-27b", LLM_FALLBACK_MODEL="x/y", LLM_FALLBACK_MODELS="openai/gpt-oss-20b")
    assert later.llm_web_models == ["openai/gpt-oss-20b"]  # a model further down the chain can search too


def test_the_groq_key_is_never_used_for_another_providers_endpoint(tmp_path):
    other = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE="https://api.openai.com/v1", LLM_MODEL="gpt-x")
    assert other.llm_provider == "custom" and other.llm_key is None  # not sent to openai.com
    assert other.llm_fallback_name is None  # Groq's fallback model is meaningless elsewhere
    own = _load(tmp_path, LLM_API_BASE="https://api.openai.com/v1", LLM_MODEL="gpt-x", LLM_API_KEY="sk-own-key-1234567890")
    assert own.llm_key == "sk-own-key-1234567890" and own.llm_configured
    on_groq = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE="https://api.groq.com/openai/v1", LLM_MODEL="openai/gpt-oss-20b")
    assert on_groq.llm_provider == "groq" and on_groq.llm_key == GROQ


@pytest.mark.parametrize("base", [
    "https://api.groq.com.evil.example/openai/v1",  # a look-alike host that merely starts with Groq's
    "https://evil.example/https://api.groq.com/openai/v1",  # Groq's host appears in the path
    "https://evil-api.groq.com.example/v1", "https://api.groq.co/openai/v1", "https://groq.com/openai/v1",
    "https://api.groq.com:8443/openai/v1", "https://api.groq.com#.evil.example/v1", "https://api.groq.com?host=evil.example",
])
def test_a_look_alike_of_groqs_host_never_receives_the_groq_key(tmp_path, base):
    s = _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE=base, LLM_MODEL="m")
    assert s.llm_provider == "custom" and s.llm_key is None


def test_plain_http_to_groqs_host_is_refused_and_never_gets_the_key(tmp_path):
    with pytest.raises(ConfigError, match="https"):
        _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE="http://api.groq.com/openai/v1", LLM_MODEL="m")


@pytest.mark.parametrize("base", [
    "https://api.groq.com@evil.example/openai/v1", "https://evil.example\\@api.groq.com/openai/v1",
    "https://api.groq.com\\.evil.example/v1", "https://api.groq.com /openai/v1", "https://user:pass@api.groq.com/openai/v1",
])
def test_a_url_a_parser_could_read_two_ways_is_refused_at_load(tmp_path, base):
    with pytest.raises(ConfigError, match="plain URL"):
        _load(tmp_path, GROQ_API_KEY=GROQ, LLM_API_BASE=base, LLM_MODEL="m")


@pytest.mark.parametrize("base", ["https://api.groq.com/openai/v1", "https://API.GROQ.COM/openai/v1/", "https://api.groq.com",
                                  "https://api.groq.com:443/openai/v1", "  https://api.groq.com/openai/v1  "])
def test_every_spelling_of_groqs_own_endpoint_is_recognised(base):
    from app.config import is_groq_url

    assert is_groq_url(base) is True


@pytest.mark.parametrize(("url", "loopback"), [
    ("http://localhost:11434/v1", True), ("http://127.0.0.1:8000/v1", True), ("http://[::1]:8000/v1", True),
    ("http://localhost.evil.example/v1", False), ("http://127.0.0.1.evil.example/v1", False),
    ("http://localhost@evil.example/v1", False), ("http://evil.example\\@localhost/v1", False), ("https://localhost/v1", False),
    ("http://evil.example/v1", False), ("not a url", False), ("http://[::1/v1", False),
])
def test_plain_http_is_allowed_for_loopback_hosts_only(url, loopback):
    from app.config import is_loopback_http

    assert is_loopback_http(url) is loopback


def test_a_local_model_server_over_http_is_accepted_and_a_remote_one_is_not(tmp_path):
    local = _load(tmp_path, LLM_API_BASE="http://localhost:11434/v1", LLM_MODEL="llama")
    assert local.llm_provider == "custom" and local.llm_configured
    with pytest.raises(ConfigError, match="https"):
        _load(tmp_path, LLM_API_BASE="http://localhost.evil.example/v1", LLM_MODEL="llama")


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
