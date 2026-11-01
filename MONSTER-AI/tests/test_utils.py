"""Canonical JSON helpers and the feature schema (compatibility hash, groups, stage-specific columns)."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import jsonutil
from features import schema


# ----------------------------------------------------------------------------------------------- jsonutil
@dataclasses.dataclass
class Point:
    x: float
    tag: str


class WithItem:
    def __init__(self, value):
        self._value = value

    def item(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def test_primitives_pass_through_and_non_finite_numbers_become_null():
    assert jsonutil.to_jsonable([None, True, "x", 3, 1.5]) == [None, True, "x", 3, 1.5]
    assert jsonutil.to_jsonable([math.nan, math.inf, -math.inf]) == [None, None, None]
    assert jsonutil.to_jsonable(Decimal("1.25")) == 1.25 and jsonutil.to_jsonable(Decimal("NaN")) is None


def test_dates_paths_containers_and_dataclasses_are_converted_deterministically():
    stamp = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
    out = jsonutil.to_jsonable({"when": stamp, "day": date(2026, 11, 2), "path": Path("a/b"), 7: (1, 2), "s": {"b", "a"},
                                "fs": frozenset({2, 1}), "dc": Point(1.0, "p")})
    assert out == {"when": "2026-11-02T03:00:00+00:00", "day": "2026-11-02", "path": str(Path("a/b")), "7": [1, 2],
                   "s": ["a", "b"], "fs": [1, 2], "dc": {"x": 1.0, "tag": "p"}}


def test_numpy_and_pandas_values_become_plain_json():
    out = jsonutil.to_jsonable({"f": np.float32(1.5), "i": np.int64(3), "b": np.bool_(True), "nan": np.float64("nan"),
                                "arr": np.array([[1, 2], [3, np.nan]]), "ts": pd.Timestamp("2026-11-02 09:00"),
                                "series": pd.Series({"a": 1.0, "b": np.nan}), "frame": pd.DataFrame({"c": [1, 2]}),
                                "index": pd.Index(["x", "y"])})
    assert out["f"] == 1.5 and out["i"] == 3 and out["b"] is True and out["nan"] is None
    assert out["arr"] == [[1, 2], [3, None]] and out["ts"] == "2026-11-02T09:00:00"
    assert out["series"] == {"a": 1.0, "b": None} and out["frame"] == {"c": {0: 1, 1: 2}} or out["frame"] == {"c": {"0": 1, "1": 2}}
    assert out["index"] == ["x", "y"]
    json.dumps(out, allow_nan=False)  # everything produced is strictly valid JSON


def test_unknown_objects_fall_back_to_item_or_their_text():
    assert jsonutil.to_jsonable(WithItem(5)) == 5
    assert "WithItem" in jsonutil.to_jsonable(WithItem(ValueError("x")))  # .item() fails: the text form is used
    assert jsonutil.to_jsonable(object).startswith("<class")
    assert jsonutil.to_jsonable(pd.NaT) is None and jsonutil.to_jsonable([pd.NaT]) == [None]
    assert jsonutil.to_jsonable(pd.Timedelta(seconds=5)) == "P0DT0H0M5S"  # durations become ISO 8601 text


def test_dumps_is_canonical_strict_and_unicode_safe():
    assert jsonutil.dumps({"b": 1, "a": [math.nan, "₹"]}) == '{"a":[null,"₹"],"b":1}'
    pretty = jsonutil.dumps({"b": 1, "a": 2}, pretty=True)
    assert pretty == '{\n  "a": 2,\n  "b": 1\n}'


def test_loads_never_raises_and_hashes_ignore_key_order():
    for empty in (None, "", b""):
        assert jsonutil.loads(empty, default={"d": 1}) == {"d": 1}
    assert jsonutil.loads("{broken", default=[]) == [] and jsonutil.loads(object(), default="x") == "x"
    assert jsonutil.loads('{"a": 1}') == {"a": 1}
    assert jsonutil.content_hash({"a": 1, "b": 2}) == jsonutil.content_hash({"b": 2, "a": 1})
    assert jsonutil.content_hash({"a": 1}) != jsonutil.content_hash({"a": 2}) and len(jsonutil.content_hash([])) == 64


# ----------------------------------------------------------------------------------------- feature schema
def test_sector_codes_are_stable_and_unknown_industries_share_one_code():
    assert schema.sector_code("Financial Services") == 0.0 and schema.sector_code("Information Technology") == 6.0
    assert schema.sector_code("Something New") == 99.0 and schema.sector_code(None) == 99.0 and schema.sector_code("") == 99.0
    assert len(set(schema.SECTOR_CODES.values())) == len(schema.SECTOR_CODES)


def test_feature_groups_follow_the_name_prefix():
    assert [schema.group_of(n) for n in ("t_ret1", "s_ret1", "m_trend", "g_us_ret", "f_iv", "c_dow", "p_gap_pct", "x_sim", "u_beta")] == [
        "technical", "sector", "regime", "global", "fno", "calendar", "premarket", "similar", "static"]
    assert schema.group_of("zz_unknown") == "other"


def test_the_preliminary_stage_never_sees_pre_open_features():
    columns = ["t_ret1", "p_gap_pct", "g_us_ret", "p_gap_atr"]
    assert schema.columns_for_stage(columns, "pre") == ["t_ret1", "g_us_ret"]
    assert schema.columns_for_stage(columns, "final") == columns and schema.columns_for_stage(columns, "final") is not columns


def test_the_schema_hash_changes_with_names_order_and_groups_but_not_between_calls():
    base = ["t_ret1", "g_us_ret", "p_gap_pct"]
    assert schema.schema_hash(base) == schema.schema_hash(list(base)) and len(schema.schema_hash(base)) == 16
    assert schema.schema_hash(base) != schema.schema_hash(base[::-1])
    assert schema.schema_hash(base) != schema.schema_hash(base[:-1])
    assert schema.schema_hash(["t_ret1"]) != schema.schema_hash(["x_ret1"])  # same suffix, different group
    assert schema.schema_hash([]) == schema.schema_hash([])
