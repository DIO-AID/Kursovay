"""Тесты ветки research/forecast: нет утечки будущего, сдвиги по времени, разбиение.
Запуск из diploma_fs: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.data import synth_factory                     # noqa: E402
from fsx.forecast import feature_sets, forward_splits  # noqa: E402
from fsx.lags import LagFE, LeakError, check_no_leak, describe   # noqa: E402


@pytest.fixture(scope="module")
def ds():
    d = synth_factory(n=2500)
    d.X["shift"] = np.where(d.t.dt.hour < 12, "A", "B")      # категориальная, не известна заранее
    d.X["plan"] = (d.t.dt.dayofweek < 5).astype(float)        # «расписание», известно заранее
    return d


@pytest.mark.parametrize("mode", ["forecast", "virtual"])
def test_no_leak(ds, mode):
    fe = LagFE("1h", mode, known_ahead=["plan"], cat_cols=["shift"]).fit(ds.X, ds.y, ds.t)
    assert fe.h_ == 4 and fe.day_ == 96
    assert check_no_leak(fe, ds.X, ds.y, ds.t, n_points=15) == 15


def test_leak_is_detected(ds):
    """Отрицательный контроль: признак «текущее y» тест обязан поймать."""
    class Leaky(LagFE):
        def transform(self, X, y, t):
            F = super().transform(X, y, t)
            F["target.bad__now"] = np.asarray(y, float)
            return F
    fe = Leaky("1h").fit(ds.X, ds.y, ds.t)
    with pytest.raises(LeakError):
        check_no_leak(fe, ds.X, ds.y, ds.t, n_points=5)


def test_forecast_mode_hides_current_sensors(ds):
    fe = LagFE("1h", "forecast", cat_cols=["shift"]).fit(ds.X, ds.y, ds.t)
    F = fe.transform(ds.X, ds.y, ds.t)
    assert not [c for c in F.columns if c.endswith("__cur") or "__curcat_" in c]
    assert "s01__lag4" in F.columns and "shift__lag4cat_A" in F.columns


def test_lag_by_time_not_by_row():
    """Пропуск в ряду: лаг должен указывать на момент t − k·шаг, а не на соседнюю строку."""
    t = pd.Series(pd.date_range("2024-01-01", periods=300, freq="1h"))
    t = t.drop(index=[100]).reset_index(drop=True)            # выпал один час
    y = np.arange(len(t), dtype=float)
    X = pd.DataFrame({"a": np.zeros(len(t))})
    F = LagFE(1).fit(X, y, t).transform(X, y, t)
    i = int(np.where(t == pd.Timestamp("2024-01-05 05:00"))[0][0])   # 101-й час -> строка 100
    assert np.isnan(F.loc[i, "target.short__lag1"])            # час назад данных нет
    assert F.loc[i, "target.short__lag2"] == y[i - 1]           # два часа назад — предыдущая строка


def test_duplicate_time_rejected():
    t = pd.Series(pd.to_datetime(["2024-01-01 00:00"] * 2 + ["2024-01-01 01:00"]))
    with pytest.raises(ValueError):
        LagFE(1).fit(pd.DataFrame({"a": [1.0, 2, 3]}), [1.0, 2, 3], t)


def test_splits_past_only():
    splits, gap = forward_splits(5000, h=4)
    assert gap >= 4 and len(splits) == 9
    for _, tr, te in splits:
        assert tr.max() < te.min() - gap


def test_feature_sets(ds):
    fe = LagFE("1h", known_ahead=["plan"], cat_cols=["shift"]).fit(ds.X, ds.y, ds.t)
    cols = list(fe.transform(ds.X, ds.y, ds.t).columns)
    s = feature_sets(fe, cols, "forecast")
    assert all(c.startswith("cal.") or c.startswith("plan__") for c in s["fs_no_lags"])
    assert "plan__now" in s["fs_no_lags"]
    for fam in ("calendar", "short", "daily", "rolling", "exog"):
        assert len(s[f"abl_no_{fam}"]) < len(cols)


# ---------- describe(): расшифровка признаков для отчёта (docs/REPORT_FORMAT.md) ----------

CTX = {"step": pd.Timedelta("15min"), "target": "Usage_kWh",
       "labels": {"Usage_kWh": ("Потребление", "кВт·ч", "цель"),
                  "NSM": ("Секунд от полуночи", "с", "известно заранее"),
                  "Load_Type": ("Тип нагрузки", "", "только прошлое")},
       "values": {"Load_Type": {"Light_Load": "лёгкая нагрузка"}}}


def test_describe_пример_из_формата():
    # ровно тот пример, что записан в docs/REPORT_FORMAT.md
    assert describe("target.daily__lag96", CTX) == "потребление сутки назад"


def test_describe_лаги_и_окна():
    assert describe("target.short__lag4", CTX) == "потребление 1 ч назад"
    assert describe("target.rolling__mean4", CTX) == "среднее «потребление» за 1 ч"
    assert describe("target.rolling__std96", CTX) == "разброс «потребление» за сутки"
    assert describe("target.daily__lag672", CTX) == "потребление неделю назад"


def test_describe_календарь_и_известное_заранее():
    assert describe("cal.hour__sin") == "синус часа суток"
    assert describe("NSM__now", CTX) == "секунд от полуночи (известно заранее)"


def test_describe_категории_расшифрованы():
    assert describe("Load_Type__lag4cat_Light_Load", CTX) == "тип нагрузки (1 ч назад): лёгкая нагрузка"


def test_describe_без_контекста_не_падает():
    for c in ("cal.hour__raw", "target.daily__lag96", "Temperature__lag6", "Whatever__now"):
        assert describe(c)                       # без ctx формулировка грубее, но текст есть


def test_describe_переводит_все_признаки(ds):
    """Ни один признак не должен остаться именем из кода — иначе отчёт не читается."""
    fe = LagFE("1h", known_ahead=["plan"], cat_cols=["shift"]).fit(ds.X, ds.y, ds.t)
    cols = fe.transform(ds.X, ds.y, ds.t).columns
    ctx = dict(CTX, step=fe.step_)
    left = [c for c in cols if describe(c, ctx) == c]
    assert not left, f"не расшифрованы: {left[:5]}"
