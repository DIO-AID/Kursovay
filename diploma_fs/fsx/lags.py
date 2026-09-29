"""Шаг 3 (ветка research/forecast): временные признаки для прогноза ряда — LagFE.

Главное правило — НЕТ УТЕЧКИ БУДУЩЕГО. Прогноз на момент t делается в момент t − h
(h — горизонт). Поэтому каждый признак строки t вычисляется только из данных с временем
<= t − h. Исключения — то, что известно заранее: календарь (час, день недели, месяц) и
колонки known_ahead из реестра (расписание, NSM, выходной/рабочий).
Тест: tests/test_lags.py портит все данные после t − h и проверяет, что признаки строки t
не изменились.

Сдвиги делаются ПО ВРЕМЕНИ, а не по номеру строки: если в ряду есть пропуск, лаг
«сутки назад» указывает на нужный момент (или NaN), а не на чужую строку.

Режимы:
  forecast — основной: все факторы, кроме known_ahead, берутся с лагом >= h;
  virtual  — «виртуальный датчик»: текущие показания факторов известны
             (лаги самой цели всё равно >= h).

Имена признаков: "<группа>__<вид>". Группа (то, что до "__") нужна методам отбора Сысоева:
  target.short__lag4     короткие лаги цели: h, h+1, h+2, h+3 шага
  target.daily__lag96    сезонные лаги: сутки, двое суток, неделя (кратные суткам и >= h)
  target.rolling__mean4  скользящие среднее/СКО цели по окну, которое кончается в t − h
  cal.hour__sin          календарь (известен заранее)
  NSM__now, Day_of_week__cat_Monday   известные заранее колонки
  Temperature__lag6      прочие факторы: в forecast — с лагом h (+ скользящее среднее)
Семейства (для абляции E2): calendar, short, daily, rolling, exog.

Ничего не «учится» на данных: только сдвиги и окна. Словарь категорий берётся по всему ряду —
это список возможных значений (дни недели, типы нагрузки), а не информация о цели.
"""
import numpy as np
import pandas as pd

from .transforms import SEP

FAMILIES = ("calendar", "short", "daily", "rolling", "exog")
TARGET = "target"


class LeakError(ValueError):
    """Нарушено условие «признак в t зависит только от данных до t − h»."""


def infer_step(t):
    """Шаг ряда = медиана разностей времени. Возвращает (шаг, доля нерегулярных интервалов)."""
    d = pd.Series(t).diff().dropna()
    if len(d) == 0:
        raise ValueError("Ряд из одной точки")
    step = d.median()
    if step <= pd.Timedelta(0):
        raise ValueError("Время не возрастает (есть дубликаты моментов?)")
    return step, float((d != step).mean())


class LagFE:
    """Строит временные признаки для всего ряда сразу (только сдвиги — утечки через fit нет).

    horizon    — горизонт: pd.Timedelta, строка ("1h") или число шагов (int)
    mode       — "forecast" | "virtual"
    known_ahead— колонки X, известные заранее (не сдвигаются)
    cat_cols   — категориальные колонки X (кодируются 0/1)
    short_lags — сколько коротких лагов цели (h, h+1, ...)
    """

    def __init__(self, horizon=1, mode="forecast", known_ahead=(), cat_cols=(), short_lags=4):
        if mode not in ("forecast", "virtual"):
            raise ValueError("mode: forecast | virtual")
        self.horizon, self.mode = horizon, mode
        self.known_ahead, self.cat_cols = tuple(known_ahead), tuple(cat_cols)
        self.short_lags = short_lags

    # ---------- параметры ряда ----------
    def fit(self, X, y, t):
        t = pd.to_datetime(pd.Series(t)).reset_index(drop=True)
        if t.duplicated().any():
            ex = list(t[t.duplicated()].astype(str).head(3))
            raise ValueError(f"В ряду повторяются моменты времени: {ex}")
        self.step_, self.irregular_ = infer_step(t)
        if isinstance(self.horizon, (int, np.integer)):
            self.h_ = int(self.horizon)
        else:
            self.h_ = max(1, int(round(pd.Timedelta(self.horizon) / self.step_)))
        self.day_ = max(1, int(round(pd.Timedelta("1D") / self.step_)))
        d = self.day_
        # сезонные лаги: кратные суткам и >= h, плюс неделя
        m0 = int(np.ceil(self.h_ / d))
        self.daily_lags_ = sorted({m0 * d, (m0 + 1) * d} | ({7 * d} if 7 * d >= self.h_ else set()))
        self.windows_ = sorted({max(2, d // 24), d})          # ~1 час и сутки
        self.warmup_ = self.h_ + max(max(self.daily_lags_), max(self.windows_))
        miss = [c for c in self.known_ahead + self.cat_cols if c not in X.columns]
        if miss:
            raise ValueError(f"Нет колонок {miss} в данных")
        self.num_ = [c for c in X.columns if c not in self.cat_cols
                     and pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
        self.cat_ = [c for c in X.columns if c not in self.num_]
        self.levels_ = {c: sorted(X[c].astype(str).unique()) for c in self.cat_}
        return self

    # ---------- построение ----------
    def _lag(self, s, k):
        """Значение ряда s (индекс — время) на момент t − k·шаг; нет такого момента -> NaN."""
        return s.shift(freq=k * self.step_).reindex(s.index)

    def _roll(self, s, w, how):
        r = s.rolling(w * self.step_, closed="right", min_periods=max(1, w // 2))
        return self._lag(getattr(r, how)(), self.h_)

    def transform(self, X, y, t):
        t = pd.to_datetime(pd.Series(t)).reset_index(drop=True)
        idx = pd.DatetimeIndex(t)
        X = X.reset_index(drop=True)
        ys = pd.Series(np.asarray(y, dtype=float), index=idx)
        h, cols = self.h_, {}

        # календарь — известен заранее
        hour = idx.hour + idx.minute / 60
        dow = idx.dayofweek
        cols["cal.hour__raw"] = hour
        cols["cal.hour__sin"] = np.sin(2 * np.pi * hour / 24)
        cols["cal.hour__cos"] = np.cos(2 * np.pi * hour / 24)
        cols["cal.dow__raw"] = dow
        cols["cal.dow__sin"] = np.sin(2 * np.pi * dow / 7)
        cols["cal.dow__cos"] = np.cos(2 * np.pi * dow / 7)
        cols["cal.weekend__raw"] = (dow >= 5).astype(float)
        cols["cal.month__raw"] = idx.month

        # цель: короткие, сезонные, скользящие
        for k in range(h, h + self.short_lags):
            cols[f"{TARGET}.short__lag{k}"] = self._lag(ys, k)
        for k in self.daily_lags_:
            cols[f"{TARGET}.daily__lag{k}"] = self._lag(ys, k)
        for w in self.windows_:
            cols[f"{TARGET}.rolling__mean{w}"] = self._roll(ys, w, "mean")
            cols[f"{TARGET}.rolling__std{w}"] = self._roll(ys, w, "std")

        # прочие факторы
        w_short = self.windows_[0]
        for c in self.num_:
            s = pd.Series(X[c].to_numpy(dtype=float), index=idx)
            if c in self.known_ahead:
                cols[f"{c}__now"] = s
                continue
            if self.mode == "virtual":
                cols[f"{c}__cur"] = s
            cols[f"{c}__lag{h}"] = self._lag(s, h)
            cols[f"{c}__mean{w_short}"] = self._roll(s, w_short, "mean")
        for c in self.cat_:
            s = pd.Series(X[c].astype(str).to_numpy(), index=idx)
            if c in self.known_ahead:
                suffix, src = "cat_", s
            elif self.mode == "virtual":
                suffix, src = "curcat_", s
            else:
                suffix, src = f"lag{h}cat_", s.shift(freq=h * self.step_).reindex(idx)
            for v in self.levels_[c]:
                cols[f"{c}__{suffix}{v}"] = (src == v).astype(float).where(src.notna())

        F = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()})
        return F

    def fit_transform(self, X, y, t):
        return self.fit(X, y, t).transform(X, y, t)

    # ---------- описание признаков ----------
    def groups(self, columns):
        g = {}
        for c in columns:
            g.setdefault(c.split(SEP)[0], []).append(c)
        return g

    def families(self, columns):
        fam = {f: [] for f in FAMILIES}
        for c in columns:
            fam[self.family(c)].append(c)
        return fam

    def family(self, col):
        g, kind = col.split(SEP)[0], col.split(SEP)[1]
        if g.startswith("cal.") or kind == "now" or kind.startswith("cat_"):
            return "calendar"
        if g == f"{TARGET}.short":
            return "short"
        if g == f"{TARGET}.daily":
            return "daily"
        if g == f"{TARGET}.rolling":
            return "rolling"
        return "exog"

    def info(self):
        return {"step": str(self.step_), "irregular_share": self.irregular_, "horizon_steps": self.h_,
                "steps_per_day": self.day_, "daily_lags": self.daily_lags_, "windows": self.windows_,
                "warmup_steps": self.warmup_, "mode": self.mode, "known_ahead": list(self.known_ahead)}


def check_no_leak(fe, X, y, t, n_points=20, seed=0, atol=1e-9):
    """Портит все НЕизвестные заранее данные после t − h и проверяет, что признаки строки t
    не изменились. Возвращает число проверенных точек; при нарушении — LeakError."""
    rng = np.random.default_rng(seed)
    t = pd.to_datetime(pd.Series(t)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    F0 = fe.transform(X, y, t)
    lo = min(len(t) - 1, fe.warmup_)
    points = rng.choice(np.arange(lo, len(t)), size=min(n_points, len(t) - lo), replace=False)
    spoil = [c for c in X.columns if c not in fe.known_ahead]
    if fe.mode == "virtual":
        spoil = []                                   # текущие факторы разрешены, портим только цель
    for i in points:
        cut = t[i] - fe.h_ * fe.step_
        after = (t > cut).to_numpy()
        y2 = y.copy()
        y2[after] = y2[after] * 7.0 + 1e3 + rng.normal(size=after.sum())
        X2 = X.copy()
        for c in spoil:
            if c in fe.num_:
                X2.loc[after, c] = X2.loc[after, c].astype(float) * 5.0 + 123.0
            else:
                lv = fe.levels_[c]
                X2.loc[after, c] = rng.choice(lv, size=after.sum())
        F2 = fe.transform(X2, y2, t)
        a, b = F0.iloc[i].to_numpy(), F2.iloc[i].to_numpy()
        same = np.isclose(a, b, atol=atol, equal_nan=True)
        if not same.all():
            bad = list(F0.columns[~same])[:5]
            raise LeakError(f"Строка {i} ({t[i]}): признаки {bad} зависят от данных после t − h")
    return len(points)
