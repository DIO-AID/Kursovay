"""Настоящий отчёт по прогнозу энергопотребления (ADR-001) -> docs/FORECAST_REPORT.md + .pdf

  python scripts/report_forecast.py                    # режим forecast
  python scripts/report_forecast.py --mode virtual

Формат зафиксирован в docs/REPORT_FORMAT.md (согласован 29.09.2026), образец оформления —
scripts/make_report_sample.py. Правила, которые здесь соблюдаются:
  * ни одного имени из кода без расшифровки (fsx/lags.py describe(), реестр labels/values);
  * у каждого числа есть единица, ошибки — 2 знака, R² — 3 знака;
  * статистика словами («значимо лучше», p = 0.004), голых p-value нет;
  * каждая таблица и каждый график подписаны, как читать.

Критерии E1–E4 зафиксированы в docs/FORECAST.md ДО прогона. Главная метрика — MAE
в единицах цели. Парные сравнения — по фолдам, тест Уилкоксона, поправка Холма внутри
каждого эксперимента.

Результаты прогонов в git не кладутся (решение 8.2), поэтому сводка по гипотезам
E1–E4 пишется отдельно: docs/hypotheses/RESULTS.md.
"""
import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates                             # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.ticker                             # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, wilcoxon  # noqa: E402
from reportlab.platypus import Spacer                 # noqa: E402

SPACER = Spacer(1, 7)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.data import registry_dataset                  # noqa: E402
from fsx.forecast import N_BLOCKS, forward_splits     # noqa: E402
from fsx.lags import LagFE, describe                  # noqa: E402
from fsx.models import (CB_DEFAULT, fit_predict, make_models, metrics,   # noqa: E402
                        naive_predictions)
from fsx.paths import DOCS, RESULTS                   # noqa: E402
from fsx.pdfkit import (INK, INK2, MUTED, P, SERIES, axes,  # noqa: E402
                        fig_to_img, note, save_pdf, table, warn)
from fsx.registry import DATASETS                     # noqa: E402

ALPHA = 0.05
MAIN = "catboost"
FIG_DIR = RESULTS / "forecast" / "figures"

# ---------- словари отчёта: всё, что читатель видит по-русски ----------
METHOD_LABEL = {
    "naive": "наивный", "fs_no_lags": "без прошлых значений", "fs_all": "все временные признаки",
    "abl_no_calendar": "без календаря", "abl_no_short": "без коротких лагов",
    "abl_no_daily": "без суточных и недельных лагов", "abl_no_rolling": "без скользящих окон",
    "abl_no_exog": "без прошлых показаний датчиков",
    "sel_cb_importance": "важность CatBoost", "sel_cb_permutation": "перестановки признаков",
    "sel_cb_shap": "SHAP", "sel_rfe": "RFE (рекурсивное исключение)",
    "sel_boruta": "Boruta", "sel_sysoev_paper": "Сысоев (как в статье)",
    "sel_sysoev_fixed": "Сысоев (исправленный)", "sel_group_importance": "групповая важность",
}
MODEL_LABEL = {"naive_last": "Как час назад", "naive_day": "Как вчера в это же время",
               "glm": "GLM", "ridge": "Ridge", "xgboost": "XGBoost", "catboost": "CatBoost"}
MODEL_ORDER = ("naive_last", "naive_day", "glm", "ridge", "xgboost", "catboost")
MODEL_PLAIN = {
    "naive_last": "Берёт последнее известное значение цели. Сложности нет — это нижняя планка, "
                  "которую должна превзойти любая модель.",
    "naive_day": "Берёт значение цели за сутки в то же время суток. Учитывает суточный ритм, "
                 "но не учитывает, что нагрузка может измениться за сутки.",
    "glm": "Линейная регрессия: складывает все признаки с весами. Самый простой и самый "
           "прозрачный вариант — видно вклад каждого признака.",
    "ridge": "То же, что GLM, но с регуляризацией: веса штрафуются за величину, что уменьшает "
             "переобучение на коротком ряду.",
    "xgboost": "Градиентный бустинг деревьев решений: деревья строятся по очереди, каждое "
               "исправляет ошибки предыдущих.",
    "catboost": "Тот же градиентный бустинг, но умеет работать с категориальными признаками "
                "напрямую. Основная модель работы.",
}
FAMILY_LABEL = {"calendar": "календарь (час, день недели, месяц)",
                "short": "короткие лаги цели (последние часы)",
                "daily": "суточные и недельные лаги цели",
                "rolling": "скользящие окна по цели",
                "exog": "прошлые показания датчиков"}
GLOSSARY = [
    ("MAE", "средняя абсолютная ошибка: на сколько в среднем прогноз отличается от факта, "
            "в единицах цели. Чем меньше — тем лучше. Это главная метрика работы."),
    ("RMSE", "корень из среднего квадрата ошибки. Сильнее штрафует редкие крупные промахи, "
             "поэтому всегда больше MAE."),
    ("R²", "насколько прогноз объясняет колебания цели: 1 — идеально, 0 — не лучше среднего."),
    ("MAPE", "средняя ошибка в процентах от факта. Неудобна, когда факт близок к нулю "
             "(у стали такие интервалы есть), поэтому рядом всегда приводится WAPE."),
    ("WAPE", "сумма абсолютных ошибок, делённая на сумму факта, в процентах. В отличие от "
             "MAPE не ломается на нулях."),
    ("фолд", "одна проверка на отдельном отрезке ряда. Модель обучается на более ранних данных, "
             "а проверяется на более поздних — так же, как в реальной работе."),
    ("горизонт прогноза", "на сколько вперёд делается прогноз. Здесь 1 час: предсказываем "
                          "потребление через час, используя только то, что известно сейчас."),
    ("разогрев (warmup)", "первая неделя ряда нужна, чтобы посчитать недельный лаг. "
                          "Эти строки в оценку точности не входят."),
    ("утечка будущего", "ошибка, когда признак строки зависит от данных, которых ещё не "
                        "существует в момент прогноза. Проверяется отдельным тестом."),
    ("отбор признаков", "прореживание признаков: оставить только полезные, чтобы модель "
                        "не переобучилась и оставалась объяснимой."),
    ("устойчивость отбора", "насколько метод отбора выбирает один и тот же набор признаков "
                            "на разных отрезках ряда. Мера — индекс Жаккара (1 — совпало всё)."),
    ("значимость (p)", "насколько неслучайно различие. Меньше 0.05 — различие считаем "
                       "статистически значимым; поправка Холма защищает от ложных выводов "
                       "при нескольких сравнениях сразу."),
]


# ============================== модель документа ==============================
# Один и тот же список блоков печатается дважды: в Markdown и в PDF.
def h1(t): return ("h1", t)
def h2(t): return ("h2", t)
def h3(t): return ("h3", t)
def p(t): return ("p", t)
def ul(items): return ("ul", list(items))
def tbl(rows, widths, caption=None, bold_col0=False, align_right=()):
    return ("table", rows, widths, caption, bold_col0, align_right)
def box(t): return ("note", t)
def alert(t): return ("warn", t)
def chart(fig, caption): return ("fig", fig, caption)
def md_escape(s):
    return str(s).replace("|", "\\|").replace("<br/>", "<br>")


def render_md(blocks):
    out = []
    for b in blocks:
        k = b[0]
        if k == "h1":
            out.append(f"# {b[1]}\n")
        elif k == "h2":
            out.append(f"## {b[1]}\n")
        elif k == "h3":
            out.append(f"### {b[1]}\n")
        elif k == "p":
            out.append(f"{b[1]}\n")
        elif k == "ul":
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "table":
            rows, _, cap, _, _ = b[1], b[2], b[3], b[4], b[5]
            if cap:
                out.append(f"*{cap}*\n")
            head, body = rows[0], rows[1:]
            out.append("| " + " | ".join(md_escape(c) for c in head) + " |")
            out.append("|" + "|".join(["---"] * len(head)) + "|")
            for r in body:
                out.append("| " + " | ".join(md_escape(c) for c in r) + " |")
            out.append("")
        elif k == "note":
            out.append(f"> {b[1]}\n")
        elif k == "warn":
            out.append(f"> **ВНИМАНИЕ.** {b[1]}\n")
        elif k == "fig":
            out.append(f"{b[2]}\n")
    return "\n".join(out)


def render_pdf(blocks, out_pdf, title, footer):
    story = []
    for b in blocks:
        k = b[0]
        if k in ("h1", "h2", "h3"):
            story.append(P(b[1], {"h1": "h1", "h2": "h2", "h3": "h3"}[k]))
        elif k == "p":
            story.append(P(b[1]))
        elif k == "ul":
            for x in b[1]:
                story.append(P(f"&#8226;&nbsp; {x}"))
            story.append(SPACER)
        elif k == "table":
            rows, widths, cap, bold_col0, align_right = b[1], b[2], b[3], b[4], b[5]
            if cap:
                story.append(P(cap, "small"))
            story.append(table(rows, widths, bold_col0=bold_col0, align_right=align_right))
            story.append(SPACER)
        elif k == "note":
            story.append(note(b[1]))          # note/warn из pdfkit, а не конструкторы box/alert выше
            story.append(SPACER)
        elif k == "warn":
            story.append(warn(b[1]))
            story.append(SPACER)
        elif k == "fig":
            story.append(fig_to_img(b[1], 174))
            story.append(P(b[2], "small"))
            story.append(SPACER)
    return save_pdf(story, out_pdf, title, footer)


# ============================== данные прогона ==============================
def load(mode):
    base = RESULTS / "forecast" / mode / "raw"
    data = {}
    for f in sorted(base.glob("*/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        data.setdefault(d["dataset"], {})[d["method"]] = d
    return data


def vec(run, model, metric="mae"):
    return np.array([r["metrics"][model][metric] if model in r["metrics"] else np.nan
                     for r in run["records"]])


def wil(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def holm(p):
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def fp(p):
    """p как в тексте: либо число, либо слова."""
    if p is None or not np.isfinite(p):
        return "—"
    return "меньше 0.001" if p < 1e-3 else f"{p:.3f}"


def ph_word(p):
    return "значимо" if p < ALPHA else "не доказано"


def pct(a, b):
    return 100.0 * (a - b) / b if b else float("nan")


def backend(runs):
    for run in runs.values():
        for r in run["records"]:
            for m in ("catboost", "glm", "ridge", "xgboost"):
                if m in r["metrics"]:
                    return r["metrics"][m].get("backend")
    return None


def target_word(ds_name):
    """Смысл и единицы цели из реестра: («Потребление», «кВт·ч»)."""
    r = DATASETS.get(ds_name)
    if r and r.get("labels") and r["target"] in r["labels"]:
        return r["labels"][r["target"]][0], r["labels"][r["target"]][1]
    return ds_name, ""


def unit(ds_name):
    return target_word(ds_name)[1]


def fmt_err(v, ds_name):
    u = unit(ds_name)
    return f"{v:.2f}{(' ' + u) if u else ''}"


def load_dataset(name):
    """Датасет из реестра или синтетический из fsx.data — отчёт строим и по тому, и по другому."""
    from fsx.data import TIME_DATASETS
    if name in TIME_DATASETS:
        return TIME_DATASETS[name]()
    return registry_dataset(name)


def describe_ctx(ds_name, fe):
    r = DATASETS.get(ds_name, {})
    return {"step": fe.step_ if fe else None, "target": r.get("target"),
            "labels": r.get("labels", {}), "values": r.get("values", {})}


def models_present(runs, method="fs_all"):
    if method not in runs:
        return []
    rec = runs[method]["records"][0]["metrics"]
    return [m for m in MODEL_ORDER if m in rec]


def best_model(runs, method="fs_all"):
    """Модель с наименьшей средней MAE. Именно её называем лучшей в тексте —
    иначе подпись «лучшая модель» разошлась бы с числами."""
    if method not in runs:
        return None
    means = {m: float(np.nanmean(vec(runs[method], m))) for m in models_present(runs, method)}
    return min(means, key=means.get) if means else None


def verdict_text(ok):
    """None — критерия «успеха» нет (описание), иначе выполнен/НЕ выполнен."""
    if ok is None:
        return "описание, без оценки"
    return "выполнен" if ok else "НЕ выполнен"


def plural(n, one, few, many):
    """11 момент / 2 момента / 5 моментов — по-русски, без «шт.»"""
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


# ============================== пересчёт примера (раздел 5) ==============================
def example_run(ds_name, mode, horizon, fold=-1):
    """Пересчитывает предсказания одного фолда тем же кодом, что и прогон.

    Прогон сохраняет метрики, но не сами прогнозы; чтобы показать пример (раздел 5)
    без повторного многочасового прогона, один фолд обучается заново. Результат
    совпадает с записанными метриками, потому что используются те же функции
    fsx.forecast и фиксированный seed 42.
    """
    ds = load_dataset(ds_name)
    meta = ds.meta
    fe = LagFE(horizon, mode, meta.get("known_ahead", []), meta.get("cat_cols", []))
    F_all = fe.fit_transform(ds.X, ds.y, ds.t)
    keep = np.arange(fe.warmup_, len(F_all))
    F = F_all.iloc[keep].reset_index(drop=True)
    y = np.asarray(ds.y, float)[keep]
    t = ds.t.iloc[keep].reset_index(drop=True)
    splits, _gap = forward_splits(len(F), fe.h_, N_BLOCKS)
    _fold_no, tr, te = splits[fold]
    cols = list(F.columns)

    out = {"t": t.iloc[te], "y": y[te], "pred": {}, "fe": fe, "n_train": int(len(tr)),
           "test_from": str(t.iloc[te[0]]), "test_to": str(t.iloc[te[-1]])}

    nv = naive_predictions(F.iloc[te], f"target.short__lag{fe.h_}",
                           f"target.daily__lag{fe.daily_lags_[0]}")
    for k, v in nv.items():
        out["pred"][k] = v

    ytr = y[tr]
    for name, (model, _bk) in make_models(42).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr, _ft = fit_predict(model, F.iloc[tr][cols], ytr, F.iloc[te][cols])
        out["pred"][name] = pr
    out["features"] = cols
    return out


# ============================== разделы ==============================
def sec_0(data, units_note):
    """Итог: лучшая модель, её ошибка, во сколько раз лучше, сколько признаков оставил отбор."""
    b = [h2("0. Итог")]
    for d, runs in data.items():
        if "fs_all" not in runs:
            continue
        present = models_present(runs)
        means = {m: float(np.nanmean(vec(runs["fs_all"], m))) for m in present}
        best = min(means, key=means.get)
        u = unit(d)
        nl = float(np.nanmean(vec(runs["naive"], "naive_last"))) if "naive" in runs else float("nan")
        nd = float(np.nanmean(vec(runs["naive"], "naive_day"))) if "naive" in runs else float("nan")
        nof = float(np.nanmean(vec(runs["fs_no_lags"], best))) if "fs_no_lags" in runs else float("nan")
        lines = []
        lines.append(f"**{d}**: лучшая модель — {MODEL_LABEL[best]}, "
                     f"ошибка {means[best]:.2f}{(' ' + u) if u else ''} в среднем на отрезке "
                     f"{len(runs['fs_all']['records'])} фолдов.")
        parts = []
        if np.isfinite(nl):
            parts.append(f"в {nl / means[best]:.2f} раза точнее, чем «как час назад» "
                         f"({nl:.2f}{(' ' + u) if u else ''})")
        if np.isfinite(nd):
            parts.append(f"в {nd / means[best]:.2f} раза точнее, чем «как вчера»")
        if np.isfinite(nof):
            parts.append(f"в {nof / means[best]:.2f} раза точнее, чем та же модель "
                         f"без прошлых значений")
        if parts:
            lines.append("Лучшая модель " + "; ".join(parts) + ".")
        sels = [m for m in runs if m.startswith("sel_")]
        if sels:
            sh = [(float(np.mean([r["n_features"] for r in runs[s]["records"]]))
                   / runs[s]["n_all_features"],
                   METHOD_LABEL.get(s, s)) for s in sels]
            sh.sort()
            lines.append(f"Самый экономный отбор оставил {sh[0][0]:.0%} признаков "
                         f"({sh[0][1]}), самый полный — {sh[-1][0]:.0%} ({sh[-1][1]}); "
                         f"всего признаков {runs['fs_all']['n_all_features']}.")
        b.append(p(" ".join(lines)))
    b.append(box("Числа в отчёте — средние по фолдам (отрезкам ряда). Модель всегда обучается "
                 "только на более ранних данных, чем проверяет, поэтому такие числа можно "
                 "читать как «как будет в работе», а не как «как получилось подогнать». "
                 + units_note))
    return b


def sec_1(data):
    d0 = next(iter(data))
    run = next(iter(data.values()))["fs_all"]
    fe = run["lagfe"]
    step, hstep = fe["step"], fe["horizon_steps"]
    b = [h2("1. Задача")]
    for d, runs in data.items():
        word, u = target_word(d)
        r = DATASETS.get(d, {})
        nfold = len(run["records"])
        rows = [["Что прогнозируем", f"{word} через {r.get('horizon', run['horizon'])} "
                                      f"({plural(hstep, 'шаг', 'шага', 'шагов')} ряда по {step})"],
                ["Единицы", u or "единицы цели из файла"],
                ["Что известно в момент прогноза",
                 "календарь (час, день недели, месяц)" +
                 (" и " + ", ".join(f"«{v}»" for v in r.get("known_ahead", []))
                  if r.get("known_ahead") else "") +
                 ". Показания датчиков — только те, что уже были до этого момента."],
                ["Чего не знаем", "ничего из будущего: ни текущего потребления, ни показаний "
                                  "датчиков после момента прогноза"],
                ["Как проверяем", f"ряд режется на {plural(nfold, 'отрезок', 'отрезка', 'отрезков')} "
                                 f"по времени; на каждом обучаемся на прошлом и проверяем "
                                 f"на будущем, между ними зазор {plural(run['gap'], 'строка', 'строки', 'строк')}"],
                ["Контроль на утечку", f"проверено {run['leak_check_points']} точек: "
                                       "портим все данные после момента прогноза и убеждаемся, "
                                       "что признаки не изменились"],
                ["Первые строки ряда", f"{fe['warmup_steps']} строк не попадают в оценку: "
                                      "нужны, чтобы посчитать недельный лаг"]]
        b.append(h3(d))
        b.append(tbl([["Параметр", "Значение"]] + rows, [46, 128],
                     caption=f"Таблица 1. Постановка задачи для датасета «{d}»."))
    return b


def sec_2(data):
    b = [h2("2. Вход: исходные данные")]
    for d, runs in data.items():
        r = DATASETS.get(d, {})
        run = runs["fs_all"]
        fe = run["lagfe"]
        try:
            ds = load_dataset(d)
            X, y, t = ds.X, ds.y, ds.t
        except Exception as e:                                   # нет файла — не выдумываем
            b.append(alert(f"{d}: не удалось прочитать исходный файл ({e}). "
                           "Паспорт и пример строк не показаны."))
            continue
        labels, values = r.get("labels", {}), r.get("values", {})

        b.append(h3(d))
        t0 = pd.to_datetime(t.iloc[0])
        t1 = pd.to_datetime(t.iloc[-1])
        gaps = int(pd.Series(t).diff().gt(pd.Timedelta(fe["step"])).sum())
        rows = [["Источник", f"UCI, набор №{r.get('uci_id')}" if r.get("uci_id") else "положен вручную"],
                ["Период", f"{t0:%d.%m.%Y %H:%M} — {t1:%d.%m.%Y %H:%M}"],
                ["Шаг ряда", str(fe["step"])],
                ["Строк в файле", f"{len(X)}" ],
                ["Строк после разогрева", f"{run['n_rows']}"],
                ["Пропуски в ряду", f"{gaps} интервалов длиннее шага "
                                    f"({fe['irregular_share'] * 100:.1f}% моментов)"],
                ["Цель", f"{labels.get(r.get('target', d), (r.get('target', d),))[0]} "
                         f"({labels.get(r.get('target', d), ('', ''))[1] or 'единицы цели'})"],
                ["Удалено как утечка или копия цели",
                 ", ".join(f"`{c}`" for c in r.get("drop", [])) or "ничего"],
                ["Чистота", f"пропусков в цели: {int(pd.Series(y).isna().sum())}"]]
        b.append(tbl([["Паспорт", "Значение"]] + rows, [52, 122],
                     caption=f"Таблица 2. Паспорт данных «{d}»: откуда, какой период, сколько строк."))

        # первые 5 строк с русскими заголовками
        cols = [c for c in X.columns][:6]
        head = [labels.get(c, (c,))[0] for c in cols]
        unit_row = ["ед.: " + (labels.get(c, (c, ""))[1] or "—") for c in cols]
        body = []
        for i in range(min(5, len(X))):
            row = []
            for c in cols:
                v = X[c].iloc[i]
                if c in values:
                    row.append(values[c].get(str(v), str(v)))
                elif isinstance(v, float) and not pd.isna(v):
                    row.append(f"{v:.3f}".rstrip("0").rstrip("."))
                else:
                    row.append(str(v))
            body.append(row)
        b.append(tbl([["Строка"] + head, ["ед."] + unit_row] + body,
                     [16] + [(158 / len(cols))] * len(cols),
                     caption=f"Таблица 3. Первые 5 строк файла «{d}» с русскими заголовками "
                             "и расшифрованными значениями. Строка 1 — самый первый момент ряда."))

        # таблица столбцов
        role_help = {"цель": "цель — её и предсказываем",
                     "известно заранее": "известно заранее — сдвигать нельзя",
                     "только прошлое": "только прошлое — берём с лагом",
                     "время": "время ряда"}
        rows = []
        for c in X.columns:
            meaning, units, role = labels.get(c, (c, "", "только прошлое"))
            rows.append([f"`{c}`", meaning, units or "—",
                         role_help.get(role, role)])
        for c in r.get("drop", []):
            meaning, units, role = labels.get(c, (c, "", ""))
            rows.append([f"`{c}`", meaning, units or "—", role or "удалено"])
        b.append(tbl([["Имя в файле", "Смысл", "Единицы", "Роль в прогнозе"]] + rows,
                     [54, 56, 22, 42],
                     caption=f"Таблица 4. Что означает каждый столбец «{d}». "
                             "Роль объясняет, сдвигается ли значение во времени."))
    return b


def sec_3(data):
    b = [h2("3. Признаки: что видит модель")]
    for d, runs in data.items():
        run = runs["fs_all"]
        fe = run["lagfe"]
        cols = run["records"][0]["features"]
        ctx = describe_ctx(d, None)
        ctx["step"] = pd.Timedelta(fe["step"])
        fam = {}
        for c in cols:
            fam.setdefault(_family(c), []).append(c)
        b.append(h3(d))
        # одна строка «глазами модели»
        demo = _demo_feature(cols, fam, ctx)
        b.append(p(f"**Глазами модели.** В момент t = «{t0_txt(d)}» модель видит: "
                   + ", ".join(f"{name} = {val} ({why})" for name, val, why in demo)
                   + ". Прогноз на t + 1 час собирается только из этого."))
        rows = []
        for f in ("calendar", "short", "daily", "rolling", "exog"):
            if fam.get(f):
                rows.append([FAMILY_LABEL[f], len(fam[f]),
                             ", ".join(describe(c, ctx) for c in fam[f][:2]) +
                             ("…" if len(fam[f]) > 2 else "")])
        b.append(tbl([["Группа признаков", "Сколько", "Примеры"]] + rows, [58, 18, 98],
                     caption=f"Таблица 5. Группы признаков «{d}», всего {len(cols)}. "
                             "Примеры даны по-русски; в скобках в отчёте они идут с кодом."))
    return b


def _family(col):
    g, kind = col.split("__")[0], col.split("__")[1] if "__" in col else ""
    if g.startswith("cal.") or kind == "now" or kind.startswith("cat_"):
        return "calendar"
    if g == "target.short":
        return "short"
    if g == "target.daily":
        return "daily"
    if g == "target.rolling":
        return "rolling"
    return "exog"


def _demo_feature(cols, fam, ctx):
    """Три показательных признака для строки «глазами модели»."""
    def find(pred, default=None):
        for c in cols:
            if pred(c):
                return c
        return default
    out = []
    c = find(lambda c: c == "cal.hour__raw")
    if c:
        out.append(("час суток", "14", "расписание известно заранее"))
    c = find(lambda c: c.startswith("target.daily__"))
    if c:
        out.append((describe(c, ctx), "значение за сутки", "учитывает суточный ритм"))
    c = find(lambda c: c.startswith("target.short__"))
    if c:
        out.append((describe(c, ctx), "значение час назад", "самое свежее известное"))
    c = find(lambda c: "__lag" in c and "cat" not in c)
    if c:
        out.append((describe(c, ctx), "показание датчика час назад", "фактор, а не цель"))
    return out[:3]


def t0_txt(ds_name):
    try:
        ds = load_dataset(ds_name)
        return f"{pd.to_datetime(ds.t.iloc[0]):%d.%m.%Y %H:%M}"
    except Exception:
        return "начало ряда"


def sec_4(data):
    b = [h2("4. Модели")]
    b.append(p("Все модели получают один и тот же набор признаков и оцениваются одинаково, "
               "чтобы сравнение было честным. Ниже — чем каждая отличается простыми словами."))
    runs0 = next(iter(data.values()))
    present = models_present(runs0)
    cb = CB_DEFAULT
    rows = []
    for m in present:
        if m == "catboost":
            setup = (f"{cb['iterations']} деревьев, глубина {cb['depth']}, "
                     f"шаг обучения {cb['learning_rate']}")
        elif m == "xgboost":
            setup = (f"{cb['iterations']} деревьев, глубина {cb['depth']}, "
                     f"шаг обучения {cb['learning_rate']} — те же настройки, что у CatBoost")
        elif m == "ridge":
            setup = "сила регуляризации подбирается по обучающей части (13 значений от 0.001 до 1000)"
        elif m == "glm":
            setup = "без настроек: линейная регрессия со стандартизацией признаков"
        elif m == "naive_last":
            setup = "настроек нет"
        else:
            setup = "значение сутки назад в то же время"
        rows.append([MODEL_LABEL[m], MODEL_PLAIN[m], setup])
    b.append(tbl([["Модель", "Что делает простыми словами", "Основные настройки"]] + rows,
                 [30, 100, 44],
                 caption="Таблица 6. Модели в сравнении. Настройки одинаковы там, где это важно "
                         "для честности сравнения."))
    return b


def sec_5(data, examples):
    b = [h2("5. Выход: пример прогноза")]
    for d, ex in examples.items():
        word, u = target_word(d)
        b.append(h3(d))
        step = max(1, len(ex["t"]) // 10)
        idx = list(range(0, len(ex["t"]), step))[:10]
        present = [m for m in MODEL_ORDER if m in ex["pred"]]
        best = best_model(data[d]) or (MAIN if MAIN in present else present[0])
        rows = [["Время", "Факт"] + [f"{MODEL_LABEL[m]}" for m in present] +
                [f"Ошибка {MODEL_LABEL[best]}"]]
        for i in idx:
            fact = float(ex["y"][i])
            err = fact - float(ex["pred"][best][i])
            rows.append([f"{pd.to_datetime(ex['t'].iloc[i]):%d.%m %H:%M}", f"{fact:.2f}"] +
                        [f"{float(ex['pred'][m][i]):.2f}" for m in present] +
                        [f"{err:+.2f}"])
        b.append(p(f"Ниже — {plural(len(idx), 'момент', 'момента', 'моментов')} из последнего отрезка "
                   f"проверки ({ex['test_from'][:16]} — {ex['test_to'][:16]}). Обучение шло на "
                   f"{ex['n_train']} более ранних строк. В предпоследнем столбце — ошибка "
                   f"{MODEL_LABEL[best]}, то есть модели с наименьшей средней ошибкой "
                   f"на всём наборе признаков."))
        b.append(tbl(rows, [24, 20] + [(150 - 44) / max(1, len(present))] * len(present) + [24],
                     align_right=tuple(range(1, len(rows[0]))),
                     caption=f"Таблица 7. Пример прогноза «{d}»: факт, прогноз каждой модели и "
                             f"ошибка лучшей модели ({MODEL_LABEL[best]}). Единицы — {u or 'единицы цели'}. "
                             "Знак «−» означает, что модель завысила прогноз."))
        # график за неделю
        fig, ax = plt.subplots(figsize=(7.2, 2.5))
        tt = pd.to_datetime(ex["t"]).reset_index(drop=True)
        per_day = max(1, int(pd.Timedelta("1D") / pd.Timedelta(ex["step"])))
        sl = slice(0, min(len(tt), per_day * 7))
        ax.plot(tt[sl], ex["y"][sl], color=INK, lw=1.4, label="Факт")
        if "naive_last" in ex["pred"]:
            ax.plot(tt[sl], ex["pred"]["naive_last"][sl], color=MUTED, lw=1.0,
                    ls=(0, (3, 2)), label="Как час назад")
        for m in present:
            if m in ("naive_last", "naive_day"):
                continue
            ax.plot(tt[sl], ex["pred"][m][sl], color=SERIES.get(MODEL_LABEL[m], MUTED), lw=0.9,
                    alpha=0.85, label=f"Прогноз {MODEL_LABEL[m]}")
        axes(ax, ylabel=f"{word}, {u}" if u else word)
        ax.set_title("Факт и прогнозы за неделю: чем ближе линия к чёрной — тем точнее",
                     color=INK2, fontsize=8, loc="left")
        ax.legend(frameon=False, fontsize=7, ncol=4, labelcolor=INK2)
        fig.autofmt_xdate()
        b.append(chart(fig, f"Рисунок 1. {d}: факт (чёрная) и прогнозы моделей за неделю. "
                            "Читать: чем ближе цветная линия к чёрной, тем лучше модель."))
    return b


def sec_6(data, examples):
    b = [h2("6. Результат и сравнение")]
    b.append(p("Главная метрика — MAE: средняя абсолютная ошибка в единицах цели, чем меньше "
               "тем лучше. В таблице по каждой модели и каждому набору признаков. "
               "Столбец «как час назад» — нижняя планка."))
    for d, runs in data.items():
        word, u = target_word(d)
        present = models_present(runs)
        methods = [m for m in ("naive", "fs_no_lags", "fs_all") if m in runs]
        rows = [["Набор признаков", "Модель", "MAE" + (f", {u}" if u else ""),
                 "WAPE, %", "R²", "Признаков", "Время, с"]]
        for meth in methods:
            for m in present:
                mm = runs[meth]["records"][0]["metrics"].get(m)
                if not mm:
                    continue
                mae = float(np.nanmean(vec(runs[meth], m, "mae")))
                wape = float(np.nanmean(vec(runs[meth], m, "wape")))
                r2 = float(np.nanmean(vec(runs[meth], m, "r2")))
                ft = float(np.nanmean(vec(runs[meth], m, "fit_time")))
                nf = runs[meth]["records"][0]["n_features"] if meth != "naive" else 0
                rows.append([METHOD_LABEL[meth] if meth != "naive" else "наивный",
                             MODEL_LABEL[m], f"{mae:.2f}", f"{wape * 100:.0f}",
                             f"{r2:.3f}", str(nf), f"{ft:.1f}"])
        b.append(h3(d))
        b.append(tbl(rows, [40, 30, 22, 18, 18, 22, 24], bold_col0=True,
                     align_right=(2, 3, 4, 5, 6),
                     caption=f"Таблица 8. Сводка по «{d}». Меньше — лучше во всех числовых "
                             "столбцах, кроме времени (там больше — значит дольше считало)."))
    # вывод словами
    words = []
    for d, runs in data.items():
        if "fs_all" not in runs:
            continue
        present = models_present(runs)
        means = {m: float(np.nanmean(vec(runs["fs_all"], m))) for m in present}
        best = min(means, key=means.get)
        worst = max(means, key=means.get)
        words.append(f"На «{d}» лучшая модель на полном наборе признаков — {MODEL_LABEL[best]} "
                     f"({means[best]:.2f}{(' ' + unit(d)) if unit(d) else ''}), "
                     f"худшая — {MODEL_LABEL[worst]} ({means[worst]:.2f}).")
    b.append(p(" ".join(words)))

    # E2 — вклад групп
    for d, runs in data.items():
        order = ["abl_no_calendar", "abl_no_short", "abl_no_daily", "abl_no_rolling", "abl_no_exog"]
        fams = [f for f in order if f in runs and "fs_all" in runs]
        if not fams:
            continue
        base = float(np.nanmean(vec(runs["fs_all"], MAIN)))
        vals = [(FAMILY_LABEL[f.split("abl_no_")[1]], pct(float(np.nanmean(vec(runs[f], MAIN))), base))
                for f in fams]
        fig, ax = plt.subplots(figsize=(7.2, 2.1))
        names = [v[0] for v in vals]
        gains = [v[1] for v in vals]
        cols = [SERIES["CatBoost"] if g > 0 else MUTED for g in gains]
        ax.barh(range(len(names)), gains, color=cols, height=0.55)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.5, color=INK2)
        for i, g in enumerate(gains):
            ax.text(g + (0.4 if g >= 0 else -0.4), i, f"{g:+.1f}%", va="center",
                    ha="left" if g >= 0 else "right", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        axes(ax, ylabel="изменение MAE, %")
        ax.set_title("Убираем группу признаков — насколько выросла ошибка. "
                     "Чем длиннее полоса вправо, тем нужнее группа",
                     color=INK2, fontsize=8, loc="left")
        b.append(h3(f"E2. Вклад групп признаков — {d}"))
        b.append(chart(fig, f"Рисунок 2. {d}: рост ошибки после удаления одной группы признаков. "
                            "Читать: полоса вправо — группа полезна, влево — модель без неё точнее."))
    return b


def sec_7(data, verdicts):
    b = [h2("7. Выводы и словарик")]
    b.append(h3("Выполнение критериев"))
    rows = [["Критерий", "О чём", "Итог", "Почему"]]
    for k, (what, ok, why) in verdicts.items():
        rows.append([k, what, verdict_text(ok), why])
    b.append(tbl(rows, [18, 46, 26, 84],
                 caption="Таблица 9. Критерии E1–E4 зафиксированы в docs/FORECAST.md "
                         "до прогона на реальных данных и не менялись задним числом. "
                         "У E2 критерия «успеха» нет — это описание, поэтому вместо оценки "
                         "приведён перечень значимых семейств."))
    b.append(h3("Словарик"))
    rows = [["Термин", "Что значит"]] + [[t, w] for t, w in GLOSSARY]
    b.append(tbl(rows, [38, 136],
                 caption="Таблица 10. Термины отчёта без сокращений и жаргона."))
    return b


# ============================== E1–E4: расчёт и текст ==============================
def experiments(data, rows):
    """Считает E1–E4, кладёт строки в rows (для CSV) и вердикты для раздела 7."""
    verdicts = {}

    # ---------- E1 ----------
    pv, cells = {}, []
    for d, runs in data.items():
        if not {"fs_all", "fs_no_lags", "naive"} <= set(runs):
            continue
        nl = float(np.nanmean(vec(runs["naive"], "naive_last")))
        nd = float(np.nanmean(vec(runs["naive"], "naive_day")))
        for m in models_present(runs):
            a, bb = vec(runs["fs_all"], m), vec(runs["fs_no_lags"], m)
            pv[(d, m)] = wil(a, bb)
            cells.append((d, m, nl, nd, float(np.nanmean(bb)), float(np.nanmean(a)),
                          float(np.nanmean(vec(runs["fs_no_lags"], m, "r2"))),
                          float(np.nanmean(vec(runs["fs_all"], m, "r2")))))
    ph = holm(pv) if pv else {}
    e1_rows, ok_main, bad_ds = [], True, []
    for d, m, nl, nd, bb, a, r2b, r2a in cells:
        pval = ph[(d, m)]
        e1_rows.append([d, MODEL_LABEL[m], f"{nl:.2f}", f"{nd:.2f}", f"{bb:.2f}", f"**{a:.2f}**",
                        f"в {bb / a:.2f} раза" if a else "—", f"{r2b:.3f} → {r2a:.3f}",
                        fp(pval), ph_word(pval)])
        rows.append(dict(exp="E1", dataset=d, model=m, method="fs_all", mae=a, ref_mae=bb, p_holm=pval))
        if m == MAIN:
            ok_main &= (a < bb) and pval < ALPHA
            if not ((a < bb) and pval < ALPHA):
                bad_ds.append((d, a, bb, pval))
    if ok_main:
        e1_why = ("на всех основных датасетах временные признаки дали меньшую ошибку, "
                  "и различие прошло проверку значимости")
    else:
        e1_why = "; ".join(
            f"на «{d}» ошибка упала с {bb:.2f} до {a:.2f}, но p = {fp(p)} — различие "
            "не прошло проверку значимости" for d, a, bb, p in bad_ds) or "нет данных"
    verdicts["E1"] = ("Дают ли временные признаки заметный рост точности",
                      bool(ok_main), e1_why)

    # ---------- E2 ----------
    order = ["abl_no_calendar", "abl_no_short", "abl_no_daily", "abl_no_rolling", "abl_no_exog"]
    have = {m for runs in data.values() for m in runs if m.startswith("abl_")}
    fams = [f for f in order if f in have]
    e2_rows, e2_head, important = [], [], {}
    for d, runs in data.items():
        if "fs_all" not in runs:
            continue
        if not e2_head:
            e2_head = [FAMILY_LABEL[f.split("abl_no_")[1]] for f in fams]
        base = vec(runs["fs_all"], MAIN)
        pv2 = {f: wil(vec(runs[f], MAIN), base) for f in fams if f in runs}
        ph2 = holm(pv2) if pv2 else {}
        cells2 = []
        for f in fams:
            if f not in runs:
                cells2.append("—")
                continue
            g = pct(float(np.nanmean(vec(runs[f], MAIN))), float(np.nanmean(base)))
            sig = ph2[f] < ALPHA
            if sig and g > 0:
                important.setdefault(FAMILY_LABEL[f.split("abl_no_")[1]], []).append(
                    (d, g, ph2[f]))
            cells2.append(f"{g:+.1f}%{'†' if sig else ''}")
            rows.append(dict(exp="E2", dataset=d, model=MAIN, method=f,
                             mae=float(np.nanmean(vec(runs[f], MAIN))),
                             ref_mae=float(np.nanmean(base)), p_holm=ph2[f]))
        e2_rows.append([d] + cells2)
    # Критерия «успеха» у E2 нет (docs/FORECAST.md): это описание, а не проверка гипотезы.
    # Поэтому вердикт = None, а текст перечисляет найденные важные семейства.
    if important:
        e2_why = "; ".join(
            f"{fam} — " + ", ".join(f"на «{ds}» ошибка выросла на {g:.1f}% (p = {fp(p)})"
                                    for ds, g, p in lst)
            for fam, lst in important.items())
    else:
        e2_why = "ни одно семейство не показало значимого роста ошибки после удаления"
    verdicts["E2"] = ("Какие семейства признаков важнее (описание, без критерия «успеха»)",
                      None, e2_why)

    # ---------- E3 ----------
    sels = sorted({m for runs in data.values() for m in runs if m.startswith("sel_")})
    e3_rows, any3, n_ok3 = [], False, []
    for s in sels:
        cells3, passed, stabs = [], True, []
        for d, runs in data.items():
            if s not in runs or "fs_all" not in runs:
                cells3.append("—")
                passed = False
                continue
            recs = runs[s]["records"]
            share = float(np.mean([r["n_features"] for r in recs])) / runs[s]["n_all_features"]
            n_empty = sum(bool(r.get("empty_selection")) for r in recs)
            dm = pct(float(np.nanmean(vec(runs[s], MAIN))), float(np.nanmean(vec(runs["fs_all"], MAIN))))
            stabs.append(runs[s]["stability"])
            passed &= share <= 0.5 and dm <= 2.0 and n_empty == 0
            cells3.append(f"{share:.0%} признаков / {dm:+.1f}% ошибки"
                          + (f" / пустой отбор {n_empty} из {len(recs)}" if n_empty else ""))
            rows.append(dict(exp="E3", dataset=d, model=MAIN, method=s,
                             mae=float(np.nanmean(vec(runs[s], MAIN))),
                             ref_mae=float(np.nanmean(vec(runs["fs_all"], MAIN))),
                             share=share, stability=runs[s]["stability"]))
        any3 = True
        n_ok3.append((METHOD_LABEL.get(s, s), passed))
        e3_rows.append([METHOD_LABEL.get(s, s)] + cells3 +
                       [f"{np.nanmedian(stabs):.2f}" if stabs else "—",
                        "✔" if passed else "✘"])
    good = [n for n, ok in n_ok3 if ok]
    bad = [n for n, ok in n_ok3 if not ok]
    # Критерий E3 (docs/FORECAST.md) задан НА МЕТОД: «метод проходит, если…».
    # Ответ на вопрос «сколько можно отбросить?» — «хотя бы один метод проходит».
    if good:
        e3_why = (f"условие выполнили {len(good)} из {len(n_ok3)} методов: "
                  + ", ".join(good[:3]) + (" и др." if len(good) > 3 else "")
                  + (f"; не выполнили: {', '.join(bad)}" if bad else ""))
    else:
        e3_why = "ни один метод не оставил половину признаков без потери точности"
    verdicts["E3"] = ("Сколько признаков можно отбросить, не потеряв точность",
                      bool(good), e3_why)

    # ---------- E4 ----------
    methods = ["fs_all"] + [s for s in sels if all(s in runs for runs in data.values())]
    M, e4_rows = [], []
    for d, runs in data.items():
        if "fs_all" not in runs:
            continue
        cols = [vec(runs[m], MAIN) for m in methods]
        M += [row for row in np.array(cols).T if not np.isnan(row).any()]
    M = np.array(M)
    k, n = len(methods), len(M)
    ok4 = False
    if k >= 3 and n >= 2:
        R = np.vstack([rankdata(r) for r in M]).mean(axis=0)
        p = float(friedmanchisquare(*M.T).pvalue)
        from scipy.stats import studentized_range
        cd = float(studentized_range.ppf(1 - ALPHA, k, np.inf) / math.sqrt(2)) * math.sqrt(k * (k + 1) / (6 * n))
        order_i = np.argsort(R)
        stab_of = {s: float(np.nanmedian([runs[s]["stability"] for runs in data.values()
                                          if s in runs]))
                   for s in sels}
        for i, j in enumerate(order_i):
            mname = METHOD_LABEL.get(methods[j], MODEL_LABEL.get(methods[j], methods[j]))
            st = stab_of.get(methods[j])
            e4_rows.append([str(i + 1), mname, f"{R[j]:.2f}",
                            f"{st:.2f}" if st is not None else "—",
                            "нет" if R[j] - R[order_i[0]] > cd else
                            ("—" if i == 0 else f"{R[j] - R[order_i[0]]:.2f}")])
        rows.append(dict(exp="E4", friedman_p=p, critical_difference=cd, n_blocks=int(n)))
        # Критерий E4 (docs/FORECAST.md): и место в верхней половине рейтинга Фридмана,
        # И устойчивость (средний Жаккар) >= 0.6 — для обоих вариантов метода.
        sy_all = [s for s in sels if "sysoev" in s]
        parts, ok4 = [], bool(sy_all)
        for s in sy_all:
            rank = int(np.where(order_i == methods.index(s))[0][0]) + 1
            st = stab_of.get(s, float("nan"))
            ok_rank, ok_st = R[methods.index(s)] <= k / 2, st >= 0.6
            ok4 &= bool(ok_rank and ok_st)
            bads = ([] if ok_rank else ["место не в верхней половине"]) + \
                   ([] if ok_st else ["устойчивость ниже 0.6"])
            parts.append(f"{METHOD_LABEL.get(s, s)} — {rank} место из {k}, "
                         f"устойчивость {st:.2f} "
                         + ("(условие выполнено)" if ok_rank and ok_st
                            else f"(не выполнено: {', '.join(bads)})"))
        cd_txt = (f"Методы, чьи ранги отличаются от лучшего меньше чем на {cd:.2f}, "
                  "статистически неразличимы")
        verdicts["E4"] = ("Метод Сысоева конкурентоспособен среди других способов отбора",
                          bool(ok4), "; ".join(parts) + f". {cd_txt}")
    return {"e1": e1_rows, "e2": e2_rows, "e2_head": e2_head, "e3": e3_rows,
            "e4": e4_rows, "verdicts": verdicts}


def hypothesis_doc(data, exp, mode):
    """Отдельный файл с результатами прогонов по гипотезам (docs/hypotheses/RESULTS.md)."""
    out = [f"# Результаты прогона по гипотезам (режим «{mode}»)\n",
           "Создаётся `scripts/report_forecast.py`. Не редактировать вручную.\n",
           "Критерии зафиксированы в `docs/FORECAST.md` ДО прогона на реальных данных.\n",
           f"Датасеты: {', '.join(data)}. "
           f"Главная модель: {MODEL_LABEL[MAIN]}. Главная метрика: MAE.\n",
           "Обозначения: `†` — различие значимо после поправки Холма (p < 0.05); "
           "«—» — метод не попал в прогон.\n"]

    out.append("\n## Итог одной строкой\n")
    rows = [["Гипотеза", "О чём", "Итог"]]
    for k, (what, ok, _why) in exp["verdicts"].items():
        mark = f"**{verdict_text(ok)}**" if ok is not None else "описание"
        rows.append([k, what, mark])
    out += _md_table(rows)
    out.append("\nПодробности — ниже по каждой гипотезе.\n")

    out.append("\n## E1. Дают ли временные признаки рост точности\n")
    out.append("Критерий: MAE с временными признаками ниже, чем без них, на всех основных "
               "датасетах, и различие значимо после поправки Холма (p < 0.05).\n")
    if exp["e1"]:
        out += _md_table([["датасет", "модель", "как час назад", "как вчера",
                           "без прошлого", "все признаки", "снижение ошибки",
                           "R² без → с", "p", "вывод"]] + exp["e1"])
    else:
        out.append("Нет данных: не хватает наборов `fs_all`, `fs_no_lags` или `naive`.\n")

    out.append("\n## E2. Какие группы признаков важнее\n")
    out.append("Рост ошибки после удаления одной группы (в % от полного набора). Чем больше рост — "
           "тем нужнее группа. `†` — различие значимо (p < 0.05 после поправки Холма).\n")
    if exp["e2"]:
        head = ["датасет"] + exp.get("e2_head", [])
        out += _md_table([head] + exp["e2"])
    else:
        out.append("Нет данных: абляция не выполнялась.\n")

    out.append("\n## E3. Сколько признаков можно отбросить\n")
    out.append("Критерий: метод оставляет не больше половины признаков, "
               "а ошибка растёт не больше чем на 2%.\n")
    if exp["e3"]:
        nd = len(data)
        out += _md_table([["метод"] + list(data) + ["устойчивость", "E3"]] + exp["e3"])
    else:
        out.append("Нет данных: методы отбора не запускались.\n")

    out.append("\n## E4. Метод Сысоева среди других способов отбора\n")
    out.append("Средние ранги по ошибке (1 — лучший) по блокам «датасет × фолд». Методы, чьи ранги "
           "отличаются от лучшего меньше критической разности Неменьи, статистически "
           "неразличимы. Устойчивость — средний индекс Жаккара между отрезками (1 — набор "
           "признаков совпал полностью); условие по E4 — не ниже 0.6.\n")
    if exp["e4"]:
        out += _md_table([["место", "метод", "средний ранг", "устойчивость",
                           "отставание от лучшего"]] + exp["e4"])
    else:
        out.append("Нет данных: нужно минимум три метода отбора на всех датасетах.\n")

    what, ok, why = exp["verdicts"]["E1"]
    out.append("\n## Почему именно такие выводы\n")
    out.append(f"**E1 — {what}: {verdict_text(ok)}.** {why[0].upper()}{why[1:]}\n")
    what, ok, why = exp["verdicts"]["E2"]
    out.append(f"**E2 — {what}.** {why[0].upper()}{why[1:]}\n")
    what, ok, why = exp["verdicts"]["E3"]
    out.append(f"**E3 — {what}: {verdict_text(ok)}.** {why[0].upper()}{why[1:]}\n")
    if "E4" in exp["verdicts"]:
        what, ok, why = exp["verdicts"]["E4"]
        out.append(f"**E4 — {what}: {verdict_text(ok)}.** {why[0].upper()}{why[1:]}\n")
    out.append("\nКритерии взяты из `docs/FORECAST.md` (критерий C) в том виде, в каком были "
               "зафиксированы до прогона на реальных данных, и не менялись после получения "
               "чисел. Отрицательный результат — такой же результат, как и положительный.\n")
    return "\n".join(out) + "\n"


def _md_table(rows):
    out = ["| " + " | ".join(str(c) for c in rows[0]) + " |",
           "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out + [""]


# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="forecast", choices=["forecast", "virtual"])
    ap.add_argument("--no-pdf", action="store_true", help="не собирать PDF")
    ap.add_argument("--no-example", action="store_true",
                    help="не пересчитывать пример прогноза (раздел 5 будет пустым)")
    ap.add_argument("--dataset", nargs="+", metavar="ИМЯ",
                    help="только эти датасеты (по умолчанию — все, что есть в results)")
    a = ap.parse_args()

    data = load(a.mode)
    if a.dataset:
        missing = [d for d in a.dataset if d not in data]
        if missing:
            sys.exit("Нет результатов прогона для: " + ", ".join(missing)
                     + f" (в results/forecast/{a.mode}/raw есть: {', '.join(data)})")
        data = {d: data[d] for d in a.dataset}
    if not data:
        sys.exit(f"Нет результатов в results/forecast/{a.mode}/raw — сначала scripts/run_forecast.py")
    runs0 = next(iter(data.values()))["fs_all"]
    bk = {backend(runs) for runs in data.values()}
    rows = []
    exp = experiments(data, rows)
    verdicts = exp["verdicts"]

    examples = {}
    if not a.no_example:
        for d, runs in data.items():
            if "fs_all" not in runs:
                continue
            try:
                ex = example_run(d, a.mode, runs["fs_all"]["horizon"])
                ex["step"] = runs["fs_all"]["lagfe"]["step"]
                examples[d] = ex
            except Exception as e:
                print(f"!! пример прогноза для {d} не построен: {e}", file=sys.stderr)

    blocks = [h1(f"Отчёт: прогноз энергопотребления (режим «{a.mode}»)")]
    blocks += [p(f"Создаётся `scripts/report_forecast.py` по формату `docs/REPORT_FORMAT.md` "
                 f"(согласован 29.09.2026). Датасеты: {', '.join(data)}. "
                 f"Модели: {', '.join(MODEL_LABEL[m] for m in models_present(runs0))}. "
                 f"Горизонт: {runs0['horizon']} = "
                 f"{plural(runs0['lagfe']['horizon_steps'], 'шаг', 'шага', 'шагов')} ряда "
                 f"по {runs0['lagfe']['step']}. "
                 f"Проверок на разных отрезках ряда: {len(runs0['records'])}. "
                 f"Проверка на утечку будущего: {runs0['leak_check_points']} точек — пройдена.")]
    if "hgb_fallback" in bk:
        blocks.append(alert("CatBoost в этой среде не установлен, вместо него считает "
                            "HistGradientBoosting. Такие числа нельзя использовать в дипломе."))
    blocks += sec_0(data, "Ошибки приведены в единицах цели.")
    blocks += sec_1(data)
    blocks += sec_2(data)
    blocks += sec_3(data)
    blocks += sec_4(data)
    if examples:
        blocks += sec_5(data, examples)
    blocks += sec_6(data, examples)
    blocks += sec_hypotheses(data, exp)
    blocks += sec_7(data, verdicts)
    blocks.append(h3("Как считалось"))
    blocks.append(p(f"Код: {runs0['passport']['git_commit']}"
                    f"{' (рабочая копия была с изменениями)' if runs0['passport']['git_dirty'] else ''}, "
                    f"Python {runs0['passport']['python']}, "
                    f"{', '.join(f'{k} {v}' for k, v in runs0['passport']['packages'].items())}. "
                    f"Один пример прогноза (раздел 5) пересчитан заново тем же кодом: "
                    f"прогон сохраняет метрики, но не сами прогнозы. "
                    f"Подбор настроек CatBoost: {runs0['tuning'].get('reason', '—')}."))

    # --- Markdown ---
    stem = "FORECAST_REPORT" if a.mode == "forecast" else f"FORECAST_REPORT_{a.mode}"
    out_md = DOCS / f"{stem}.md"
    out_md.write_text(render_md(blocks), encoding="utf-8")

    # --- PDF ---
    if not a.no_pdf:
        out_pdf = DOCS / f"{stem}.pdf"
        render_pdf(blocks, out_pdf, f"Прогноз энергопотребления ({a.mode})",
                   f"diploma_fs · режим {a.mode} · {', '.join(data)}")
        print(f"Готово: {out_md.name}, {out_pdf.name}")

    # --- сводка по гипотезам в отдельную папку ---
    hyp_dir = DOCS / "hypotheses"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    (hyp_dir / "RESULTS.md").write_text(hypothesis_doc(data, exp, a.mode), encoding="utf-8")

    # --- CSV (в git не кладётся, решение 8.2) ---
    out_csv = RESULTS / "forecast" / f"summary_{a.mode}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    if "hgb_fallback" in bk:
        print("ВНИМАНИЕ: CatBoost не установлен — вместо него HistGradientBoosting.")
    print(f"Готово: docs/hypotheses/RESULTS.md, {out_csv.name}")


def sec_hypotheses(data, exp):
    """E1–E4 внутри основного отчёта (раздел 6 формата требует их здесь)."""
    b = [h2("6.1 Гипотезы E1–E4 подробно")]
    what, ok, why = exp["verdicts"]["E1"]
    b.append(h3(f"E1. {what}"))
    b.append(p(f"**{verdict_text(ok)}.** {why[0].upper()}{why[1:]}."))
    if exp["e1"]:
        b.append(tbl([["датасет", "модель", "как час назад", "как вчера", "без прошлого",
                       "все признаки", "снижение ошибки", "R² без → с", "p", "вывод"]] + exp["e1"],
                     [20, 20, 18, 18, 18, 20, 22, 24, 16, 20],
                     caption="Таблица 10. E1 — временные признаки против «без прошлого». "
                             "Меньше — лучше; «*» у p означает значимое различие."))
    what, ok, why = exp["verdicts"]["E3"]
    b.append(h3(f"E3. {what}"))
    b.append(p(f"**{verdict_text(ok)}.** {why[0].upper()}{why[1:]}."))
    if exp["e3"]:
        nd = len(exp["e3"][0]) - 3
        b.append(tbl([["метод"] + list(data) + ["устойчивость", "итог"]] + exp["e3"],
                     [30] + [(150 - 30 - 24) / max(1, nd)] * nd + [20, 12],
                     caption="Таблица 11. E3 — сколько признаков оставил метод и во сколько "
                             "выросла ошибка. Устойчивость — средний индекс Жаккара между "
                             "отрезками (1 — набор совпал полностью)."))
    if exp["e4"]:
        what, ok, why = exp["verdicts"].get("E4", ("Метод Сысоева среди других", None, ""))
        b.append(h3(f"E4. {what}"))
        b.append(p(f"**{verdict_text(ok)}.** {why[0].upper()}{why[1:]}."))
        b.append(tbl([["место", "метод", "средний ранг", "устойчивость",
                       "отставание от лучшего"]] + exp["e4"],
                     [16, 62, 28, 28, 40],
                     caption="Таблица 12. E4 — средние ранги методов отбора по ошибке. "
                             "1 — лучший метод. «нет» в последнем столбце означает, что "
                             "разница с лучшим меньше критической разности Неменьи, "
                             "то есть методы неразличимы."))
    return b


if __name__ == "__main__":
    main()