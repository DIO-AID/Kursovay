"""Отчёт по гипотезам о настройке Optuna -> docs/HPO_REPORT_<имя>.md и results/hpo/<имя>/summary.csv

  python scripts/report_hpo.py --experiment base
  python scripts/report_hpo.py --experiment h_trials
  python scripts/report_hpo.py --experiment h_early_stopping

Все ошибки — MAE на БУДУЩЕМ отрезке (тест), в единицах цели. «Стандартная» — модель с параметрами библиотеки
по умолчанию. Критерии гипотез записаны в docs/hypotheses/*.md до прогона.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsx.paths import DOCS, RESULTS  # noqa: E402

MODEL = {"hgb": "sklearn-бустинг (HGB)", "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}
ES = {"shared": "общая выборка (как в базе)", "separate": "отдельная выборка", "none": "без остановки"}
SCHEME = {"random": "случайные 20% строк", "time": "последние 20% по времени"}


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def load(name):
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RESULTS / "hpo" / name).glob("*/*.json"))]
    if not runs:
        sys.exit(f"Нет результатов в results/hpo/{name} — сначала scripts/run_hpo.py --experiment {name}")
    return runs


def rows_of(runs):
    """Одна строка = (датасет, модель, остановка, фолд, бюджет)."""
    rows = []
    for r in runs:
        v = np.array([t["val_mae"] for t in r["trials"]])
        te = np.array([t["test_mae"] for t in r["trials"]])
        rho = float(spearmanr(v, te).correlation) if len(v) > 3 else float("nan")
        for b in r["budgets"]:
            rows.append(dict(dataset=r["dataset"], model=r["model"], es_mode=r["es_mode"], fold=r["fold"],
                             budget=b["budget"], default=r["default"]["test_mae"], tuned=b["test_mae"],
                             val=b["val_mae"], vs_default=pct(b["test_mae"], r["default"]["test_mae"]),
                             vs_default_refit=pct(b.get("refit_test_mae", b["test_mae"]), r["default"]["test_mae"]),
                             optimism=pct(b["test_mae"], b["val_mae"]),
                             regret=pct(b["test_mae"], b["oracle_test_mae"]), rho=rho, n_iter=b["n_iter"],
                             naive=r["reference"]["naive_last"], ridge=r["reference"]["ridge"],
                             time=r["study_time"] * b["budget"] / max(len(r["trials"]), 1), sampler=r["sampler"]))
    return rows


def group(rows, keys):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def main_table(rows, md):
    md += ["| датасет | модель | остановка | попыток | стандартная | настроенная | изменение | хуже стандартной | "
           "изменение при обучении на всех данных | "
           "ошибка на проверке | будущее против проверки | связь проверка ↔ будущее | потеря к лучшей попытке | время |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (d, m, e, b), rs in sorted(group(rows, ("dataset", "model", "es_mode", "budget")).items()):
        worse = sum(r["vs_default"] > 0 for r in rs)
        md.append(f"| {d} | {MODEL.get(m, m)} | {ES.get(e, e)} | {b} | {med([r['default'] for r in rs]):.2f} | "
                  f"{med([r['tuned'] for r in rs]):.2f} | {med([r['vs_default'] for r in rs]):+.1f}% | "
                  f"{worse} из {len(rs)} | {med([r['vs_default_refit'] for r in rs]):+.1f}% | "
                  f"{med([r['val'] for r in rs]):.2f} | {med([r['optimism'] for r in rs]):+.0f}% | "
                  f"{med([r['rho'] for r in rs]):.2f} | {med([r['regret'] for r in rs]):+.1f}% | "
                  f"{med([r['time'] for r in rs]):.0f} с |")
    md += ["", "Как читать: «изменение» — ошибка настроенной модели к стандартной на будущем отрезке (минус — лучше); "
               "«настроенная» — модель лучшей попытки как в базовом проекте (обучена на 80% обучающих строк); "
               "«изменение при обучении на всех данных» — те же настройки, но обучение на всех обучающих строках, "
               "как у стандартной (честное сравнение настроек, а не объёма данных); "
               "«будущее против проверки» — на сколько реальная ошибка выше той, что видела Optuna (большой плюс — самообман); "
               "«связь» — от −1 до 1: насколько оценка на проверке предсказывает ошибку на будущем; "
               "«потеря» — на сколько выбранная попытка хуже лучшей из сделанных. Все числа — медианы по отрезкам времени.", ""]


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 6 or np.allclose(a, b):
        return None
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return None


def words(p):
    if p is None:
        return "слишком мало наблюдений для проверки значимости"
    return f"различие значимо (p = {p:.3f})" if p < 0.05 else f"различие не доказано (p = {p:.3f})"


def verdict_trials(rows, md):
    budgets = sorted({r["budget"] for r in rows})
    if len(budgets) < 2:
        return
    lo = 25 if 25 in budgets else budgets[0]
    hi = budgets[-1]
    by = group(rows, ("dataset", "model", "es_mode", "fold"))
    dv, dt, o_lo, o_hi = [], [], [], []
    for rs in by.values():
        a = {r["budget"]: r for r in rs}
        if lo in a and hi in a:
            dv.append(pct(a[hi]["val"], a[lo]["val"]))
            dt.append(pct(a[hi]["tuned"], a[lo]["tuned"]))
            o_lo.append(a[budgets[0]]["optimism"])
            o_hi.append(a[hi]["optimism"])
    md += ["## Гипотеза «Число попыток»", "",
           "| попыток | изменение к стандартной | хуже стандартной | при обучении на всех данных | "
           "будущее против проверки |", "|---|---|---|---|---|"]
    for b in budgets:
        rs = [r for r in rows if r["budget"] == b]
        md.append(f"| {b} | {med([r['vs_default'] for r in rs]):+.1f}% | "
                  f"{sum(r['vs_default'] > 0 for r in rs)} из {len(rs)} | "
                  f"{med([r['vs_default_refit'] for r in rs]):+.1f}% | {med([r['optimism'] for r in rs]):+.0f}% |")
    ok = med(dv) <= -2.0 and med(dt) > -1.0 and med(o_hi) > med(o_lo)
    md += ["", f"От {lo} к {hi} попыткам: ошибка на проверке изменилась на {med(dv):+.1f}%, ошибка на будущем — на "
               f"{med(dt):+.1f}% ({words(paired([0] * len(dt), dt))}); самообман: {med(o_lo):+.0f}% → {med(o_hi):+.0f}%.",
           f"**Вывод:** гипотеза {'подтверждается' if ok else 'НЕ подтверждается'} по медианам на {len(dt)} наблюдениях "
           "(критерий: проверка улучшается на ≥ 2%, будущее — меньше чем на 1%, самообман растёт).", ""]


def verdict_es(rows, md):
    modes = sorted({r["es_mode"] for r in rows})
    if not {"shared", "separate"} <= set(modes):
        return
    b = max(r["budget"] for r in rows)
    by = group([r for r in rows if r["budget"] == b], ("dataset", "model", "fold"))
    d_opt, d_test = [], []
    for rs in by.values():
        a = {r["es_mode"]: r for r in rs}
        if "shared" in a and "separate" in a:
            d_opt.append(a["shared"]["optimism"] - a["separate"]["optimism"])
            d_test.append(pct(a["separate"]["tuned"], a["shared"]["tuned"]))
    md += ["## Гипотеза «Ранняя остановка»", "",
           "| остановка | изменение к стандартной | хуже стандартной | при обучении на всех данных | "
           "будущее против проверки | деревьев |", "|---|---|---|---|---|---|"]
    for e in modes:
        rs = [r for r in rows if r["es_mode"] == e and r["budget"] == b]
        md.append(f"| {ES.get(e, e)} | {med([r['vs_default'] for r in rs]):+.1f}% | "
                  f"{sum(r['vs_default'] > 0 for r in rs)} из {len(rs)} | "
                  f"{med([r['vs_default_refit'] for r in rs]):+.1f}% | {med([r['optimism'] for r in rs]):+.0f}% | "
                  f"{med([r['n_iter'] for r in rs]):.0f} |")
    ok = med(d_opt) >= 3.0 and med(d_test) <= 1.0
    p = paired(d_opt, [0] * len(d_opt))
    verdict = "НЕ подтверждается" if not ok else ("подтверждается" if p is not None and p < 0.05 else
                                                  "подтверждается по медианам, но различие статистически не доказано")
    md += ["", f"Отдельная выборка вместо общей: самообман меньше на {med(d_opt):.1f} процентных пункта "
               f"({words(p)}), ошибка на будущем изменилась на {med(d_test):+.1f}%. Наблюдений: {len(d_opt)}.",
           f"**Вывод:** гипотеза {verdict} "
           "(критерий: самообман меньше на ≥ 3 п.п., ошибка на будущем не хуже чем на 1%).", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="base")
    a = ap.parse_args()
    runs = load(a.experiment)
    rows = rows_of(runs)
    cfg = runs[0]["config"]
    samplers = sorted({r["sampler"] for r in rows})
    top = max(r["budget"] for r in rows)
    last = [r for r in rows if r["budget"] == top]
    md = [f"# Отчёт: настройка Optuna, эксперимент «{a.experiment}»", "",
          "Создаётся `scripts/report_hpo.py`. Не редактировать вручную.", "",
          "## Итог", "",
          f"Настроенные модели против стандартных на будущих данных: медианное изменение ошибки "
          f"{med([r['vs_default'] for r in last]):+.1f}%; настройка оказалась хуже стандартной в "
          f"{sum(r['vs_default'] > 0 for r in last)} случаях из {len(last)}. Реальная ошибка выше той, что видела Optuna, "
          f"на {med([r['optimism'] for r in last]):+.0f}% (медиана). Если те же настройки обучить на всех обучающих "
          f"строках (как стандартную): {med([r['vs_default_refit'] for r in last]):+.1f}%, хуже стандартной в "
          f"{sum(r['vs_default_refit'] > 0 for r in last)} из {len(last)}.", "",
          "## Условия", "",
          f"- датасеты: {', '.join(sorted({r['dataset'] for r in rows}))}; прогноз на 1 час вперёд "
          f"(только прошлые данные); отрезков времени на датасет: {len({r['fold'] for r in rows})}",
          f"- проверка внутри настройки: {SCHEME.get(cfg['val_scheme'], cfg['val_scheme'])}; попыток: {cfg['n_trials']}",
          f"- способ перебора: {', '.join(samplers)}"
          + (" — **ВНИМАНИЕ: Optuna не установлена, это случайный поиск; результаты не для диплома**"
             if "random_fallback" in samplers else ""), "",
          "## Опорные модели (ошибка на будущем, медиана)", "",
          "| датасет | как час назад | линейная (Ridge) |", "|---|---|---|"]
    for (d,), rs in sorted(group(rows, ("dataset",)).items()):
        md.append(f"| {d} | {med([r['naive'] for r in rs]):.2f} | {med([r['ridge'] for r in rs]):.2f} |")
    md += ["", "## Результат", ""]
    main_table(rows, md)
    verdict_trials(rows, md)
    verdict_es(rows, md)
    out = DOCS / f"HPO_REPORT_{a.experiment}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    keys = list(rows[0])
    with open(RESULTS / "hpo" / a.experiment / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Готово: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
