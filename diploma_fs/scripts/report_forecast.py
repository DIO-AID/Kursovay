"""Отчёт по экспериментам E1–E4 (ADR-001) -> docs/FORECAST_REPORT.md и results/forecast/summary.csv

  python scripts/report_forecast.py                    # режим forecast
  python scripts/report_forecast.py --mode virtual

Критерии зафиксированы в docs/FORECAST.md (критерий C) ДО прогона на реальных данных.
Главная метрика — MAE (в единицах цели, для стали — кВт·ч). Парные сравнения — по фолдам
(9 блоков теста), тест Уилкоксона, поправка Холма внутри каждого эксперимента.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsx.paths import DOCS, RESULTS  # noqa: E402

ALPHA = 0.05
MAIN = "catboost"
LABEL = {"naive": "наивный", "fs_no_lags": "без прошлых значений", "fs_all": "все временные признаки",
         "abl_no_calendar": "без календаря", "abl_no_short": "без коротких лагов",
         "abl_no_daily": "без суточных/недельных лагов", "abl_no_rolling": "без скользящих окон",
         "abl_no_exog": "без прошлых показаний датчиков",
         "sel_cb_importance": "важность CatBoost", "sel_cb_permutation": "перестановки",
         "sel_cb_shap": "SHAP", "sel_rfe": "RFE", "sel_boruta": "Boruta",
         "sel_sysoev_paper": "Сысоев (статья)", "sel_sysoev_fixed": "Сысоев (исправл.)",
         "sel_group_importance": "групповая важность"}


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
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 1e-3 else f"{p:.3f}")


def pct(a, b):
    return 100.0 * (a - b) / b


def backend(runs):
    for run in runs.values():
        for r in run["records"]:
            if MAIN in r["metrics"]:
                return r["metrics"][MAIN].get("backend")
    return None


def e1(data, md, rows):
    md += ["## E1. Дают ли временные признаки рост?", "",
           "Критерий: MAE CatBoost с временными признаками ниже, чем без них, на всех основных "
           "датасетах, p_holm < 0.05. Дополнительно — сравнение с наивным прогнозом.", "",
           "| датасет | модель | наивный (час назад) | наивный (вчера) | без прошлого | все признаки | "
           "снижение ошибки | R² без → с | p_holm |", "|---|---|---|---|---|---|---|---|---|"]
    pv, cells = {}, []
    for d, runs in data.items():
        if not {"fs_all", "fs_no_lags", "naive"} <= set(runs):
            continue
        nl, nd = vec(runs["naive"], "naive_last"), vec(runs["naive"], "naive_day")
        for m in ("ridge", MAIN):
            a, b = vec(runs["fs_all"], m), vec(runs["fs_no_lags"], m)
            pv[(d, m)] = wil(a, b)
            r2a, r2b = vec(runs["fs_all"], m, "r2"), vec(runs["fs_no_lags"], m, "r2")
            cells.append((d, m, nl.mean(), nd.mean(), b.mean(), a.mean(), r2b.mean(), r2a.mean()))
    ph = holm(pv)
    ok_all = bool(cells)
    for d, m, nl, nd, b, a, r2b, r2a in cells:
        p = ph[(d, m)]
        md.append(f"| {d} | {m} | {nl:.3f} | {nd:.3f} | {b:.3f} | **{a:.3f}** | в {b / a:.2f} раза | "
                  f"{r2b:.3f} → {r2a:.3f} | {fp(p)} |")
        rows.append(dict(exp="E1", dataset=d, model=m, method="fs_all", mae=a, ref_mae=b, p_holm=p))
        if m == MAIN:
            ok_all &= (a < b) and p < ALPHA
    md += ["", f"**Вывод E1:** {'критерий выполнен' if ok_all else 'критерий НЕ выполнен'} "
               f"(по модели {MAIN}).", ""]


def e2(data, md, rows):
    md += ["## E2. Какие группы признаков важнее? (абляция)", "",
           f"Рост MAE {MAIN}, если убрать одно семейство признаков (в % от набора «все»). "
           "Чем больше рост — тем важнее семейство.", ""]
    order = ["abl_no_calendar", "abl_no_short", "abl_no_daily", "abl_no_rolling", "abl_no_exog"]
    have = {m for runs in data.values() for m in runs if m.startswith("abl_")}
    fams = [f for f in order if f in have]
    md += ["| датасет | " + " | ".join(LABEL.get(f, f) for f in fams) + " |",
           "|---|" + "---|" * len(fams)]
    for d, runs in data.items():
        if "fs_all" not in runs:
            continue
        base = vec(runs["fs_all"], MAIN)
        pv = {f: wil(vec(runs[f], MAIN), base) for f in fams if f in runs}
        ph = holm(pv) if pv else {}
        cells = []
        for f in fams:
            if f not in runs:
                cells.append("—")
                continue
            v = pct(vec(runs[f], MAIN).mean(), base.mean())
            star = "*" if ph[f] < ALPHA else ""
            cells.append(f"{v:+.1f}%{star}")
            rows.append(dict(exp="E2", dataset=d, model=MAIN, method=f, mae=vec(runs[f], MAIN).mean(),
                             ref_mae=base.mean(), p_holm=ph[f]))
        md.append(f"| {d} | " + " | ".join(cells) + " |")
    md += ["", "\\* — различие значимо (Уилкоксон, Холм, p < 0.05).", ""]


def e3_e4(data, md, rows):
    sels = sorted({m for runs in data.values() for m in runs if m.startswith("sel_")})
    if not sels:
        return
    md += ["## E3. Сколько признаков можно отбросить?", "",
           f"Критерий: метод оставляет ≤ 50% признаков при росте MAE {MAIN} ≤ 2% от полного набора.", "",
           "| метод | " + " | ".join(f"{d}: доля / ΔMAE" for d in data) + " | устойчивость | E3 |",
           "|---|" + "---|" * (len(data) + 2)]
    for s in sels:
        cells, passed, stab = [], True, []
        for d, runs in data.items():
            if s not in runs or "fs_all" not in runs:
                cells.append("—")
                passed = False
                continue
            recs = runs[s]["records"]
            share = np.mean([r["n_features"] for r in recs]) / runs[s]["n_all_features"]
            n_empty = sum(bool(r.get("empty_selection")) for r in recs)
            dm = pct(np.nanmean(vec(runs[s], MAIN)), vec(runs["fs_all"], MAIN).mean())
            stab.append(runs[s]["stability"])
            passed &= share <= 0.5 and dm <= 2.0
            passed &= n_empty == 0
            cells.append(f"{share:.0%} / {dm:+.1f}%" + (f" (пустой отбор {n_empty}/{len(recs)})" if n_empty else ""))
            rows.append(dict(exp="E3", dataset=d, model=MAIN, method=s, mae=np.nanmean(vec(runs[s], MAIN)),
                             ref_mae=vec(runs["fs_all"], MAIN).mean(), share=share,
                             stability=runs[s]["stability"]))
        md.append(f"| {LABEL.get(s, s)} | " + " | ".join(cells) +
                  f" | {np.nanmedian(stab):.2f} | {'✔' if passed else '✘'} |")
    md += ["", "Доля — среднее число оставленных признаков / все; устойчивость — средний Жаккар "
               "наборов между фолдами (медиана по датасетам). Пустой отбор оценивается как прогноз "
               "средним train (так метод честно попадает в конец рейтинга, а не выпадает из него).", ""]

    # E4: Фридман по блокам «датасет × фолд»
    methods = ["fs_all"] + [s for s in sels if all(s in runs for runs in data.values())]
    M = []
    for d, runs in data.items():
        if "fs_all" not in runs:
            continue
        cols = [vec(runs[m], MAIN) for m in methods]
        M += [row for row in np.array(cols).T if not np.isnan(row).any()]
    M = np.array(M)
    k, n = len(methods), len(M)
    md += ["## E4. Метод Сысоева среди других", "",
           f"Средние ранги по MAE {MAIN} (1 = лучший) по блокам «датасет × фолд» (n = {n}). "
           "Критерий: место исправленного метода Сысоева в рейтинге и устойчивость ≥ 0.6.", ""]
    if k >= 3 and n >= 2:
        R = np.vstack([rankdata(r) for r in M]).mean(axis=0)
        p = float(friedmanchisquare(*M.T).pvalue)
        from scipy.stats import studentized_range
        cd = float(studentized_range.ppf(1 - ALPHA, k, np.inf) / math.sqrt(2)) * math.sqrt(k * (k + 1) / (6 * n))
        md += ["| место | метод | средний ранг |", "|---|---|---|"]
        for i, j in enumerate(np.argsort(R)):
            md.append(f"| {i + 1} | {LABEL.get(methods[j], methods[j])} | {R[j]:.2f} |")
        md += ["", f"Тест Фридмана p = {fp(p)}; критическая разница Неменьи CD = {cd:.2f} "
                   "(методы, чьи ранги отличаются меньше чем на CD, статистически неразличимы).", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="forecast", choices=["forecast", "virtual"])
    a = ap.parse_args()
    data = load(a.mode)
    if not data:
        sys.exit(f"Нет результатов в results/forecast/{a.mode}/raw — сначала scripts/run_forecast.py")
    any_run = next(iter(next(iter(data.values())).values()))
    bk = {backend(runs) for runs in data.values()}
    md = [f"# Отчёт: прогноз энергопотребления (режим {a.mode})", "",
          "Создаётся `scripts/report_forecast.py`. Не редактировать вручную.", "",
          f"- датасеты: {', '.join(data)}",
          f"- горизонт: {any_run.get('horizon')} = {any_run['lagfe']['horizon_steps']} шаг(ов) ряда по "
          f"{any_run['lagfe']['step']}; "
          f"фолдов: {len(any_run['records'])}; тест на утечку: {any_run['leak_check_points']} точек — пройден",
          f"- модель «{MAIN}»: {', '.join(str(b) for b in bk)}"
          + (" — **ВНИМАНИЕ: CatBoost не установлен, результаты не для диплома**" if "hgb_fallback" in bk else ""),
          f"- настройка: {any_run['tuning']}", ""]
    rows = []
    e1(data, md, rows)
    e2(data, md, rows)
    e3_e4(data, md, rows)
    out_md = DOCS / ("FORECAST_REPORT.md" if a.mode == "forecast" else f"FORECAST_REPORT_{a.mode}.md")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    out_csv = RESULTS / "forecast" / f"summary_{a.mode}.csv"
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if "hgb_fallback" in bk:
        print("ВНИМАНИЕ: CatBoost не установлен — вместо него HistGradientBoosting.")
    print(f"Готово: {out_md.relative_to(DOCS.parent)}, {out_csv.relative_to(DOCS.parent)}")


if __name__ == "__main__":
    main()
