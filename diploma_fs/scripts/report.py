"""Шаг 7 конвейера: сводный отчёт.

Запуск (из папки diploma_fs), после scripts/run.py и scripts/theory_check.py:
  python scripts/report.py
  python scripts/report.py --datasets synth_indep synth_corr --focus synth_corr

Читает results/raw/<датасет>/<метод>.json и создаёт:
  results/summary.csv          — одна строка на (датасет, метод, модель)
  results/ranking.csv          — средние ранги методов (общий рейтинг)
  results/figures/fig*.svg     — графики (текст в SVG остаётся текстом)
  docs/REPORT.md               — отчёт со ссылками на графики

Статистика (правила заданы заранее, см. docs/DECISIONS.md):
  - каждый метод сравнивается с опорным (по умолчанию base_fe_all) парным тестом
    Уилкоксона по фолдам; поправка Холма внутри блока «датасет × модель»;
  - вердикт: ▲/▼ — значимо (p_holm < 0.05) и |ΔR²| ≥ 0.005;
             ≈   — значимо, но |ΔR²| < 0.005 (практически без разницы);
             пусто — различие не доказано;
  - общий рейтинг: ранги методов по R² в каждом блоке «датасет × модель»,
    тест Фридмана, критическая разница Неменьи (Demšar, 2006).
"""
import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, wilcoxon  # noqa: E402

from fsx.paths import DOCS, FIG, RAW, RESULTS       # noqa: E402
from fsx.transforms import TRANSFORMS, base_of, tr_of  # noqa: E402

# ------------------------------------------------------------------ настройки
REF = "base_fe_all"
ALPHA = 0.05
MIN_EFFECT = 0.005                     # порог практической значимости по R²
MODELS = ["ridge", "mlp", "hgb"]
MLAB = {"ridge": "Ridge", "mlp": "MLP", "hgb": "Бустинг"}
METRICS = ["r2", "mae", "rmse", "fit_time"]
ORDER = ["base_raw", "base_fe_all", "sysoev_paper", "sysoev_fixed",
         "group_importance", "shape_fit", "boruta", "rfe"]     # новые методы — в конец
LABEL = {"base_raw": "Исходные", "base_fe_all": "Все модификации",
         "sysoev_paper": "Сысоев (статья)", "sysoev_fixed": "Сысоев (испр.)",
         "group_importance": "Групповая важность", "shape_fit": "Форма зависимости",
         "boruta": "Boruta", "rfe": "RFE"}
# q_0.05 для теста Неменьи (Demšar, 2006, табл. 5), k = 2..10
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

# ------------------------------------------------------------------ оформление
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # 1–3 категориальные слоты
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def label(m):
    return LABEL.get(m, m)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return f"../results/figures/{name}.svg"


# ------------------------------------------------------------------ загрузка
def load(only=None):
    data = {}
    if not RAW.exists():
        sys.exit(f"Нет {RAW}. Сначала: python scripts/run.py --method all")
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        if only and d.name not in only:
            continue
        runs = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json"))}
        if runs:
            data[d.name] = runs
    if not data:
        sys.exit("Результатов нет. Сначала: python scripts/run.py --method all")
    return data


def methods_of(runs):
    known = [m for m in ORDER if m in runs]
    return known + sorted(m for m in runs if m not in ORDER)


def keyed(run):
    """{(seed, fold): record} — для парного сравнения по одним и тем же фолдам."""
    return {(r["seed"], r["fold"]): r for r in run["records"]}


def vec(run, model, metric="r2", keys=None):
    k = keyed(run)
    keys = keys or sorted(k)
    return np.array([k[x]["metrics"][model][metric] for x in keys], dtype=float)


def jaccard(sets):
    s = [set(x) for x in sets]
    if len(s) < 2:
        return 1.0
    return float(np.mean([len(a & b) / len(a | b) if a | b else 1.0
                          for a, b in itertools.combinations(s, 2)]))


def wilcoxon_p(a, b):
    if len(a) < 2 or np.allclose(a - b, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def holm(pvals):
    """Поправка Холма–Бонферрони. pvals: {имя: p} -> {имя: p_holm}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run_max = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        run_max = max(run_max, min(1.0, (m - i) * p))
        out[name] = run_max
    return out


def verdict(delta, p_holm):
    if not np.isfinite(p_holm) or p_holm >= ALPHA:
        return ""
    if abs(delta) < MIN_EFFECT:
        return "≈"
    return "▲" if delta > 0 else "▼"


def truth_scores(run):
    """Точность и полнота по факторам + доля фолдов, где у фактора выбрана ровно истинная форма."""
    truth = run["truth"]
    tg = set(truth)
    prec, rec, exact = [], [], []
    for r in run["records"]:
        sel = [c for c in r["features"] if "__cat_" not in c]
        g = {base_of(c) for c in sel}
        prec.append(len(g & tg) / len(g) if g else 0.0)
        rec.append(len(g & tg) / len(tg))
        exact.append(np.mean([{tr_of(c) for c in sel if base_of(c) == b} == {t}
                              for b, t in truth.items()]))
    p, r = float(np.mean(prec)), float(np.mean(rec))
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "exact": float(np.mean(exact))}


# ------------------------------------------------------------------ сводная таблица
def summarize(data):
    rows = []
    for d, runs in data.items():
        ref = runs.get(REF)
        for model in MODELS:
            block, pv = [], {}
            for meth in methods_of(runs):
                run = runs[meth]
                recs = run["records"]
                row = {"dataset": d, "method": meth, "model": model, "n_folds": len(recs)}
                for met in METRICS:
                    v = vec(run, model, met)
                    row[met] = float(v.mean())
                    if met == "r2":
                        row["r2_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                row["n_features"] = float(np.mean([r["n_features"] for r in recs]))
                row["stability"] = jaccard([r["features"] for r in recs])
                row["select_time"] = float(np.mean([r["select_time"] for r in recs]))
                row["delta"], row["p"] = float("nan"), float("nan")
                if ref is not None and meth != REF:
                    common = sorted(set(keyed(run)) & set(keyed(ref)))
                    if len(common) != len(recs) or len(common) != len(ref["records"]):
                        print(f"! {d}/{meth}: фолды не совпадают с {REF}, сравнение по {len(common)} общим")
                    a, b = vec(run, model, "r2", common), vec(ref, model, "r2", common)
                    row["delta"], row["p"] = float(a.mean() - b.mean()), wilcoxon_p(a, b)
                    pv[meth] = row["p"]
                elif meth == REF:
                    row["delta"] = 0.0
                block.append(row)
            ph = holm(pv) if pv else {}
            for row in block:
                row["p_holm"] = ph.get(row["method"], float("nan"))
                row["verdict"] = verdict(row["delta"], row["p_holm"])
            rows.extend(block)
    return rows


COLS = ["dataset", "method", "model", "n_folds", "r2", "r2_std", "delta", "mae", "rmse",
        "fit_time", "n_features", "stability", "select_time", "p", "p_holm", "verdict"]


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # -sig: Excel видит кириллицу
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


# ------------------------------------------------------------------ общий рейтинг
def ranking(rows, data):
    """Ранги по блокам «датасет × модель» (1 = лучший R²). Только методы, есть во всех датасетах."""
    common = [m for m in methods_of(next(iter(data.values())))
              if all(m in runs for runs in data.values())]
    blocks = [(d, m) for d in data for m in MODELS]
    allm = {m for runs in data.values() for m in runs}
    k, n = len(common), len(blocks)
    out = {"excluded": sorted(allm - set(common)), "methods": common, "blocks": blocks,
           "avg_rank": {}, "k": k, "n": n, "friedman_p": float("nan"), "cd": float("nan")}
    if k < 2:
        return out
    r2 = {(r["dataset"], r["model"], r["method"]): r["r2"] for r in rows}
    M = np.array([[r2[(d, mo, meth)] for meth in common] for d, mo in blocks])
    R = np.vstack([rankdata(-row) for row in M])
    out["avg_rank"] = dict(zip(common, R.mean(axis=0)))
    if k >= 3 and n >= 2:
        out["friedman_p"] = float(friedmanchisquare(*M.T).pvalue)
    q = Q05.get(k)
    if q is None:
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(1 - ALPHA, k, np.inf) / math.sqrt(2))
    out["cd"] = q * math.sqrt(k * (k + 1) / (6 * n))
    return out


# ------------------------------------------------------------------ графики
def fig0_invariance(th):
    rows = th["rows"] if isinstance(th, dict) else th
    order = np.argsort([r["ridge"] for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(x, [rows[i]["ridge"] for i in order], s=26, color=SERIES[1],
               edgecolor="white", linewidth=1, label="Ridge", zorder=3)
    ax.scatter(x, [rows[i]["hgb"] for i in order], s=26, color=SERIES[0],
               edgecolor="white", linewidth=1, label="Бустинг", zorder=3)
    ti = [j for j, i in enumerate(order) if rows[i].get("is_truth")]
    if ti:
        j = ti[0]
        combo = rows[order[j]].get("combo")
        txt = ", ".join(combo.values()) if isinstance(combo, dict) else ", ".join(combo or [])
        ax.scatter([j, j], [rows[order[j]]["ridge"], rows[order[j]]["hgb"]], s=110,
                   facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4,
                   label=f"истинная комбинация ({txt})")
    ax.set_xlabel(f"{len(rows)} комбинаций модификаций x1..x4 (по возрастанию R² Ridge)")
    ax.set_ylabel("R² на тесте")
    ax.set_xticks([])
    ax.legend(loc="lower right")
    ax.set_title("Выбор модификации важен для Ridge и почти безразличен бустингу", loc="left")
    return save(fig, "fig0_invariance")


def fig1_delta_heatmap(rows, data):
    ds = list(data)
    union = {}
    for runs in data.values():
        union.update(dict.fromkeys(methods_of(runs)))
    meths = [m for m in methods_of(union) if m != REF]
    ncol = min(4, len(ds))                          # много датасетов — переносим панели по строкам
    nrow = math.ceil(len(ds) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 + 2.4 * ncol, (0.45 * len(meths) + 1.2) * nrow + 0.4),
                             sharey=True, squeeze=False)
    for ax in axes.flat[len(ds):]:
        ax.axis("off")
    lim = 0.1
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    idx = {(r["dataset"], r["method"], r["model"]): r for r in rows}
    for ax, d in zip(axes.flat, ds):
        Z = np.full((len(meths), len(MODELS)), np.nan)
        for i, m in enumerate(meths):
            for j, mo in enumerate(MODELS):
                r = idx.get((d, m, mo))
                if r is None or not np.isfinite(r["delta"]):
                    continue
                Z[i, j] = r["delta"]
                c = np.clip(r["delta"], -lim, lim)
                ax.text(j, i, f"{r['delta']:+.3f}{r['verdict']}", ha="center", va="center",
                        fontsize=8.5, color="white" if abs(c) > 0.6 * lim else INK)
        ax.imshow(np.clip(Z, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(MODELS)), [MLAB[m] for m in MODELS])
        ax.set_yticks(range(len(meths)), [label(m) for m in meths])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(MODELS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(meths)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(d)
    fig.suptitle(f"ΔR² относительно «{label(REF)}»: синий — лучше, красный — хуже "
                 f"(цвет обрезан на ±{lim}; ▲▼ — значимо и ≥{MIN_EFFECT})",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "fig1_delta_heatmap")


def fig2_nfeat_vs_r2(rows, focus):
    """Точечная диаграмма: строки — методы (по числу признаков), столбцы — модели.
    Вертикальная линия — R² опорного метода; значения левее окна показаны стрелкой."""
    pts = [r for r in rows if r["dataset"] == focus]
    meths = sorted({r["method"] for r in pts},
                   key=lambda m: next(r["n_features"] for r in pts if r["method"] == m))
    y = {m: i for i, m in enumerate(meths)}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.3 * len(MODELS) + 2.2, 0.36 * len(meths) + 1.5),
                             sharey=True)
    for ax, mo in zip(axes, MODELS):
        sub = {r["method"]: r for r in pts if r["model"] == mo}
        ref = sub.get(REF, max(sub.values(), key=lambda r: r["r2"]))["r2"]
        hi = max(r["r2"] for r in sub.values())
        lo = ref - 0.1
        ax.axvline(ref, color=RULE, linewidth=1.5, zorder=1)
        for m, r in sub.items():
            col = BLUE if r["verdict"] == "▲" else RED if r["verdict"] == "▼" else MUTED
            if r["r2"] < lo:
                ax.scatter(lo, y[m], marker="<", s=50, color=col, zorder=3)
                ax.text(lo + 0.004, y[m], f"{r['r2']:.2f}", va="center", fontsize=8, color=INK2)
            else:
                ax.scatter(r["r2"], y[m], s=48, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xlim(lo - 0.006, hi + 0.01)
        ax.set_title(MLAB[mo], loc="left")
        ax.set_xlabel("R² (среднее по фолдам)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    n = {r["method"]: r["n_features"] for r in pts}
    axes[0].set_yticks(range(len(meths)), [f"{label(m)} · {n[m]:.0f}" for m in meths])
    axes[0].invert_yaxis()
    fig.suptitle(f"{focus}: число признаков и R² (линия — «{label(REF)}»; синий — значимо лучше, "
                 f"красный — хуже, серый — без значимой разницы)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "fig2_nfeat_vs_r2")


def fig3_truth_recovery(truth_tab):
    ds = sorted({t["dataset"] for t in truth_tab})
    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 0.42 * 8 + 1.4),
                             sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        sub = [t for t in truth_tab if t["dataset"] == d]
        y = np.arange(len(sub))
        h = 0.36
        ax.barh(y - h / 2 - 0.02, [t["exact"] for t in sub], height=h, color=SERIES[0],
                label="верная форма")
        ax.barh(y + h / 2 + 0.02, [t["f1"] for t in sub], height=h, color=SERIES[2],
                label="F1 по факторам")
        for yi, t in zip(y, sub):
            ax.text(t["exact"] + 0.02, yi - h / 2 - 0.02, f"{t['exact']:.2f}", va="center",
                    fontsize=8, color=INK2)
        ax.set_yticks(y, [label(t["method"]) for t in sub])
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("доля (1 = идеально)")
        ax.tick_params(axis="y", length=0)
        ax.set_title(d, loc="left")
    axes[0][0].invert_yaxis()                 # оси общие: переворачиваем один раз
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Восстановление истины на синтетике: нашёл ли метод нужные факторы и их форму",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "fig3_truth_recovery")


def fig4_shape_curves(run):
    rec = sorted(run["records"], key=lambda r: (r["seed"], r["fold"]))[0]
    info = {b: v for b, v in rec["info"].items() if isinstance(v, dict) and "pd" in v}
    if not info:
        return None
    truth = run.get("truth") or {}
    bases = [b for b in truth if b in info] or \
        sorted(info, key=lambda b: -info[b].get("share", 0))[:4]
    fig, axes = plt.subplots(1, len(bases), figsize=(3.7 * len(bases), 3.5), squeeze=False)
    for ax, b in zip(axes[0], bases):
        inf = info[b]
        x, f = np.array(inf["grid"]), np.array(inf["pd"])
        shift = inf.get("shift")
        if shift is None:            # старые JSON без сдвига: приближённо, как в GroupFE
            shift = 0.0 if x.min() > 0 else 1.0 - x.min()
        pos = np.maximum(x + shift, 1e-3)
        ax.plot(x, f, "o", color=INK2, ms=4, label="форма из модели", zorder=3)
        for t in dict.fromkeys([inf["transform"], "raw"]):      # найденная форма и raw
            col = SERIES[1] if t == "raw" else SERIES[0]
            g = pos if t == "raw" else TRANSFORMS[t](pos)
            A = np.c_[np.ones_like(g), g]
            c, *_ = np.linalg.lstsq(A, f, rcond=None)
            ax.plot(x, A @ c, color=col, lw=2, label=f"{t}: R²={inf['fit_r2'][t]:.3f}")
        tt = f", истина {truth[b]}" if b in truth else ""
        ax.set_title(f"{b}: найдено {inf['transform']}{tt}", loc="left", fontsize=10)
        ax.set_xlabel(b)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("вклад в прогноз")
    fig.suptitle(f"Форма зависимости, выученная аддитивным бустингом ({run['dataset']}, первый фолд)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "fig4_shape_curves")


def fig5_critical_difference(rk):
    """Классическая CD-диаграмма (Demšar, 2006): ось рангов 1..k, лучшие подписаны слева."""
    avg, cd, k = rk["avg_rank"], rk["cd"], rk["k"]
    items = sorted(avg.items(), key=lambda kv: kv[1])
    half = math.ceil(k / 2)
    left, right = items[:half], items[half:]
    step = 0.32
    fig_h = 1.4 + step * max(len(left), len(right))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(0.2, k + 0.8)
    ax.set_ylim(-(step * (max(len(left), len(right)) + 0.6)), 0.95)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color=INK, lw=1)
    for i in range(1, k + 1):
        ax.plot([i, i], [0, 0.1], color=INK, lw=1)
        ax.text(i, 0.16, str(i), ha="center", va="bottom", fontsize=9, color=INK)
    for i in range(1, k):
        ax.plot([i + .5, i + .5], [0, 0.05], color=INK, lw=0.7)
    # подписи: лучшие слева, худшие справа
    for n, (m, r) in enumerate(left):
        y = -step * (n + 1)
        ax.plot([r, r, 0.9], [0, y, y], color=INK, lw=1)
        ax.text(0.85, y, f"{label(m)}  {r:.2f}", ha="right", va="center", fontsize=9, color=INK)
    for n, (m, r) in enumerate(reversed(right)):
        y = -step * (n + 1)
        ax.plot([r, r, k + 0.1], [0, y, y], color=INK, lw=1)
        ax.text(k + 0.15, y, f"{r:.2f}  {label(m)}", ha="left", va="center", fontsize=9, color=INK)
    # группы без значимых различий (|Δранг| ≤ CD) — толстые линии
    ranks = [r for _, r in items]
    groups, last_end = [], -1
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] <= cd)
        if j > i and j > last_end:
            groups.append((i, j))
            last_end = j
    for g, (i, j) in enumerate(groups):
        y = -0.12 - 0.09 * g
        ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [y, y], color=INK, lw=3.2,
                solid_capstyle="round")
    # отрезок CD сверху
    ax.plot([1, 1 + cd], [0.62, 0.62], color=INK, lw=1)
    for xx in (1, 1 + cd):
        ax.plot([xx, xx], [0.57, 0.67], color=INK, lw=1)
    ax.text(1 + cd / 2, 0.72, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    return save(fig, "fig5_critical_difference")


# ------------------------------------------------------------------ отчёт
def fmt_p(p):
    if not np.isfinite(p):
        return "—"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Сводный отчёт по results/raw")
    ap.add_argument("--datasets", nargs="+", help="только эти датасеты")
    ap.add_argument("--focus", help="датасет для графиков fig2 и fig4 (по умолчанию synth_indep)")
    a = ap.parse_args()

    data = load(a.datasets)
    rows = summarize(data)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "summary.csv", rows, COLS)
    rk = ranking(rows, data)
    write_csv(RESULTS / "ranking.csv",
              [{"method": m, "label": label(m), "avg_rank": r}
               for m, r in sorted(rk["avg_rank"].items(), key=lambda kv: kv[1])],
              ["method", "label", "avg_rank"])
    focus = a.focus or ("synth_indep" if "synth_indep" in data else next(iter(data)))

    idx = {(r["dataset"], r["method"], r["model"]): r for r in rows}
    passports = [run.get("passport", {}) for runs in data.values() for run in runs.values()]
    commits = sorted({p.get("git_commit") or "?" for p in passports})
    dates = sorted(p.get("created", "") for p in passports if p.get("created"))

    md = ["# Отчёт по экспериментам: выбор модификаций признаков", "",
          "> Файл создан `scripts/report.py` — не редактировать вручную. "
          "Графики: `results/figures/*.svg`, таблица: `results/summary.csv`.", "",
          f"Результаты: {len(data)} датасет(ов), коммит(ы) кода {', '.join(commits)}, "
          f"прогоны {dates[0][:10] if dates else '?'} … {dates[-1][:10] if dates else '?'}.", "",
          f"**Как читать.** Δ — разница средних R² с «{label(REF)}» на одних и тех же фолдах. "
          f"p — парный тест Уилкоксона, p_holm — с поправкой Холма внутри блока «датасет × модель». "
          f"▲/▼ — значимо лучше/хуже (p_holm < {ALPHA} и |Δ| ≥ {MIN_EFFECT}); "
          f"≈ — значимо, но разница меньше {MIN_EFFECT} (практически нет); пусто — различие не доказано.", ""]

    # 0. H0
    th_path = RESULTS / "theory_check.json"
    fig0 = None
    if th_path.exists():
        th = json.loads(th_path.read_text(encoding="utf-8"))
        trs = th["rows"] if isinstance(th, dict) else th
        h, l = [r["hgb"] for r in trs], [r["ridge"] for r in trs]
        fig0 = fig0_invariance(th)
        md += ["## 0. Проверка H0: деревьям форма безразлична", "",
               f"{len(trs)} комбинаций модификаций x1..x4 на synth_indep, по одной модификации на фактор "
               "(`scripts/theory_check.py`). Факт известный — здесь он проверен на наших данных.", "",
               "| Модель | min R² | max R² | Разброс |", "|---|---|---|---|",
               f"| Бустинг | {min(h):.4f} | {max(h):.4f} | {max(h) - min(h):.4f} |",
               f"| Ridge | {min(l):.4f} | {max(l):.4f} | {max(l) - min(l):.4f} |", "",
               f"![H0]({fig0})", ""]
    else:
        md += ["## 0. Проверка H0", "", "Нет results/theory_check.json — запустите "
               "`python scripts/theory_check.py`.", ""]

    # 1. Общий рейтинг
    md += ["## 1. Общий рейтинг методов", "",
           f"Ранг метода по R² в каждом из {rk['n']} блоков «датасет × модель» "
           f"({', '.join(data)} × {', '.join(MLAB[m] for m in MODELS)}); 1 — лучший, "
           "ниже средний ранг — лучше.", "",
           "| Место | Метод | Средний ранг |", "|---|---|---|"]
    for i, (m, r) in enumerate(sorted(rk["avg_rank"].items(), key=lambda kv: kv[1]), 1):
        md.append(f"| {i} | {label(m)} | {r:.2f} |")
    md += ["", f"Тест Фридмана: p = {fmt_p(rk['friedman_p'])}. "
           f"Критическая разница Неменьи (α = {ALPHA}, k = {rk['k']}, N = {rk['n']}): "
           f"CD = {rk['cd']:.2f}. Методы, чьи средние ранги отличаются меньше чем на CD, "
           "статистически не различимы (на диаграмме соединены толстой линией)."]
    if rk["excluded"]:
        md.append("Не вошли в рейтинг (есть не на всех датасетах): "
                  + ", ".join(label(m) for m in rk["excluded"]) + ". "
                  "Для полного рейтинга прогоните их везде или ограничьте `--datasets`.")
    if rk["n"] < 10:
        md.append(f"Блоков всего {rk['n']} — тест Неменьи консервативен, "
                  "рейтинг предварительный до прогона на реальных датасетах.")
    fig5 = fig5_critical_difference(rk) if rk["k"] >= 2 else None
    if fig5:
        md += ["", f"![CD]({fig5})"]
    md.append("")

    # 2. По датасетам
    truth_tab = []
    for d, runs in data.items():
        any_run = next(iter(runs.values()))
        split = any_run.get("passport", {}).get("split", "?")
        md += [f"## Датасет `{d}` ({any_run['n_rows']} строк, {any_run['n_base']} факторов, "
               f"разбиение {split})", "",
               "| Метод | Призн. | " + " | ".join(f"{MLAB[m]} R² (Δ)" for m in MODELS)
               + " | Стабильность | Отбор, с |",
               "|---|---|" + "---|" * len(MODELS) + "---|---|"]
        for meth in methods_of(runs):
            cells = []
            for mo in MODELS:
                r = idx[(d, meth, mo)]
                dl = "" if meth == REF or not np.isfinite(r["delta"]) else f" ({r['delta']:+.4f}){r['verdict']}"
                cells.append(f"{r['r2']:.4f}{dl}")
            r0 = idx[(d, meth, MODELS[0])]
            md.append(f"| {label(meth)} | {r0['n_features']:.1f} | " + " | ".join(cells)
                      + f" | {r0['stability']:.2f} | {r0['select_time']:.2f} |")
        md.append("")
        truth = any_run.get("truth")
        if truth:
            md += ["**Восстановление истины** (истина: "
                   + ", ".join(f"{t}({b})" for b, t in truth.items()) + ")", "",
                   "| Метод | Точность факторов | Полнота факторов | Верная форма |", "|---|---|---|---|"]
            for meth in methods_of(runs):
                if meth == "base_raw":
                    continue
                s = truth_scores(runs[meth])
                truth_tab.append({"dataset": d, "method": meth, **s})
                md.append(f"| {label(meth)} | {s['precision']:.2f} | {s['recall']:.2f} | {s['exact']:.2f} |")
            md.append("")

    # графики
    md += ["## Графики", ""]
    md += [f"![delta]({fig1_delta_heatmap(rows, data)})", ""]
    md += [f"![nfeat]({fig2_nfeat_vs_r2(rows, focus)})", ""]
    if truth_tab:
        md += [f"![truth]({fig3_truth_recovery(truth_tab)})", ""]
    if "shape_fit" in data.get(focus, {}):
        f4 = fig4_shape_curves(data[focus]["shape_fit"])
        if f4:
            md += [f"![shape]({f4})", ""]

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Датасеты: {', '.join(data)}; методов в рейтинге: {rk['k']}")
    for m, r in sorted(rk["avg_rank"].items(), key=lambda kv: kv[1]):
        print(f"  {label(m):22s} {r:.2f}")
    print(f"Фридман p = {fmt_p(rk['friedman_p'])}, CD = {rk['cd']:.2f}")
    if rk["excluded"]:
        print("! не вошли в рейтинг (есть не на всех датасетах):", ", ".join(rk["excluded"]))
    print(f"-> {DOCS / 'REPORT.md'}, {RESULTS / 'summary.csv'}, {FIG}/*.svg")


if __name__ == "__main__":
    main()
