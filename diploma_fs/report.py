"""Сводный отчёт: python report.py -> REPORT.md + figures/*.png"""
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from fsx.transforms import SEP, TRANSFORMS, base_of, tr_of

METHODS = ["base_raw", "base_fe_all", "sysoev_paper", "sysoev_fixed",
           "group_importance", "shape_fit", "boruta", "rfe"]
LABEL = {"base_raw": "Исходные (10)", "base_fe_all": "Все трансф. (50)",
         "sysoev_paper": "Exp1a Сысоев (статья)", "sysoev_fixed": "Exp1b Сысоев (испр.)",
         "group_importance": "Exp2 Групповая важн.", "shape_fit": "Exp3 Форма завис.",
         "boruta": "Exp4 Boruta", "rfe": "Exp5 RFE"}
MODELS = ["ridge", "mlp", "hgb"]
MLAB = {"ridge": "Ridge", "mlp": "MLP", "hgb": "Бустинг"}
REF = "base_fe_all"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


def load():
    data = {}
    for d in sorted(p.name for p in Path("results").iterdir() if p.is_dir()):
        data[d] = {m: json.load(open(f"results/{d}/{m}.json"))
                   for m in METHODS if Path(f"results/{d}/{m}.json").exists()}
    return data


def r2_vec(res, model):
    recs = sorted(res["records"], key=lambda r: (r["seed"], r["fold"]))
    return np.array([r["r2"][model] for r in recs])


def pval(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def jaccard_mean(sets):
    s = [set(x) for x in sets]
    return float(np.mean([len(a & b) / len(a | b) if a | b else 1.0
                          for a, b in itertools.combinations(s, 2)]))


def truth_scores(res):
    truth = res["truth"]
    tg = set(truth)
    exact, prec, rec = [], [], []
    for r in res["records"]:
        sel = r["features"]
        g = {base_of(c) for c in sel}
        prec.append(len(g & tg) / len(g) if g else 0.0)
        rec.append(len(g & tg) / len(tg))
        exact.append(np.mean([{tr_of(c) for c in sel if base_of(c) == b} == {t}
                              for b, t in truth.items()]))
    return np.mean(prec), np.mean(rec), np.mean(exact)


def main():
    data = load()
    md = ["# Отчёт по экспериментам: отбор признаков с учётом модификаций", "",
          "Протокол: 5-fold CV × 3 сида = 15 парных точек; FE и отбор внутри train-фолда. "
          "Δ — разность средних R² с базовой линией «все трансформации (50)»; "
          "p — парный тест Уилкоксона; ▲/▼ — значимо лучше/хуже (p < 0.05).", ""]

    # --- H0
    th = json.load(open("results/theory_check.json"))
    h = [r["hgb"] for r in th]; l = [r["ridge"] for r in th]
    md += ["## 0. Проверка гипотезы H0 (инвариантность дерева)", "",
           "40 комбинаций модификаций (x1..x4), по одному представителю на фактор (`theory_check.py`):", "",
           "| Модель | min R² | max R² | Разброс |", "|---|---|---|---|",
           f"| Бустинг | {min(h):.4f} | {max(h):.4f} | {max(h)-min(h):.4f} |",
           f"| Ridge | {min(l):.4f} | {max(l):.4f} | {max(l)-min(l):.4f} |", "",
           "![H0](figures/fig0_invariance.png)", ""]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(range(len(th)), sorted(l), label="Ridge", color="#d95f02", s=22)
    ax.scatter(range(len(th)), [sorted(zip(l, h))[i][1] for i in range(len(th))],
               label="Бустинг", color="#1b9e77", s=22)
    ax.set_title("Выбор трансформации важен для Ridge и почти безразличен бустингу",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("40 комбинаций трансформаций (x1..x4), по возрастанию R² Ridge")
    ax.set_ylabel("R² на тесте"); ax.legend(); ax.set_ylim(0.65, 1.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig0_invariance.png", dpi=150); plt.close(fig)

    # --- таблицы по датасетам
    summary = []
    for d, res in data.items():
        md += [f"## Датасет `{d}` ({res[REF]['n_rows']} строк, {res[REF]['n_base']} факторов)", "",
               "| Метод | Призн. | " + " | ".join(f"{MLAB[m]} R² (Δ)" for m in MODELS)
               + " | Стаб. (Жаккар) | Время отбора, с |",
               "|---|---|" + "---|" * len(MODELS) + "---|---|"]
        for meth in METHODS:
            if meth not in res: continue
            r = res[meth]; cells = []
            for m in MODELS:
                a, b = r2_vec(r, m), r2_vec(res[REF], m)
                p = pval(a, b); dlt = a.mean() - b.mean()
                mark = ("▲" if dlt > 0 else "▼") if p < 0.05 and meth != REF else ""
                cells.append(f"{a.mean():.4f} ({dlt:+.4f}){mark}")
                summary.append({"dataset": d, "method": meth, "model": m,
                                "r2": a.mean(), "delta": dlt, "p": p})
            nf = np.mean([x["n_features"] for x in r["records"]])
            st = jaccard_mean([x["features"] for x in r["records"]])
            tm = np.mean([x["select_time"] for x in r["records"]])
            md.append(f"| {LABEL[meth]} | {nf:.1f} | " + " | ".join(cells) + f" | {st:.2f} | {tm:.2f} |")
        md.append("")
        if res[REF]["truth"]:
            md += ["**Восстановление истины** (истинные факторы: "
                   + ", ".join(f"{t}({b})" for b, t in res[REF]["truth"].items()) + ")", "",
                   "| Метод | Точность групп | Полнота групп | Точная трансформация |", "|---|---|---|---|"]
            for meth in METHODS:
                if meth in res and meth != "base_raw":
                    pr, rc, ex = truth_scores(res[meth])
                    md.append(f"| {LABEL[meth]} | {pr:.2f} | {rc:.2f} | {ex:.2f} |")
            md.append("")
    S = pd.DataFrame(summary)

    # --- fig1: тепловая карта Δ
    ds = list(data)
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 5.2), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), ds):
        sub = S[(S.dataset == d) & (S.method != REF)]
        piv = sub.pivot(index="method", columns="model", values="delta").reindex(
            [m for m in METHODS if m != REF])[MODELS]
        pv = sub.pivot(index="method", columns="model", values="p").reindex(piv.index)[MODELS]
        ann = piv.map(lambda v: f"{v:+.3f}") + np.where(pv < 0.05, "*", "")
        lim = np.nanmax(np.abs(piv.values.clip(-0.1, 0.1)))
        sns.heatmap(piv.clip(-0.1, 0.1), annot=ann, fmt="", cmap="RdYlGn", center=0,
                    vmin=-lim, vmax=lim, cbar=False, ax=ax, linewidths=.5)
        ax.set_title(d, fontsize=12, fontweight="bold")
        ax.set_yticklabels([LABEL[m] for m in piv.index], rotation=0)
        ax.set_xticklabels([MLAB[m] for m in MODELS]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("ΔR² относительно «все 50 трансформаций» (* — p<0.05; шкала обрезана ±0.1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig1_delta_heatmap.png", dpi=150); plt.close(fig)

    # --- fig2: признаки vs R2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    pal = dict(zip(METHODS, sns.color_palette("tab10", len(METHODS))))
    for ax, m in zip(axes, ["ridge", "hgb"]):
        res = data["synth_indep"]
        for meth in METHODS:
            nf = np.mean([x["n_features"] for x in res[meth]["records"]])
            ax.scatter(nf, r2_vec(res[meth], m).mean(), s=90, color=pal[meth], label=LABEL[meth])
        ax.set_xscale("log"); ax.set_title(f"synth_indep, {MLAB[m]}", fontweight="bold")
        ax.set_xlabel("Число признаков (лог. шкала)"); ax.set_ylim(0.2, 1.0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("R² (среднее по 15 фолдам)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Групповая важность и форма зависимости: 4 признака и R² как у 50",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig2_nfeat_vs_r2.png", dpi=150); plt.close(fig)

    # --- fig3: восстановление истины
    rows = []
    for d in [x for x in ds if data[x][REF]["truth"]]:
        for meth in METHODS[1:]:
            pr, rc, ex = truth_scores(data[d][meth])
            rows.append({"dataset": d, "method": LABEL[meth], "Точная трансформация": ex,
                         "F1 групп": 2 * pr * rc / (pr + rc) if pr + rc else 0})
    T = pd.DataFrame(rows).melt(id_vars=["dataset", "method"], var_name="metric")
    g = sns.catplot(data=T, x="value", y="method", hue="metric", col="dataset",
                    kind="bar", height=4.6, aspect=1.1, palette=["#7570b3", "#1b9e77"])
    g.set_axis_labels("Доля (1 = идеально)", ""); g.set_titles("{col_name}")
    g.figure.suptitle("Только восстановление формы находит истинные трансформации",
                      fontsize=13, fontweight="bold", y=1.03)
    g.savefig(FIG / "fig3_truth_recovery.png", dpi=150, bbox_inches="tight"); plt.close("all")

    # --- fig4: кривые формы
    rec = sorted(data["synth_indep"]["shape_fit"]["records"],
                 key=lambda r: (r["seed"], r["fold"]))[0]
    truth = data["synth_indep"]["shape_fit"]["truth"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, b in zip(axes, ["x1", "x2", "x3", "x4"]):
        inf = rec["info"][b]; x = np.array(inf["grid"]); f = np.array(inf["pd"])
        ax.plot(x, f, "o", color="#444", ms=4, label="форма из модели")
        for t, col in [(inf["transform"], "#1b9e77"), ("raw", "#d95f02")]:
            gx = x if t == "raw" else TRANSFORMS[t](np.maximum(x, 1e-3))
            A = np.c_[np.ones_like(gx), gx]; c, *_ = np.linalg.lstsq(A, f, rcond=None)
            ax.plot(x, A @ c, color=col, lw=2,
                    label=f"{t}: R²={inf['fit_r2'][t]:.3f}")
        ax.set_title(f"{b}: найдено {inf['transform']}, истина {truth[b]}", fontweight="bold")
        ax.set_xlabel(b); ax.legend(fontsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("вклад в прогноз")
    fig.suptitle("Exp3: трансформация извлекается из формы зависимости, выученной бустингом",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig4_shape_curves.png", dpi=150); plt.close(fig)

    md += ["## Графики", "", "![delta](figures/fig1_delta_heatmap.png)", "",
           "![nfeat](figures/fig2_nfeat_vs_r2.png)", "", "![truth](figures/fig3_truth_recovery.png)",
           "", "![shape](figures/fig4_shape_curves.png)", ""]
    Path("REPORT.md").write_text("\n".join(md), encoding="utf-8")
    S.to_csv("results/summary.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
