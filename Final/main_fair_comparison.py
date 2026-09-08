import sys
import time
import os
import json
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data
from src.preprocessing import Preprocessor
from src.models_library import get_xgb_model, get_lgbm_model, get_catboost_model

DATA_PATH = Path(__file__).parent / "data" / "housing.csv"
TARGET = "median_house_value"
RESULTS_DIR = Path(__file__).parent / "results_fine"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FAIR COMPARISON: Grid Search vs Optuna (identical parameter spaces)")
print("=" * 70)

# ===================== DATA =====================
print("\n[1] Loading data...")
df = load_data(DATA_PATH)
num_cols = [c for c in df.columns if c not in [TARGET, "ocean_proximity"]]
X = df[num_cols].copy()
y = df[TARGET]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

preprocessor = Preprocessor()
X_train_pp = preprocessor.fit_transform(X_train)
X_val_pp = preprocessor.transform(X_val)
X_test_pp = preprocessor.transform(X_test)

# ===================== PARAMETER SPACES =====================
# Для КАЖДОЙ модели Grid и Optuna используют ОДИНАКОВЫЕ значения

xgb_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 8, 12],
    "learning_rate": [0.05, 0.1, 0.15],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "min_child_weight": [1, 5],
    "gamma": [0.0, 1.0],
    "reg_alpha": [0.0, 1.0],
    "reg_lambda": [0.0, 1.0],
}

lgbm_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 8, 12],
    "learning_rate": [0.05, 0.1, 0.15],
    "num_leaves": [15, 31, 55],
    "min_child_samples": [5, 20],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_alpha": [0.0, 1.0],
    "reg_lambda": [0.0, 1.0],
}

catboost_grid = {
    "iterations": [200],
    "depth": [4, 5, 6, 7, 8, 9],
    "learning_rate": [0.05, 0.1, 0.15],
    "l2_leaf_reg": [1, 3, 5, 7],
    "bagging_temperature": [0.0, 0.5, 1.0],
    "random_strength": [1, 2, 3, 4],
}

print(f"\n  XGBoost combos:  {np.prod([len(v) for v in xgb_grid.values()]):,}")
print(f"  LightGBM combos: {np.prod([len(v) for v in lgbm_grid.values()]):,}")
print(f"  CatBoost combos: {np.prod([len(v) for v in catboost_grid.values()]):,}")

# ===================== GRID SEARCH =====================
def run_grid_search(model_fn, grid, X_train, y_train, X_val, y_val, X_test, y_test, model_name, checkpoint_path, max_combos=None):
    keys = list(grid.keys())
    values = list(grid.values())
    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    if max_combos:
        all_combinations = all_combinations[:max_combos]
    total = len(all_combinations)

    print(f"\n  {model_name} Grid: {total:,} combos")

    start = time.time()
    best_r2 = -np.inf
    best_params = None
    all_r2 = []
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("last_idx", 0)
        all_r2 = ckpt.get("all_r2", [])
        best_r2 = ckpt.get("best_r2", -np.inf)
        best_params = ckpt.get("best_params", None)
        print(f"  Resuming from [{start_idx}/{total}]")
        start -= ckpt.get("elapsed", 0)

    for i, params in enumerate(all_combinations[start_idx:], start=start_idx):
        model = model_fn(**params)
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            r2 = r2_score(y_val, pred)
            all_r2.append(r2)

            if r2 > best_r2:
                best_r2 = r2
                best_params = params.copy()
                print(f"    *** NEW BEST [{i+1}] R2={r2:.4f} | params={params}")

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - start
                print(f"    [{i+1}/{total}] R2={r2:.4f} | Best={best_r2:.4f} | {elapsed/60:.1f}min")

                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        "last_idx": i + 1,
                        "all_r2": all_r2,
                        "best_r2": best_r2,
                        "best_params": best_params,
                        "elapsed": elapsed,
                    }, f)

        except Exception as e:
            all_r2.append(-1e10)

    grid_time = time.time() - start

    best_model = model_fn(**best_params)
    best_model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, best_model.predict(X_test))
    train_r2 = r2_score(y_train, best_model.predict(X_train))

    np.save(RESULTS_DIR / f"fair_r2_grid_{model_name.lower()}.npy", np.array(all_r2))

    print(f"  COMPLETE: {model_name} | Time: {grid_time/60:.1f}min | Best Val R2: {best_r2:.4f} | Test R2: {test_r2:.4f}")

    return {
        "model": model_name,
        "method": "Grid Search",
        "best_val_r2": best_r2,
        "test_r2": test_r2,
        "train_r2": train_r2,
        "best_params": best_params,
        "all_r2": all_r2,
        "time": grid_time,
        "n_combos": total,
        "model_obj": best_model,
    }


# ===================== OPTUNA (suggest_categorical — те же значения что и Grid) =====================
def run_optuna(model_fn, model_name, grid, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=100):
    print(f"\n  {model_name} Optuna (100 trials, same space as Grid)...")

    # Собираем те же значения что и в grid для categorical sampling
    grid_values = {k: list(v) for k, v in grid.items()}
    keys = list(grid_values.keys())

    def objective(trial):
        params = {}
        for k in keys:
            vals = grid_values[k]
            if len(vals) == 1:
                params[k] = vals[0]
            else:
                params[k] = trial.suggest_categorical(k, vals)

        model = model_fn(**params)
        try:
            model.fit(X_train, y_train)
            return r2_score(y_val, model.predict(X_val))
        except:
            return -1e10

    start = time.time()

    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=15)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=3)

    storage_path = RESULTS_DIR / f"optuna_fair_{model_name.lower()}.db"
    if storage_path.exists():
        storage_path.unlink()

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name.lower()}_fair",
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    optuna_time = time.time() - start

    best_params = study.best_params
    best_model = model_fn(**best_params)
    best_model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, best_model.predict(X_test))
    train_r2 = r2_score(y_train, best_model.predict(X_train))

    all_r2 = [t.value if t.value else -1e10 for t in study.trials]

    np.save(RESULTS_DIR / f"fair_r2_optuna_{model_name.lower()}.npy", np.array(all_r2))

    print(f"  COMPLETE: {model_name} | Time: {optuna_time:.1f}s | Best Val R2: {study.best_value:.4f} | Test R2: {test_r2:.4f}")

    return {
        "model": model_name,
        "method": "Optuna (TPE)",
        "best_val_r2": study.best_value,
        "test_r2": test_r2,
        "train_r2": train_r2,
        "best_params": best_params,
        "all_r2": all_r2,
        "time": optuna_time,
        "model_obj": best_model,
    }


# ===================== RUN ALL =====================
xgb_ckpt = RESULTS_DIR / "fair_ckpt_xgboost.json"
lgbm_ckpt = RESULTS_DIR / "fair_ckpt_lightgbm.json"
cb_ckpt = RESULTS_DIR / "fair_ckpt_catboost.json"

# --- XGBoost Grid ---
print("\n[2] XGBoost Grid Search...")
xgb_grid_res = run_grid_search(get_xgb_model, xgb_grid, X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, "XGBoost", xgb_ckpt)

# --- LightGBM Grid ---
print("\n[3] LightGBM Grid Search...")
lgbm_grid_res = run_grid_search(get_lgbm_model, lgbm_grid, X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, "LightGBM", lgbm_ckpt)

# --- CatBoost Grid ---
print("\n[4] CatBoost Grid Search...")
cb_grid_res = run_grid_search(get_catboost_model, catboost_grid, X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, "CatBoost", cb_ckpt)

# --- Optuna for all ---
print("\n[5] Optuna for all models...")
xgb_opt_res = run_optuna(get_xgb_model, "XGBoost", xgb_grid, X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, n_trials=100)
lgbm_opt_res = run_optuna(get_lgbm_model, "LightGBM", lgbm_grid, X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, n_trials=100)
cb_opt_res = run_optuna(get_catboost_model, "CatBoost", catboost_grid, X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, n_trials=100)

# ===================== COMPILE RESULTS =====================
print("\n[6] Compiling results...")

grid_results = [xgb_grid_res, lgbm_grid_res, cb_grid_res]
optuna_results = [xgb_opt_res, lgbm_opt_res, cb_opt_res]

comparison = []
for g, o in zip(grid_results, optuna_results):
    comparison.append({
        "model": g["model"],
        "grid_val_r2": g["best_val_r2"],
        "grid_test_r2": g["test_r2"],
        "grid_time_s": g["time"],
        "grid_combos": g["n_combos"],
        "optuna_val_r2": o["best_val_r2"],
        "optuna_test_r2": o["test_r2"],
        "optuna_time_s": o["time"],
        "optuna_trials": 100,
    })

pd.DataFrame(comparison).to_csv(RESULTS_DIR / "comparison_fair.csv", index=False)

print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<12} {'Method':<15} {'Val R2':>8} {'Test R2':>8} {'Combos/Trials':>14} {'Time':>10}")
print("-" * 75)

for c in comparison:
    print(f"{c['model']:<12} {'Grid':<15} {c['grid_val_r2']:>8.4f} {c['grid_test_r2']:>8.4f} {c['grid_combos']:>14,} {c['grid_time_s']/60:>8.1f}min")
    print(f"{'':<12} {'Optuna':<15} {c['optuna_val_r2']:>8.4f} {c['optuna_test_r2']:>8.4f} {c['optuna_trials']:>14} {c['optuna_time_s']/60:>8.1f}min")
    print()

# ===================== KDE =====================
print(f"\n{'='*70}")
print("KDE ANALYSIS")
print(f"{'='*70}")

for g, o in zip(grid_results, optuna_results):
    model_name = g["model"]
    grid_r2 = np.array(g["all_r2"])
    grid_r2 = grid_r2[grid_r2 > -1e9]
    optuna_r2 = np.array(o["all_r2"])
    optuna_r2 = optuna_r2[optuna_r2 > -1e9]

    print(f"\n  {model_name}:")
    print(f"    Grid:  n={len(grid_r2)}, mean={grid_r2.mean():.4f}, std={grid_r2.std():.4f}, max={grid_r2.max():.4f}")
    print(f"    Optuna: n={len(optuna_r2)}, mean={optuna_r2.mean():.4f}, std={optuna_r2.std():.4f}, max={optuna_r2.max():.4f}")

# ===================== PLOTS =====================
print(f"\n{'='*70}")
print("GENERATING PLOTS")
print(f"{'='*70}")

def compute_kde(data, x_grid, bandwidth):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data.reshape(-1, 1))
    return np.exp(kde.score_samples(x_grid.reshape(-1, 1)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, g, o in zip(axes, grid_results, optuna_results):
    model_name = g["model"]
    grid_r2 = np.array(g["all_r2"])
    grid_r2 = grid_r2[grid_r2 > -1e9]
    optuna_r2 = np.array(o["all_r2"])
    optuna_r2 = optuna_r2[optuna_r2 > -1e9]

    all_vals = np.concatenate([grid_r2, optuna_r2])
    data_range = all_vals.max() - all_vals.min()
    x_min = all_vals.min() - data_range * 0.1
    x_max = all_vals.max() + data_range * 0.1
    x_grid_arr = np.linspace(x_min, x_max, 500)

    grid_bw = max(data_range / 20, 0.002)
    optuna_bw = max(data_range / 10, 0.002)

    l_x = compute_kde(grid_r2, x_grid_arr, grid_bw)
    g_x = compute_kde(optuna_r2, x_grid_arr, optuna_bw) if len(optuna_r2) > 1 else np.zeros_like(x_grid_arr)

    ax.fill_between(x_grid_arr, 0, l_x, alpha=0.25, color="#1f77b4", label=f"Grid (n={len(grid_r2)})")
    ax.plot(x_grid_arr, l_x, linewidth=2, color="#1f77b4")

    if len(optuna_r2) > 1:
        ax.fill_between(x_grid_arr, 0, g_x, alpha=0.35, color="#ff7f0e", label=f"Optuna (n={len(optuna_r2)})")
        ax.plot(x_grid_arr, g_x, linewidth=2, color="#ff7f0e")
        trial_density = np.ones_like(optuna_r2) * 0.01
        ax.scatter(optuna_r2, trial_density, color="#ff7f0e", s=15, alpha=0.7, edgecolors="white", linewidths=0.5)

    ax.axvline(grid_r2.max(), color="#1f77b4", linestyle="--", linewidth=2, label=f"Grid max={grid_r2.max():.4f}")
    if len(optuna_r2) > 0:
        ax.axvline(optuna_r2.max(), color="#ff7f0e", linestyle="--", linewidth=2, label=f"Optuna max={optuna_r2.max():.4f}")

    ax.set_title(model_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("R2 Score", fontsize=12)
    ax.set_ylabel("Density f(x)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    grid_mean = grid_r2.mean()
    grid_std = grid_r2.std()
    opt_mean = optuna_r2.mean() if len(optuna_r2) > 0 else 0
    opt_std = optuna_r2.std() if len(optuna_r2) > 0 else 0

    stats_text = f"Grid: μ={grid_mean:.4f}, σ={grid_std:.4f}\nOptuna: μ={opt_mean:.4f}, σ={opt_std:.4f}"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.suptitle("Fair Comparison: Grid Search vs Optuna (identical parameter spaces)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "kde_fair_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig(RESULTS_DIR / "kde_fair_comparison.pdf", bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'kde_fair_comparison.png'}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
