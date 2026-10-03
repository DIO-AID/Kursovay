"""Тесты стенда гипотез о настройке (fsx/hpo.py). Запуск: python -m pytest tests/test_hpo.py -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx import hpo                                   # noqa: E402


@pytest.mark.parametrize("scheme", ["random", "time"])
@pytest.mark.parametrize("es_mode", ["shared", "separate", "none"])
def test_inner_split_disjoint(scheme, es_mode):
    i_fit, i_es, i_val = hpo.inner_split(1000, scheme, es_mode, h=4)
    assert not set(i_fit) & set(i_val)                 # модель не учится на проверочных строках
    if es_mode == "shared":
        assert np.array_equal(i_es, i_val)
    elif es_mode == "separate":
        assert not set(i_es) & set(i_val) and not set(i_es) & set(i_fit)
    else:
        assert i_es is None
    if scheme == "time":
        assert i_fit.max() < i_val.min() - 3           # зазор h между обучением и проверкой


def test_validation_set_same_across_es_modes():
    """Режимы остановки сравниваются на одной и той же проверочной выборке."""
    v = [hpo.inner_split(500, "random", m, h=4)[2] for m in ("shared", "separate", "none")]
    assert np.array_equal(v[0], v[1]) and np.array_equal(v[0], v[2])


def test_budget_prefix():
    trials = [{"number": i, "val_mae": v, "test_mae": t, "n_iter": 1, "params": {}}
              for i, (v, t) in enumerate([(5, 9), (3, 8), (4, 2), (1, 7)])]
    b2, b4 = hpo.summarize_budget(trials, 2), hpo.summarize_budget(trials, 4)
    assert b2["best_trial"] == 1 and b2["test_mae"] == 8 and b2["oracle_test_mae"] == 8
    assert b4["best_trial"] == 3 and b4["test_mae"] == 7 and b4["oracle_test_mae"] == 2


@pytest.mark.parametrize("space", [hpo.xgb_space, hpo.lgbm_space, hpo.catboost_space, hpo.hgb_space])
def test_random_trial_in_bounds(space):
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = space(hpo._RandomTrial(rng))
        assert 0.01 <= p["learning_rate"] <= 0.1
        n = p.get("n_estimators", p.get("iterations", p.get("max_iter")))
        assert 100 <= n <= 1000


def test_end_to_end_small(tmp_path):
    cfg = dict(name="t", datasets=["synth_factory"], models=["hgb"], n_trials=4, budgets=[2, 4], folds=1,
               es_modes=["shared", "separate"], max_rows=1500)
    brief, missing = hpo.run_experiment(cfg, out_dir=tmp_path)
    assert len(brief) == 2 and not missing
    files = list((tmp_path / "t" / "synth_factory").glob("*.json"))
    assert len(files) == 2
    import json
    r = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(r["trials"]) == 4 and [b["budget"] for b in r["budgets"]] == [2, 4]
    assert all(b["refit_test_mae"] > 0 for b in r["budgets"])       # настройки, обученные на всём train
    assert r["n_fit"] + r["n_val"] + (r["n_es"] if r["es_mode"] == "separate" else 0) <= r["n_train"]
