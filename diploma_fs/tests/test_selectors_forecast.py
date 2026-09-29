"""Методы отбора работают на временных признаках и возвращают подмножество столбцов."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.data import synth_factory     # noqa: E402
from fsx.lags import LagFE             # noqa: E402
from fsx.selectors import REGISTRY     # noqa: E402


@pytest.fixture(scope="module")
def data():
    ds = synth_factory(n=1800, n_sensors=10)
    fe = LagFE("1h").fit(ds.X, ds.y, ds.t)
    F = fe.transform(ds.X, ds.y, ds.t).iloc[fe.warmup_:].reset_index(drop=True)
    F = F.fillna(F.median())
    y = np.asarray(ds.y)[fe.warmup_:]
    return F, y, fe.groups(F.columns)


@pytest.mark.parametrize("name", ["cb_importance", "cb_permutation", "cb_shap", "rfe",
                                  "sysoev_paper", "sysoev_fixed", "group_importance"])
def test_selector(data, name):
    F, y, groups = data
    feats, info = REGISTRY[name](F, y, groups, 0)
    assert set(feats) <= set(F.columns)
    assert len(feats) == len(set(feats))
