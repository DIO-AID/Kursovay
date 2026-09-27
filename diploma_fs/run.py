"""Запуск: python run.py --method base_raw base_fe_all [--datasets synth_indep ...]"""
import argparse
import time

from fsx.data import DATASETS
from fsx.evaluate import run_method
from fsx.selectors import REGISTRY

p = argparse.ArgumentParser()
p.add_argument("--method", nargs="+", required=True, choices=sorted(REGISTRY))
p.add_argument("--datasets", nargs="+", default=list(DATASETS))
a = p.parse_args()
for m in a.method:
    for d in a.datasets:
        t = time.time()
        r2, nf = run_method(d, DATASETS[d], m, REGISTRY[m])
        print(f"{m:22s} {d:12s} n_feat={nf:5.1f} " +
              " ".join(f"{k}={v:.4f}" for k, v in r2.items()) + f"  [{time.time()-t:.0f}s]", flush=True)
