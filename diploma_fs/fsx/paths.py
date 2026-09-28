"""Все пути проекта считаются от корня diploma_fs/, откуда бы ни запускался скрипт."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = (ROOT / "data", ROOT.parent / "data")   # diploma_fs/data и корневая data/ репозитория
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"            # JSON по фолдам (не в git)
FIG = RESULTS / "figures"        # графики SVG
OPTUNA = RESULTS / "optuna"      # studies.db (не в git)
DOCS = ROOT / "docs"
