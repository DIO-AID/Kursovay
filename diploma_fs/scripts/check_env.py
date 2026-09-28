"""Проверка окружения: python scripts/check_env.py"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORE = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
        "matplotlib": "matplotlib", "seaborn": "seaborn"}
EXTRA = {"catboost": ("catboost", "ветка research/catboost-shap"),
         "shap": ("shap", "ветка research/catboost-shap"),
         "optuna": ("optuna", "ветка research/optuna"),
         "optuna_dashboard": ("optuna-dashboard", "просмотр studies.db"),
         "streamlit": ("streamlit", "ветка research/app"),
         "ucimlrepo": ("ucimlrepo", "скачивание датасетов, research/real-data")}

ok = True
print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "— нужен 3.10+")
ok &= sys.version_info >= (3, 10)
print("\nЯдро (обязательно):")
for mod, pkg in CORE.items():
    try:
        v = importlib.import_module(mod).__version__
        print(f"  ✓ {pkg:16s} {v}")
    except ImportError:
        ok = False
        print(f"  ✗ {pkg:16s} НЕТ  ->  pip install {pkg}")
print("\nДополнительно (нужно для отдельных веток):")
for mod, (pkg, why) in EXTRA.items():
    try:
        v = getattr(importlib.import_module(mod), "__version__", "ok")
        print(f"  ✓ {pkg:16s} {v}")
    except ImportError:
        print(f"  – {pkg:16s} не установлен ({why})  ->  pip install {pkg}")
try:
    from fsx.data import DATASETS
    from fsx.selectors import REGISTRY
    print(f"\nКод проекта: {len(REGISTRY)} методов, датасеты: {', '.join(DATASETS)}")
except Exception as e:
    ok = False
    print(f"\n✗ Код проекта не импортируется: {e}")
print("\nИТОГ:", "всё готово к запуску" if ok else "есть проблемы, см. выше")
sys.exit(0 if ok else 1)
