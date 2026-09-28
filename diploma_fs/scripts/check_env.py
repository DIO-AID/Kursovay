"""Проверка окружения: python scripts/check_env.py"""
import importlib
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):          # Windows-консоль в cp1251 падала на символах ✓ и –
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORE = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
        "matplotlib": "matplotlib"}
EXTRA = {"ucimlrepo": ("ucimlrepo", "scripts/download_data.py, есть в requirements.txt"),
         "catboost": ("catboost", "ветка research/catboost-shap"),
         "shap": ("shap", "ветка research/catboost-shap"),
         "optuna": ("optuna", "ветка research/optuna"),
         "optuna_dashboard": ("optuna-dashboard", "просмотр studies.db"),
         "streamlit": ("streamlit", "ветка research/app")}

ok = True
print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "— нужен 3.10+")
ok &= sys.version_info >= (3, 10)
print("\nЯдро (обязательно):")
for mod, pkg in CORE.items():
    try:
        v = getattr(importlib.import_module(mod), "__version__", "ok")
        print(f"  ✓ {pkg:16s} {v}")
    except ImportError:
        ok = False
        print(f"  ✗ {pkg:16s} НЕТ  ->  pip install {pkg}  (или: pip install -r requirements.txt)")
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
    from fsx.registry import DATASETS as REG
    print(f"\nКод проекта: {len(REGISTRY)} методов, встроенные: {', '.join(DATASETS)}; "
          f"в реестре: {len(REG)}")
except Exception as e:
    ok = False
    print(f"\n✗ Код проекта не импортируется: {e}")
print("\nИТОГ:", "всё готово к запуску" if ok else "есть проблемы, см. выше")
sys.exit(0 if ok else 1)
