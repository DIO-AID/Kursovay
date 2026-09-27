# fsx-diploma — отбор признаков с учётом модификаций

Экспериментальный стенд к диплому. План: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md),
результаты: [REPORT.md](REPORT.md), журнал решений: [DECISIONS_LOG.md](DECISIONS_LOG.md).

Требования: python 3.10+, scikit-learn, scipy, pandas, matplotlib, seaborn.

## Быстрый старт

```bash
cd diploma_fs
pip install scikit-learn scipy pandas matplotlib seaborn
python run.py --method base_raw base_fe_all sysoev_paper sysoev_fixed group_importance shape_fit boruta rfe
python theory_check.py
python report.py        # пересоздаёт REPORT.md и figures/*.png
```

Данные ищутся в `diploma_fs/data/` и в корневой `data/` репозитория
(`housing.csv`, `superconductivity/train.csv`) и подключаются автоматически.
Сырые JSON-результаты и PNG-графики не коммитятся через коннектор — они
воспроизводятся командами выше (фиксированные сиды).
