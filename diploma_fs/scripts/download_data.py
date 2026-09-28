"""Скачивание датасетов из fsx/registry.py в diploma_fs/data/.

  python scripts/download_data.py            # все из реестра
  python scripts/download_data.py steel ccpp # выбранные
  python scripts/download_data.py --check    # только проверить уже скачанные файлы

Файл сохраняется ЦЕЛИКОМ (все колонки, как отдаёт UCI): удаление утечек и выбор цели —
дело реестра, а не скачивания. После сохранения проверяется, что в файле есть цель,
столбец времени и колонки из drop/cat_cols реестра, и что цель числовая.
Нужен пакет ucimlrepo (есть в requirements.txt).
"""
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                               # noqa: E402

from fsx.data import DataError, registry_dataset  # noqa: E402
from fsx.paths import ROOT                        # noqa: E402
from fsx.registry import DATASETS as REG          # noqa: E402

DATA = ROOT / "data"
RETRIES, PAUSE = 3, 5


def fetch(uci_id):
    from ucimlrepo import fetch_ucirepo
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fetch_ucirepo(id=uci_id)
        except Exception as e:                  # сеть/лимиты UCI: пауза и повтор
            last = e
            print(f"   попытка {attempt}/{RETRIES}: {e}")
            time.sleep(PAUSE * attempt)
    raise last


def check(name):
    try:
        ds = registry_dataset(name)
    except DataError as e:
        return f"ОШИБКА: {e}"
    m = ds.meta
    return (f"ok: {len(ds.y)} строк, {ds.X.shape[1]} факторов "
            f"(категориальных {len(m['cat_cols'])}), уникальных значений цели {m['n_unique_target']}, "
            f"цель {ds.y.min():.4g}..{ds.y.max():.4g}")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    only_check = "--check" in sys.argv
    names = args or list(REG)
    bad = [n for n in names if n not in REG]
    if bad:
        sys.exit(f"Нет в реестре: {bad}. Есть: {list(REG)}")
    DATA.mkdir(exist_ok=True)
    for n in names:
        r = REG[n]
        path = DATA / r["file"]
        print(f"{n} (UCI {r['uci_id']}) -> data/{r['file']}")
        if not only_check and not path.exists():
            if r["uci_id"] is None:
                print("   нет uci_id: положите файл вручную")
                continue
            try:
                d = fetch(r["uci_id"])
            except Exception as e:
                print(f"   НЕ СКАЧАЛСЯ: {e}")
                continue
            df = d.data.original if getattr(d.data, "original", None) is not None else \
                pd.concat([d.data.features, d.data.targets], axis=1)
            df.to_csv(path, index=False)
            print(f"   сохранено: {len(df)} строк, колонки: {list(df.columns)}")
            time.sleep(1)
        print("   " + (check(n) if path.exists() else "файла нет"))


if __name__ == "__main__":
    main()
