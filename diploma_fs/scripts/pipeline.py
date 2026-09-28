"""Полный прогон одной командой: run.py -> theory_check.py -> report.py -> make_html.py (results/report.html).

  python scripts/pipeline.py                                   # синтетика + diabetes, все методы
  python scripts/pipeline.py --dataset all --select-sample 3000
  python scripts/pipeline.py --dataset steel seoul_bike --method base_fe_all shape_fit ace

Удобно запускать в фоне и смотреть прогресс в дашборде (scripts/dashboard.py) или в
results/run.log. Шаг, который упал, не прерывает остальные: отчёт строится по тому, что есть.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "run.log"
PY = sys.executable


def step(title, args, log):
    log.write(f"\n=== {datetime.now():%H:%M:%S} {title}: {' '.join(args)}\n")
    log.flush()
    t = time.time()
    rc = subprocess.call([PY, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"=== {title}: {'OK' if rc == 0 else f'ОШИБКА (код {rc})'} за {time.time() - t:.0f} с\n")
    log.flush()
    return rc


def main():
    extra = sys.argv[1:]
    if "--method" not in extra:
        extra = ["--method", "all"] + extra
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as log:
        log.write(f"СТАРТ {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        step("Прогон методов", ["scripts/run.py", *extra], log)
        if not (ROOT / "results" / "theory_check.json").exists():
            step("Проверка H0", ["scripts/theory_check.py"], log)
        rc = step("Отчёт", ["scripts/report.py"], log)
        if rc == 0:
            rc = step("HTML-страница", ["scripts/make_html.py"], log)
        log.write(f"\nГОТОВО {datetime.now():%H:%M:%S}. "
                  + ("Откройте results/report.html или дашборд.\n" if rc == 0 else "Отчёт не построен, см. выше.\n"))


if __name__ == "__main__":
    main()
