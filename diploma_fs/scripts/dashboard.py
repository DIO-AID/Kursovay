"""Дашборд: запуск прогонов в фоне и просмотр результатов в браузере.

  pip install streamlit            (или requirements-extra.txt)
  streamlit run scripts/dashboard.py

Три вкладки:
  Запуск     — выбрать датасеты и методы, запустить scripts/pipeline.py в фоне, смотреть лог;
  Результаты — готовый отчёт results/report.html (таблицы + графики) и рейтинг;
  Данные     — скачать датасеты из реестра и проверить файлы.
Дашборд ничего не считает сам: всё делают те же скрипты, что и из консоли.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsx.data import DATASETS, find_data  # noqa: E402
from fsx.registry import DATASETS as REG  # noqa: E402
from fsx.selectors import REGISTRY        # noqa: E402

RES = ROOT / "results"
RUN_LOG, DL_LOG = RES / "run.log", RES / "download.log"
FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

st.set_page_config(page_title="Выбор модификаций признаков", layout="wide")


def start(args, log_path):
    """Запускает скрипт отдельным процессом: закрытие вкладки браузера его не останавливает."""
    RES.mkdir(exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     creationflags=FLAGS)


def tail(path, n=40):
    if not path.exists():
        return "(лога ещё нет)"
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[-n:])


def status(path):
    if not path.exists():
        return "не запускался"
    t = path.read_text(encoding="utf-8", errors="replace")
    return "завершён" if "ГОТОВО" in t else "идёт (или прерван)"


page = st.sidebar.radio("Раздел", ["Результаты", "Запуск", "Данные"])
st.sidebar.caption(f"Папка проекта: {ROOT}")

if page == "Запуск":
    st.header("Запуск прогона")
    have = [k for k, r in REG.items() if find_data(r["file"])]
    kind = st.radio("Что прогоняем", ["Встроенные (синтетика, diabetes)", "Реальные из реестра"],
                    horizontal=True)
    c1, c2 = st.columns(2)
    if kind.startswith("Встроенные"):
        chosen = c1.multiselect("Датасеты", list(DATASETS), default=list(DATASETS))
        ds_args = ["--datasets", *chosen]
    else:
        chosen = c1.multiselect("Датасеты (только скачанные)", have, default=have,
                                help="Если список пуст — скачайте данные во вкладке «Данные».")
        ds_args = ["--dataset", *chosen]
    methods = c2.multiselect("Методы", sorted(REGISTRY), default=sorted(REGISTRY))
    sample = c2.number_input("Подвыборка для отбора (--select-sample, 0 = нет)",
                             min_value=0, value=0 if kind.startswith("Встроенные") else 3000, step=500)
    if status(RUN_LOG) == "идёт (или прерван)":
        st.warning("Похоже, прогон ещё идёт. Второй одновременно замедлит оба.")
    if st.button("▶ Запустить в фоне", type="primary", disabled=not chosen or not methods):
        args = ["scripts/pipeline.py", "--method", *methods, *ds_args]
        if sample:
            args += ["--select-sample", str(int(sample))]
        start(args, RES / "launcher.log")   # свой лог пишет сам pipeline.py (run.log)
        st.success("Запущено. Можно закрыть страницу — прогон продолжится. «Обновить» показывает прогресс.")
    st.subheader(f"Лог: {status(RUN_LOG)}")
    st.button("↻ Обновить")
    st.code(tail(RUN_LOG), language="text")

elif page == "Результаты":
    st.header("Результаты")
    html = RES / "report.html"
    rk = RES / "ranking.csv"
    if rk.exists():
        st.subheader("Общий рейтинг (меньше — лучше)")
        st.dataframe(pd.read_csv(rk, encoding="utf-8-sig"), hide_index=True)
    if html.exists():
        st.caption(f"Отчёт от {pd.Timestamp(html.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M} (UTC). "
                   f"Файл: {html}")
        st.download_button("Скачать отчёт (HTML)", html.read_bytes(), "report.html", "text/html")
        components.html(html.read_text(encoding="utf-8"), height=1400, scrolling=True)
    else:
        st.info("Отчёта ещё нет. Запустите прогон во вкладке «Запуск».")
    sm = RES / "summary.csv"
    if sm.exists():
        with st.expander("Полная таблица summary.csv с фильтрами"):
            df = pd.read_csv(sm, encoding="utf-8-sig")
            dsel = st.multiselect("Датасет", sorted(df.dataset.unique()), default=None)
            msel = st.multiselect("Модель", sorted(df.model.unique()), default=None)
            if dsel:
                df = df[df.dataset.isin(dsel)]
            if msel:
                df = df[df.model.isin(msel)]
            st.dataframe(df, hide_index=True)

else:
    st.header("Данные")
    rows = [{"имя": k, "роль": r["role"], "UCI": r["uci_id"], "цель": r["target"],
             "файл": r["file"], "скачан": bool(find_data(r["file"])), "статус": r["status"]}
            for k, r in REG.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2 = st.columns(2)
    if c1.button("⬇ Скачать недостающие"):
        start(["scripts/download_data.py"], DL_LOG)
        st.success("Скачивание запущено, нажимайте «Обновить».")
    if c2.button("✔ Проверить файлы"):
        start(["scripts/download_data.py", "--check"], DL_LOG)
    st.button("↻ Обновить")
    st.code(tail(DL_LOG, 80), language="text")
