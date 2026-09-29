"""Образец формата отчёта (docs/REPORT_FORMAT.md) -> docs/REPORT_SAMPLE.pdf

ВСЕ ЧИСЛА УСЛОВНЫЕ: это макет, показывающий структуру и оформление. Реальный отчёт строит
scripts/report_forecast.py по результатам прогона.

  python scripts/make_report_sample.py            # нужен reportlab и matplotlib
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402,F401
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from reportlab.lib import colors                    # noqa: E402
from reportlab.lib.enums import TA_LEFT             # noqa: E402
from reportlab.lib.pagesizes import A4              # noqa: E402
from reportlab.lib.styles import ParagraphStyle     # noqa: E402
from reportlab.lib.units import mm                  # noqa: E402
from reportlab.pdfbase import pdfmetrics            # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont        # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "REPORT_SAMPLE.pdf"

# ---------- шрифты с кириллицей ----------
FONT_DIRS = [Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts"), Path("/Library/Fonts")]


def _font(names):
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return str(d / n)
    sys.exit(f"Не найден шрифт с кириллицей ({names}). Укажите путь в FONT_DIRS.")


pdfmetrics.registerFont(TTFont("Body", _font(["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"])))
pdfmetrics.registerFont(TTFont("Bold", _font(["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"])))
pdfmetrics.registerFont(TTFont("Mono", _font(["DejaVuSansMono.ttf", "consola.ttf", "Courier New.ttf"])))

INK, INK2, MUTED, RULE, TINT = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6", "#f3f2ee"
SERIES = {"CatBoost": "#2a78d6", "XGBoost": "#eb6834", "GLM": "#1baf7a"}   # фиксированный порядок

S = {
    "h1": ParagraphStyle("h1", fontName="Bold", fontSize=17, leading=21, spaceAfter=4, textColor=INK),
    "h2": ParagraphStyle("h2", fontName="Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5, textColor=INK),
    "p": ParagraphStyle("p", fontName="Body", fontSize=9.2, leading=13, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Body", fontSize=7.8, leading=10.5, textColor=INK2),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=7.8, leading=10, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Bold", fontSize=7.8, leading=10, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7, leading=9, textColor=INK2),
    "box": ParagraphStyle("box", fontName="Body", fontSize=9.5, leading=14, textColor=INK),
}


def P(text, st="p"):
    return Paragraph(text, S[st])


def table(rows, widths, head=True, bold_col0=False, zebra=True):
    data = []
    for i, r in enumerate(rows):
        st = "cellb" if (head and i == 0) else "cell"
        data.append([c if not isinstance(c, str) else
                     Paragraph(c.replace("\n", "<br/>"), S["cellb"] if (bold_col0 and j == 0 and i > 0) else S[st])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TINT)))
    t.setStyle(TableStyle(style))
    return t


def note(text):
    t = Table([[P(text, "box")]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4fc")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(SERIES["CatBoost"])),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def fig_to_img(fig, width_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def _axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(axis="y", color=RULE, linewidth=0.5)
    ax.set_axisbelow(True)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


# ---------- условные данные для графиков ----------
def demo_week(seed=3):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-11-05", periods=3 * 96, freq="15min")
    h = t.hour + t.minute / 60
    work = t.dayofweek < 5
    base = np.where((h >= 8) & (h < 20), 1.0, 0.25) * np.where(work, 1.0, 0.2)
    y = 4 + 55 * base + 8 * base * np.sin(np.arange(len(t)) / 9) + rng.normal(0, 1.5, len(t))
    y = np.clip(y, 2.5, None)
    lag = np.r_[y[:4], y[:-4]]                                 # «как час назад»
    pred = {"CatBoost": y + rng.normal(0, 1.8, len(t)),
            "XGBoost": y + rng.normal(0, 2.0, len(t)),
            "GLM": 0.6 * y + 0.4 * lag + rng.normal(0, 2.5, len(t)) + 1.5,
            "naive": lag}
    return t, y, pred


def chart_week():
    t, y, pred = demo_week()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(t, y, color=INK, lw=1.4, label="Факт")
    ax.plot(t, pred["naive"], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="Как час назад")
    for name in ("CatBoost", "GLM"):
        ax.plot(t, pred[name], color=SERIES[name], lw=1.0, label=f"Прогноз {name}")
    ax.set_ylabel("кВт·ч за 15 мин", color=INK2)
    days = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: (lambda d: f"{days[d.weekday()]} {d:%d.%m}\n{d:%H:%M}")(matplotlib.dates.num2date(v))))
    _axes(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), labelcolor=INK)
    return fig_to_img(fig, 174)


def chart_ablation():
    groups = ["прошлые показания\nдатчиков", "календарь", "короткие лаги\n(1–2 ч назад)",
              "скользящие окна", "суточные и\nнедельные лаги"]
    vals = [26.7, 15.9, 8.5, 1.2, -0.8]
    sig = [True, True, True, False, False]
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    y = np.arange(len(groups))[::-1]
    ax.barh(y, vals, color=[SERIES["CatBoost"] if s else "#b7cdea" for s in sig], height=0.55)
    for yi, v, s in zip(y, vals, sig):
        ax.text(max(v, 0) + 0.6, yi, f"{v:+.1f}%" + (" — значимо" if s else " — не доказано"),
                va="center", ha="left", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, groups)
    ax.set_xlabel("Рост ошибки MAE, если убрать группу, % (больше — группа важнее)", color=INK2)
    ax.set_xlim(-3, 38)
    _axes(ax)
    ax.grid(axis="x", color=RULE, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    return fig_to_img(fig, 174)


# ---------- страницы ----------
def banner(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "ОБРАЗЕЦ ФОРМАТА ОТЧЁТА — все числа условные. "
                                        "Реальный отчёт строит scripts/report_forecast.py")
    canvas.drawRightString(192 * mm, 10 * mm, f"стр. {doc.page}")
    canvas.restoreState()


def build():
    st = []
    st += [P("Прогноз энергопотребления металлургического цеха", "h1"),
           P("Отчёт по эксперименту · датасет «Потребление стального завода» (UCI 851) · "
             "горизонт 1 час · режим «прогноз»", "small"), Spacer(1, 6)]

    # 0. Итог
    st += [P("0. Итог", "h2"), note(
        "<b>Лучшая модель — CatBoost со всеми временными признаками.</b> Средняя ошибка прогноза на час "
        "вперёд — <b>3,12 кВт·ч</b> (на 15-минутный интервал). Без прошлых значений ошибка была 8,40 кВт·ч — "
        "стала <b>меньше в 2,7 раза</b>; наивный прогноз «как час назад» ошибается на 4,21 кВт·ч. "
        "Отбор признаков оставил <b>18 из 45</b> признаков, ошибка выросла лишь на 0,8%.")]

    # 1. Задача
    st += [P("1. Задача", "h2"), table([
        ["Вопрос", "Ответ"],
        ["Что прогнозируем", "потребление электроэнергии цехом за 15 минут, кВт·ч"],
        ["На сколько вперёд", "на 1 час (4 шага по 15 минут)"],
        ["Что известно в момент прогноза", "всё, что измерено до этого момента, и календарь (час, день недели). "
                                           "Текущие показания датчиков НЕ известны"],
        ["Как проверяем", "год делится на 10 отрезков; модель учится только на прошлом и проверяется на "
                          "следующем отрезке — 9 проверок на разных отрезках года"],
        ["Главная мера ошибки", "MAE — средняя ошибка в кВт·ч (меньше — лучше)"],
    ], [52, 122], bold_col0=True)]

    # 2. Вход
    st += [P("2. Вход: исходные данные", "h2"), table([
        ["Источник", "Период", "Шаг", "Строк", "Пропуски"],
        ["DAEWOO Steel (Корея), UCI 851", "01.01.2018 – 31.12.2018", "15 минут", "35 040 (взято 10 000 последних)", "нет"],
    ], [48, 38, 18, 44, 26]), Spacer(1, 6),
        P("<b>Первые 5 строк таблицы</b> (заголовки и значения переведены; часть столбцов о реактивной мощности "
          "свёрнута в «…»):", "p"), Spacer(1, 3),
        table([
            ["Дата и время", "Потребле-\nние, кВт·ч", "Реакт. мощн.\n(отст.), кВАр·ч", "Коэфф. мощн.\n(отст.), %",
             "Секунд от\nполуночи", "Тип дня", "День недели", "Тип нагрузки"],
            ["01.01.2018 00:15", "3,17", "2,95", "73,2", "900", "рабочий", "понедельник", "лёгкая"],
            ["01.01.2018 00:30", "4,00", "4,46", "66,8", "1 800", "рабочий", "понедельник", "лёгкая"],
            ["01.01.2018 00:45", "3,24", "3,28", "70,3", "2 700", "рабочий", "понедельник", "лёгкая"],
            ["01.01.2018 01:00", "3,31", "3,56", "68,1", "3 600", "рабочий", "понедельник", "лёгкая"],
            ["01.01.2018 01:15", "3,82", "4,50", "64,7", "4 500", "рабочий", "понедельник", "лёгкая"],
        ], [24, 20, 23, 21, 19, 17, 25, 25]), Spacer(1, 3),
        P("Значения в строках условные — в настоящем отчёте берутся из файла.", "small"), Spacer(1, 6),
        P("<b>Расшифровка столбцов</b>", "p"), Spacer(1, 3),
        table([
            ["Столбец в файле", "Что это", "Ед.", "Как используем"],
            ["Usage_kWh", "потребление электроэнергии за 15 мин", "кВт·ч", "<b>цель — это прогнозируем</b>"],
            ["Lagging_Current_Reactive.Power_kVarh", "реактивная мощность, отстающий ток", "кВАр·ч", "только прошлые значения"],
            ["Leading_Current_Reactive_Power_kVarh", "реактивная мощность, опережающий ток", "кВАр·ч", "только прошлые значения"],
            ["Lagging_Current_Power_Factor", "коэффициент мощности, отстающий ток", "%", "только прошлые значения"],
            ["Leading_Current_Power_Factor", "коэффициент мощности, опережающий ток", "%", "только прошлые значения"],
            ["CO2(tCO2)", "выбросы CO₂", "т", "<b>удалён</b>: считается из самого потребления — "
                                               "модель видела бы ответ (утечка)"],
            ["NSM", "секунд от полуночи", "с", "известно заранее"],
            ["WeekStatus", "тип дня: Weekday — рабочий, Weekend — выходной", "—", "известно заранее"],
            ["Day_of_week", "день недели: Monday — понедельник, …", "—", "известно заранее"],
            ["Load_Type", "тип нагрузки: Light_Load — лёгкая, Medium_Load — средняя, "
                          "Maximum_Load — максимальная", "—", "только прошлые значения"],
        ], [52, 64, 14, 44])]

    # 3. Признаки
    st += [P("3. Признаки: что видит модель", "h2"),
           P("Из исходных столбцов строятся признаки. Так выглядит <b>одна строка</b> (прогноз на "
             "05.11.2018 14:00, прогноз делается в 13:00):", "p"), Spacer(1, 3),
           table([
               ["Признак", "Значение", "Смысл", "Группа"],
               ["потребление 1 ч назад", "57,3 кВт·ч", "последнее известное значение", "короткие лаги"],
               ["потребление 1 ч 15 мин назад", "55,9 кВт·ч", "предыдущее значение", "короткие лаги"],
               ["потребление сутки назад в 14:00", "52,1 кВт·ч", "суточный ритм цеха", "суточные лаги"],
               ["потребление неделю назад в 14:00", "54,4 кВт·ч", "недельный ритм", "суточные лаги"],
               ["среднее за последний известный час", "56,2 кВт·ч", "сглаженный уровень", "скользящие окна"],
               ["разброс за последние сутки", "21,7 кВт·ч", "насколько неровно работает цех", "скользящие окна"],
               ["реакт. мощность 1 ч назад", "31,4 кВАр·ч", "прошлое показание датчика", "датчики"],
               ["час суток", "14", "календарь", "календарь"],
               ["день недели", "понедельник", "календарь", "календарь"],
           ], [58, 26, 56, 34]), Spacer(1, 4),
           P("Технические имена (например, <font name='Mono'>target.short__lag4</font> = «потребление 1 ч назад») "
             "приведены в приложении один раз; в тексте отчёта используются только русские названия.", "small"),
           Spacer(1, 6),
           table([
               ["Группа признаков", "Что внутри", "Число"],
               ["календарь", "час, день недели, выходной, месяц, секунд от полуночи", "10"],
               ["короткие лаги", "потребление 1 ч, 1 ч 15 мин, 1 ч 30 мин, 1 ч 45 мин назад", "4"],
               ["суточные и недельные лаги", "потребление сутки, двое суток и неделю назад", "3"],
               ["скользящие окна", "среднее и разброс за последний час и сутки", "4"],
               ["датчики", "реактивная мощность, коэффициенты мощности, тип нагрузки — час назад", "24"],
               ["<b>всего</b>", "", "<b>45</b>"],
           ], [44, 116, 14])]

    # 4. Модели
    st += [P("4. Модели", "h2"), table([
        ["Модель", "Что делает простыми словами", "Настройки"],
        ["Наивный «как час назад»", "ставит последнее известное значение — нижняя планка", "нет"],
        ["Наивный «как вчера»", "ставит значение сутки назад в это же время", "нет"],
        ["GLM (линейная регрессия)", "складывает признаки с весами; легко объяснить", "стандартизация признаков"],
        ["XGBoost", "ансамбль деревьев решений, каждое исправляет ошибки предыдущих", "500 деревьев, глубина 6"],
        ["CatBoost", "то же семейство, устойчивее к переобучению — основная модель", "500 деревьев, глубина 6"],
    ], [42, 88, 44], bold_col0=True)]

    # 5. Выход
    st += [P("5. Выход: как выглядит прогноз", "h2"),
           P("8 моментов из теста. Ошибка = прогноз − факт (кВт·ч).", "p"), Spacer(1, 3),
           table([
               ["Время", "Факт", "Как час\nназад", "GLM", "XGBoost", "CatBoost", "Ошибка\nCatBoost"],
               ["05.11 08:00", "31,4", "4,1", "27,9", "30,2", "30,8", "−0,6"],
               ["05.11 09:00", "58,7", "33,0", "55,1", "57,5", "58,1", "−0,6"],
               ["05.11 12:00", "61,2", "63,5", "59,8", "60,4", "61,9", "+0,7"],
               ["05.11 13:00", "44,3", "60,8", "53,6", "47,9", "46,0", "+1,7"],
               ["05.11 14:00", "59,8", "44,9", "56,2", "58,6", "59,1", "−0,7"],
               ["05.11 20:00", "14,6", "57,2", "22,4", "16,9", "15,8", "+1,2"],
               ["06.11 02:00", "3,9", "4,3", "6,8", "4,5", "4,2", "+0,3"],
               ["10.11 11:00", "12,1", "11,8", "18,5", "13,0", "12,6", "+0,5"],
           ], [24, 18, 22, 20, 22, 22, 22]), Spacer(1, 6),
           P("<b>Факт и прогнозы за три дня теста</b> (для читаемости показаны три модели; остальные — в приложении):", "p"), Spacer(1, 10),
           chart_week(),
           P("Чёрная линия — факт. Серый пунктир «как час назад» запаздывает на час на каждом скачке смены; "
             "модели с календарём предсказывают скачки заранее.", "small")]

    # 6. Результат
    st += [PageBreak(), P("6. Результат и сравнение", "h2"),
           P("<b>Главная таблица.</b> Средняя ошибка MAE, кВт·ч, по 9 проверкам (меньше — лучше).", "p"), Spacer(1, 3),
           table([
               ["Модель", "Без прошлых\nзначений", "Все признаки\n(45)", "После отбора\n(18)", "R², все\nпризнаки",
                "MAPE, %", "Время\nобучения"],
               ["Наивный «как час назад»", "—", "4,21", "—", "0,891", "14,8", "0 с"],
               ["Наивный «как вчера»", "—", "6,95", "—", "0,742", "23,5", "0 с"],
               ["GLM", "9,12", "4,03", "4,10", "0,902", "13,9", "0,1 с"],
               ["XGBoost", "8,61", "3,27", "3,30", "0,941", "11,2", "4 с"],
               ["<b>CatBoost</b>", "8,40", "<b>3,12</b>", "3,15", "<b>0,948</b>", "<b>10,6</b>", "9 с"],
           ], [44, 22, 22, 22, 20, 20, 24]), Spacer(1, 4),
           note("CatBoost <b>значимо лучше</b> GLM (p = 0.004) и наивного прогноза (p = 0.004). "
                "Разница CatBoost и XGBoost <b>не доказана</b> (p = 0.31) — модели практически равны. "
                "Временные признаки снижают ошибку CatBoost в 2,7 раза — <b>значимо</b> (p = 0.004)."),
           Spacer(1, 4),
           P("<b>Какие группы признаков важнее</b> (E2): насколько растёт ошибка CatBoost, если убрать одну группу.", "p"),
           chart_ablation()]

    st += [KeepTogether([P("<b>Отбор признаков</b> (E3, E4). Все методы отбирают из 45 признаков; модель — CatBoost.", "p"),
                         Spacer(1, 3), table([
                             ["Метод отбора", "Оставил", "Рост ошибки", "Устойчи-\nвость", "Что оставил (главное)", "Критерий E3"],
                             ["важность CatBoost", "16 (36%)", "+1,1%", "0,71", "лаги 1 ч, сутки; час; реакт. мощность", "выполнен"],
                             ["SHAP", "18 (40%)", "+0,8%", "0,74", "то же + разброс за сутки", "выполнен"],
                             ["перестановки", "14 (31%)", "+1,9%", "0,58", "лаги 1 ч, сутки; час", "выполнен"],
                             ["RFE", "21 (47%)", "+0,2%", "0,66", "лаги, окна, календарь", "выполнен"],
                             ["Boruta", "33 (73%)", "+0,1%", "0,85", "почти всё", "нет: > 50%"],
                             ["Сысоев (исправленный)", "7 (16%)", "+6,4%", "0,62", "по одному из каждой группы", "нет: +6,4%"],
                             ["Сысоев (как в статье)", "0 (0%)", "—", "—", "пустой отбор в 9 из 9 проверок", "нет"],
                         ], [34, 18, 18, 16, 60, 28])])]
    st += [Spacer(1, 4), P("Устойчивость — насколько совпадают наборы признаков в разных проверках (1 = всегда одинаковые). "
                          "Порог — 0,6.", "small")]

    st += [P("<b>Сравнение с опубликованными работами</b>", "p"), Spacer(1, 3), table([
        ["Работа", "Их постановка", "Их результат", "Наш в той же\nпостановке", "Можно ли сравнивать"],
        ["CatBoost, 2023 (ResearchGate)", "текущие показания + CO₂, случайное 80/20", "R² 0,998", "R² 0,95 (без CO₂)",
         "нет: CO₂ считается из ответа, у них утечка"],
        ["Sathishkumar и др., 2021", "то же, CUBIST", "RMSE 0,24", "—", "нет: та же утечка"],
        ["Sci. Reports, 2025 (Тетуан)", "10 мин вперёд, 3 лага, 70/30", "R² 0,989", "R² 0,98", "да, если их разбиение по времени"],
    ], [34, 44, 20, 26, 50])]

    # 7. Выводы
    st += [P("7. Выводы", "h2"), table([
        ["Эксперимент", "Вопрос", "Итог", "Почему"],
        ["E1", "Дают ли временные признаки рост?", "<b>выполнен</b>", "ошибка CatBoost меньше в 2,7 раза, значимо"],
        ["E2", "Какие группы важнее?", "описан", "датчики и календарь; суточные лаги почти ничего не добавляют"],
        ["E3", "Сколько можно отбросить?", "<b>выполнен</b>", "SHAP: 60% признаков убрано, ошибка +0,8%"],
        ["E4", "Метод Сысоева", "частично", "исправленный: 7 признаков, устойчив, но ошибка +6,4%; "
                                             "статейный даёт пустой отбор"],
    ], [24, 44, 22, 84], bold_col0=True)]

    st += [P("Словарик", "h2"), table([
        ["Термин", "Что значит"],
        ["MAE", "средняя ошибка прогноза в кВт·ч: на сколько в среднем прогноз отличается от факта"],
        ["MAPE", "та же ошибка в % от фактического значения"],
        ["R²", "какую долю колебаний потребления объясняет модель; 1 — идеально, 0 — не лучше среднего"],
        ["лаг", "значение величины в прошлом: «лаг 1 ч» — сколько было час назад"],
        ["утечка", "когда модели случайно доступна информация из будущего или сам ответ — оценка завышена"],
        ["значимо (p &lt; 0.05)", "разница между моделями не объясняется случайностью на 9 проверках "
                                 "(тест Уилкоксона с поправкой Холма)"],
        ["устойчивость отбора", "средняя доля общих признаков у наборов из разных проверок (мера Жаккара)"],
    ], [34, 140], bold_col0=True)]

    st += [Spacer(1, 8), P("<b>Приложение: технические имена</b> (один раз, для воспроизведения)", "p"),
           Spacer(1, 3), table([
               ["В отчёте", "В коде"],
               ["потребление 1 ч назад", "<font name='Mono'>target.short__lag4</font>"],
               ["потребление сутки назад", "<font name='Mono'>target.daily__lag96</font>"],
               ["среднее за последний час", "<font name='Mono'>target.rolling__mean4</font>"],
               ["реакт. мощность 1 ч назад", "<font name='Mono'>Lagging_Current_Reactive.Power_kVarh__lag4</font>"],
               ["Наивный «как час назад» / GLM", "<font name='Mono'>naive_last</font> / <font name='Mono'>glm</font>"],
               ["Сысоев (исправленный)", "<font name='Mono'>sysoev_fixed</font>"],
           ], [64, 110])]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Образец отчёта: прогноз энергопотребления", author="diploma_fs")
    doc.build(st, onFirstPage=banner, onLaterPages=banner)
    print(f"Готово: {OUT}")


if __name__ == "__main__":
    build()
