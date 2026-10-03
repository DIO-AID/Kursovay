"""Оформление PDF для отчётов: шрифты с кириллицей, стили, таблицы, картинки.

Общая часть scripts/make_report_sample.py (образец формата) и scripts/report_forecast.py
(настоящий отчёт), чтобы оба выглядели одинаково. Числовые данные сюда не попадают —
это только внешний вид.

Требуется reportlab и matplotlib (оба в requirements-extra.txt).
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from reportlab.lib import colors                    # noqa: E402
from reportlab.lib.enums import TA_LEFT             # noqa: E402
from reportlab.lib.pagesizes import A4              # noqa: E402
from reportlab.lib.styles import ParagraphStyle     # noqa: E402
from reportlab.lib.units import mm                  # noqa: E402
from reportlab.pdfbase import pdfmetrics            # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont        # noqa: E402
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

FONT_DIRS = [Path("C:/Windows/Fonts"),
             Path("/usr/share/fonts/truetype/dejavu"),
             Path("/Library/Fonts")]

INK, INK2, MUTED, RULE, TINT = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6", "#f3f2ee"
ACCENT = "#2a78d6"
# фиксированный порядок цветов серий, чтобы графики читались одинаково от отчёта к отчёту
SERIES = {"CatBoost": "#2a78d6", "XGBoost": "#eb6834", "GLM": "#1baf7a",
          "Ridge": "#7b6cd9", "Как час назад": "#8a8984", "Как вчера": "#b9b8b2"}

_registered = False


def _font(names):
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return str(d / n)
    sys.exit(f"Не найден шрифт с кириллицей ({names}). Добавьте путь в FONT_DIRS.")


def register_fonts():
    """Регистрация шрифтов один раз за процесс."""
    global _registered
    if _registered:
        return
    pdfmetrics.registerFont(TTFont("Body", _font(["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"])))
    pdfmetrics.registerFont(TTFont("Bold", _font(["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"])))
    pdfmetrics.registerFont(TTFont("Mono", _font(["consola.ttf", "DejaVuSansMono.ttf", "Courier New.ttf"])))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body",
                                  boldItalic="Bold")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    _registered = True


# Стили Paragraph создаются ниже и сразу читают зарегистрированные шрифты, поэтому
# регистрация обязана произойти до их создания — иначе reportlab не может определить
# семейство шрифта «Bold» и падает уже при сборке документа.
register_fonts()


S = {
    "h1": ParagraphStyle("h1", fontName="Bold", fontSize=17, leading=21, spaceAfter=4, textColor=INK),
    "h2": ParagraphStyle("h2", fontName="Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5, textColor=INK),
    "h3": ParagraphStyle("h3", fontName="Bold", fontSize=10, leading=13, spaceBefore=7, spaceAfter=3, textColor=INK),
    "p": ParagraphStyle("p", fontName="Body", fontSize=9.2, leading=13, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Body", fontSize=7.8, leading=10.5, textColor=INK2),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=7.8, leading=10, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Bold", fontSize=7.8, leading=10, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7, leading=9, textColor=INK2),
    "box": ParagraphStyle("box", fontName="Body", fontSize=9.5, leading=14, textColor=INK),
}


def P(text, st="p"):
    return Paragraph(str(text), S[st])


def table(rows, widths, head=True, bold_col0=False, align_right=()):
    """Таблица с заголовком и тонкими линиями. В ячейках \n -> перенос строки."""
    data = []
    for i, r in enumerate(rows):
        st = "cellb" if (head and i == 0) else "cell"
        cells = []
        for j, c in enumerate(r):
            use = "cellb" if (bold_col0 and j == 0 and i > 0) else st
            if isinstance(c, str):
                c = c.replace("\n", "<br/>")
            cells.append(c if not isinstance(c, str) else Paragraph(c, S[use]))
        data.append(cells)
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for j in align_right:
        style.append(("ALIGN", (j, 0), (j, -1), "RIGHT"))
    if head:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TINT)))
    t.setStyle(TableStyle(style))
    return t


def note(text, accent=ACCENT):
    """Врезка с полосой слева — для выводов и предупреждений."""
    t = Table([[P(text, "box")]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4fc")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(accent)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def warn(text):
    return note(text, accent="#c2452d")


def fig_to_img(fig, width_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def axes(ax, ylabel=None):
    """Единый вид осей: без верхней и правой рамки, сетка только по горизонтали."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(axis="y", color=RULE, linewidth=0.5)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)


def save_pdf(story, out_path, title, footer=""):
    """Собирает PDF: поля, колонтитул с номером страницы."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, 12 * mm, footer or title)
        canvas.drawRightString(192 * mm, 12 * mm, f"стр. {canvas.getPageNumber()}")
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.restoreState()

    SimpleDocTemplate(str(out_path), pagesize=A4,
                      leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=18 * mm,
                      title=title, author="diploma_fs").build(
        story, onFirstPage=_page, onLaterPages=_page)
    return out_path