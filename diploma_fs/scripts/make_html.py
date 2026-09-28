"""HTML-версия отчёта: docs/REPORT.md -> results/report.html (одна страница, графики внутри).

  python scripts/make_html.py          (pipeline.py вызывает сам после report.py)

Файл открывается двойным щелчком в любом браузере и пересылается как есть.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.paths import DOCS, RESULTS  # noqa: E402

HTML_CSS = """
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f7f7f5;color:#0b0b0b}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px;background:#fff}
h1{font-size:26px;margin:8px 0 4px}h2{font-size:20px;margin:32px 0 8px;border-bottom:1px solid #e3e2de;padding-bottom:4px}
p,li{line-height:1.5;font-size:15px}blockquote{margin:8px 0;padding:6px 12px;border-left:3px solid #d9d8d4;color:#52514e}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13.5px;display:block;overflow-x:auto}
th,td{border:1px solid #e3e2de;padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#f0efec}
tr:nth-child(even) td{background:#fafaf8}.up{color:#1c5cab;font-weight:600}.down{color:#c0392b;font-weight:600}
figure{margin:12px 0;overflow-x:auto}figure svg{max-width:100%;height:auto}code{background:#f0efec;padding:0 4px;border-radius:3px}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e2de;padding:8px 0;font-size:13px}
nav a{margin-right:12px;color:#1c5cab;text-decoration:none}
"""


def _inline(t):
    import html as _h
    import re
    t = _h.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = t.replace("▲", '<span class="up">▲</span>').replace("▼", '<span class="down">▼</span>')
    return t


def write_html(md_lines, path):
    """Мини-конвертер нашего Markdown (заголовки, абзацы, таблицы, картинки) в одну HTML-страницу.
    SVG вставляются внутрь страницы — файл можно открыть двойным щелчком или переслать."""
    import re
    out, heads, table, para = [], [], [], []

    def flush_table():
        if not table:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table
                if not re.fullmatch(r"\|?[\s|:-]+\|?", r.strip())]
        out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr>"
                   + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                   + "</table>")
        table.clear()

    def flush_para():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    for line in md_lines:
        for ln in line.split("\n"):
            s = ln.rstrip()
            if s.startswith("|"):
                flush_para()
                table.append(s)
                continue
            flush_table()
            m = re.fullmatch(r"!\[(.*?)\]\((.+?)\)", s)
            if m:
                flush_para()
                f = (DOCS / m.group(2)).resolve()
                svg = f.read_text(encoding="utf-8") if f.exists() else f"<p>нет файла {f.name}</p>"
                svg = svg[svg.find("<svg"):] if "<svg" in svg else svg
                out.append(f"<figure>{svg}</figure>")
            elif s.startswith("#"):
                flush_para()
                lvl = len(s) - len(s.lstrip("#"))
                txt = s[lvl:].strip()
                aid = f"s{len(heads)}"
                if lvl == 2:
                    heads.append((aid, txt))
                out.append(f'<h{lvl} id="{aid}">{_inline(txt)}</h{lvl}>')
            elif s.startswith(">"):
                flush_para()
                out.append(f"<blockquote>{_inline(s.lstrip('> '))}</blockquote>")
            elif not s:
                flush_para()
            else:
                para.append(s)
    flush_table()
    flush_para()
    nav = "<nav>" + "".join(f'<a href="#{a}">{_inline(t)}</a>' for a, t in heads) + "</nav>"
    path.write_text("<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    f"<title>Отчёт: выбор модификаций признаков</title><style>{HTML_CSS}</style></head>"
                    f"<body><main>{nav}{''.join(out)}</main></body></html>", encoding="utf-8")


def main():
    md = DOCS / "REPORT.md"
    if not md.exists():
        sys.exit("Нет docs/REPORT.md — сначала python scripts/report.py")
    out = RESULTS / "report.html"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_html(md.read_text(encoding="utf-8").split("\n"), out)
    print(f"-> {out} (открыть в браузере)")


if __name__ == "__main__":
    main()
