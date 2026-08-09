"""Regenerates SmartIR-2_Report_Team47.pdf from REPORT.md.

Run after editing REPORT.md so the PDF submission artifact stays in sync:

    python3 generate_pdf.py

Requires `selenium` (already in dev use, not a runtime app dependency) and a
Chromium/Chrome binary. Set CHROME_BINARY if it isn't at the default path
this script tries first. No other dependency (no pandoc, no `markdown`
package) -- written from scratch specifically so this works without
internet access for `pip install`.

Styled to match assignment1's SmartIR_Lab_Report_Team47.pdf: cover page,
navy table headers, numbered figure captions, running header/footer with
page numbers (via Chrome DevTools Protocol Page.printToPDF -- the only way
to get header/footer templates without an external tool like pandoc).
"""
import base64
import html
import os
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

CHROME_BINARY = os.environ.get(
    "CHROME_BINARY",
    "/opt/homebrew/bin/chromium" if Path("/opt/homebrew/bin/chromium").exists()
    else "/usr/bin/chromium-browser" if Path("/usr/bin/chromium-browser").exists()
    else "/usr/bin/google-chrome",
)

REPORT_DIR = Path(__file__).resolve().parent
MD_PATH = REPORT_DIR / "REPORT.md"
HTML_PATH = REPORT_DIR / "_REPORT_render.html"
PDF_PATH = REPORT_DIR / "SmartIR-2_Report_Team47.pdf"

FIGURE_COUNTER = [0]


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def make_figure(alt, path):
    FIGURE_COUNTER[0] += 1
    abs_path = (REPORT_DIR / path).resolve()
    return (f'<figure><img src="file://{abs_path}" alt="{html.escape(alt)}">'
            f'<figcaption>Figure {FIGURE_COUNTER[0]} — {inline(alt)}</figcaption></figure>')


def parse_table(lines):
    header = [c.strip() for c in lines[0].strip().strip('|').split('|')]
    rows = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines[2:]]
    out = ['<table>', '<thead><tr>'] + [f'<th>{inline(h)}</th>' for h in header] + ['</tr></thead><tbody>']
    for row in rows:
        out.append('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in row) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def _leading_ws(line):
    return len(line) - len(line.lstrip(' '))


def collect_list_item(lines, i, n, first_line_text):
    """Consumes the current list-item's first line plus any indented
    continuation lines (direct wraps, or blank-line-separated indented
    paragraphs) that belong to it, returning (html_text, next_index).

    Without this, every multi-line list item (any wrapped line, or an
    indented explanatory paragraph under a bullet) silently breaks out of
    the <ol>/<ul> and starts a new list afterwards -- which both loses the
    visual nesting AND resets <ol> numbering back to 1.
    """
    marker_line = lines[i]
    marker_indent = _leading_ws(marker_line)
    paragraphs = [[first_line_text]]
    i += 1
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if stripped == '':
            # Blank line: continue only if the next non-blank line is a
            # continuation paragraph (indented deeper than the marker,
            # and not itself a new list marker / heading / etc).
            j = i + 1
            while j < n and lines[j].strip() == '':
                j += 1
            if j < n and _leading_ws(lines[j]) > marker_indent and not re.match(
                    r'^\d+\.\s|^[-*]\s|^#{1,4}\s|^>|^\|', lines[j].strip()):
                paragraphs.append([lines[j].strip()])
                i = j + 1
                continue
            break
        if _leading_ws(raw) > marker_indent and not re.match(
                r'^\d+\.\s|^[-*]\s|^#{1,4}\s|^>|^\|', stripped):
            paragraphs[-1].append(stripped)
            i += 1
            continue
        break
    html_paragraphs = [inline(' '.join(p)) for p in paragraphs]
    return '<br><br>'.join(html_paragraphs) if len(html_paragraphs) > 1 else html_paragraphs[0], i


def convert(md_text):
    lines = md_text.split('\n')
    out = []
    i, n = 0, len(lines)
    in_ul = in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append('</ul>'); in_ul = False
        if in_ol:
            out.append('</ol>'); in_ol = False

    while i < n:
        line = lines[i]
        s = line.strip()

        if s == '':
            close_lists(); i += 1; continue
        if s == '---':
            close_lists(); out.append('<hr>'); i += 1; continue

        m = re.match(r'^(#{1,4})\s+(.*)', s)
        if m:
            close_lists()
            level = len(m.group(1))
            out.append(f'<h{level}>{inline(m.group(2))}</h{level}>')
            i += 1; continue

        if s.startswith('>'):
            close_lists()
            quote_lines = []
            while i < n and lines[i].strip().startswith('>'):
                quote_lines.append(re.sub(r'^\s*>\s?', '', lines[i]))
                i += 1
            inner = []
            for ql in quote_lines:
                hm = re.match(r'^(#{1,6})\s+(.*)', ql.strip())
                if hm:
                    inner.append(f'<strong class="bq-heading">{inline(hm.group(2))}</strong><br>')
                elif ql.strip() == '':
                    inner.append('<br>')
                else:
                    inner.append(inline(ql))
            out.append('<blockquote>' + ' '.join(inner) + '</blockquote>')
            continue

        if s.startswith('|') and i + 1 < n and re.match(r'^\|?[\s:|-]+\|?$', lines[i + 1].strip()):
            close_lists()
            table_lines = []
            while i < n and lines[i].strip().startswith('|'):
                table_lines.append(lines[i]); i += 1
            out.append(parse_table(table_lines))
            continue

        m = re.match(r'^([-*])\s+(.*)', s)
        if m:
            if not in_ul:
                close_lists(); out.append('<ul>'); in_ul = True
            item_text, i = collect_list_item(lines, i, n, m.group(2))
            out.append(f'<li>{item_text}</li>')
            continue

        m = re.match(r'^\d+\.\s+(.*)', s)
        if m:
            if not in_ol:
                close_lists(); out.append('<ol>'); in_ol = True
            item_text, i = collect_list_item(lines, i, n, m.group(1))
            out.append(f'<li>{item_text}</li>')
            continue

        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', s)
        if m:
            close_lists()
            out.append(make_figure(m.group(1), m.group(2)))
            i += 1; continue

        close_lists()
        para = [s]
        i += 1
        while i < n and lines[i].strip() != '' and not re.match(r'^(#{1,4})\s', lines[i].strip()) \
                and lines[i].strip() != '---' and not lines[i].strip().startswith('|') \
                and not lines[i].strip().startswith('>') and not re.match(r'^[-*]\s', lines[i].strip()) \
                and not re.match(r'^\d+\.\s', lines[i].strip()) \
                and not re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', lines[i].strip()):
            para.append(lines[i].strip()); i += 1
        out.append(f'<p>{inline(" ".join(para))}</p>')

    close_lists()
    return '\n'.join(out)


CSS = """
@page { margin: 20mm 16mm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; color: #1a1a1a; line-height: 1.55; font-size: 13.5px; }
.cover { page-break-after: always; padding-top: 60px; }
.cover .kicker { text-align: center; color: #444; font-size: 13px; margin-bottom: 60px; }
.cover .kicker img { height: 46px; display: block; margin: 0 auto 10px; }
.cover h1.title { text-align: center; font-size: 27px; margin: 0 0 6px; color: #16294f; }
.cover .subtitle { text-align: center; color: #333; font-size: 14.5px; margin-bottom: 4px; }
.cover .team { text-align: center; font-weight: 700; font-size: 15px; margin: 22px 0 4px; }
.cover .org { text-align: center; color: #444; font-size: 13px; margin-bottom: 50px; }
.cover table { margin-top: 30px; }
h1 { font-size: 22px; border-bottom: 3px solid #16294f; padding-bottom: 8px; color: #16294f; margin-top: 0; }
h2 { font-size: 18px; margin-top: 30px; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #16294f; }
h3 { font-size: 15px; margin-top: 22px; color: #16294f; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; page-break-inside: avoid; }
th, td { border: 1px solid #a9b4c6; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #16294f; color: #fff; font-weight: 600; }
tr:nth-child(even) td { background: #f2f4f8; }
code { background: #eef0f4; padding: 1px 5px; border-radius: 3px; font-size: 11.5px; }
blockquote { border-left: 4px solid #16294f; margin: 12px 0; padding: 6px 16px; background: #f2f4f8; color: #222; }
blockquote .bq-heading { font-size: 14.5px; display: block; margin-bottom: 4px; color: #16294f; }
figure { margin: 14px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #ccc; border-radius: 3px; }
figcaption { font-size: 11.5px; color: #444; margin-top: 5px; font-style: italic; }
hr { border: none; border-top: 1px solid #ccc; margin: 22px 0; }
ul, ol { margin: 8px 0; padding-left: 24px; }
a { color: #0b5fff; }
strong { color: #111; }
"""

COVER = """
<div class="cover">
  <div class="kicker">Work Integrated Learning Programmes Division<br>Birla Institute of Technology &amp; Science, Pilani</div>
  <h1 class="title">SmartIR-2: End-to-End Information Retrieval System</h1>
  <div class="subtitle">Information Retrieval (AIMLCZG537 / DSECLZG537) — S2 2025-26</div>
  <div class="subtitle">Assignment 2 — Report</div>
  <div class="team">Team 47</div>
  <table>
    <thead><tr><th>Submission Component</th><th>File / Evidence</th></tr></thead>
    <tbody>
      <tr><td>Streamlit application code</td><td><code>app.py</code>, <code>modules/*.py</code></td></tr>
      <tr><td>Dataset used</td><td><code>data/news_corpus.csv</code> (BBC-style news collection, 1,100 documents) + simulated API feed + crawled pages</td></tr>
      <tr><td>README (install &amp; run steps)</td><td><code>README.md</code> — <code>pip install -r requirements.txt</code>; <code>streamlit run app.py</code></td></tr>
      <tr><td>Live deployment</td><td>https://assignment2-irgroup47.streamlit.app/</td></tr>
      <tr><td>Demo evidence</td><td>Screenshots embedded throughout this report (Sections 2–8)</td></tr>
    </tbody>
  </table>
</div>
"""

HEADER_TEMPLATE = """
<div style="font-size:8px; color:#666; width:100%; padding:0 16mm; display:flex; justify-content:space-between;">
  <span>SmartIR-2 — Information Retrieval Assignment 2 — Team 47</span>
</div>
"""
FOOTER_TEMPLATE = """
<div style="font-size:8px; color:#666; width:100%; padding:0 16mm; display:flex; justify-content:space-between;">
  <span>SmartIR-2 — Information Retrieval Assignment 2 — Team 47</span>
  <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
</div>
"""


def main():
    md_text = MD_PATH.read_text()
    body = convert(md_text)
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{COVER}{body}</body></html>"
    HTML_PATH.write_text(html_doc)
    print(f"Wrote {HTML_PATH} ({len(html_doc)} chars), {FIGURE_COUNTER[0]} figures")

    opts = Options()
    opts.binary_location = CHROME_BINARY
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)
    driver.get(f"file://{HTML_PATH.resolve()}")
    time.sleep(1.5)
    result = driver.execute_cdp_cmd("Page.printToPDF", {
        "printBackground": True,
        "displayHeaderFooter": True,
        "headerTemplate": HEADER_TEMPLATE,
        "footerTemplate": FOOTER_TEMPLATE,
        "marginTop": 0.6, "marginBottom": 0.6, "marginLeft": 0.5, "marginRight": 0.5,
        "preferCSSPageSize": False,
    })
    driver.quit()
    pdf_bytes = base64.b64decode(result["data"])
    PDF_PATH.write_bytes(pdf_bytes)
    print(f"PDF written: {PDF_PATH} ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    main()
