"""Convert Markdown documents into ISE-coursework-formatted PDFs.

Pipeline: markdown -> styled HTML -> headless Chrome -> PDF.

Style: Times New Roman 10 pt, single column, single line spacing. Figures
embedded inline via <img> tags reference the figures/ directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import markdown as md_lib


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 1.5cm 1.8cm; }
html, body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.15; color: #000; }
body { margin: 0; padding: 0; }
h1 { font-size: 14pt; margin: 0.5em 0 0.3em; }
h2 { font-size: 12pt; margin: 0.7em 0 0.25em; border-bottom: 1px solid #888; padding-bottom: 1pt; }
h3 { font-size: 11pt; margin: 0.5em 0 0.2em; }
h4 { font-size: 10pt; margin: 0.4em 0 0.15em; }
p, li { font-size: 10pt; margin: 0.25em 0; text-align: justify; }
table { border-collapse: collapse; margin: 0.4em 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #444; padding: 2pt 5pt; text-align: left; }
th { background: #eee; }
code, pre { font-family: 'Menlo', 'Courier New', monospace; font-size: 8.5pt; }
pre { background: #f5f5f5; padding: 4pt 6pt; border: 1px solid #ddd; overflow-wrap: break-word; white-space: pre-wrap; }
img { max-width: 80%; display: block; margin: 0.3em auto; }
hr  { border: none; border-top: 1px solid #888; margin: 0.6em 0; }
blockquote { margin: 0.3em 0 0.3em 1em; padding-left: 0.6em; border-left: 2px solid #888; color: #333; font-style: italic; }
ul, ol { margin: 0.2em 0 0.2em 1.2em; }
"""

HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head><body>{body}</body></html>"""


def md_to_html(md_text: str, title: str) -> str:
    body = md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    return HTML_TEMPLATE.format(title=title, css=CSS, body=body)


def html_to_pdf(html_path: str, pdf_path: str) -> None:
    if not os.path.exists(CHROME):
        sys.exit(f"Chrome not found at {CHROME!r}. Install Google Chrome first.")
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--no-pdf-header-footer",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={pdf_path}",
            f"file://{html_path}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def convert(md_path: str, pdf_path: str, title: str | None = None) -> None:
    title = title or os.path.basename(md_path).replace(".md", "")
    with open(md_path) as f:
        md_text = f.read()
    html = md_to_html(md_text, title)
    src_dir = os.path.dirname(os.path.abspath(md_path))
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=src_dir) as tf:
        tf.write(html)
        html_path = tf.name
    try:
        html_to_pdf(html_path, pdf_path)
    finally:
        os.unlink(html_path)
    print(f"  {md_path} -> {pdf_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = p.parse_args(argv)
    pairs = [
        ("report.md",       "report.pdf",       "GA-FT Report"),
        ("requirements.md", "requirements.pdf", "Requirements"),
        ("manual.md",       "manual.pdf",       "User Manual"),
        ("replication.md",  "replication.pdf",  "Replication Guide"),
    ]
    for md, pdf, title in pairs:
        md_path  = os.path.join(args.root, md)
        pdf_path = os.path.join(args.root, pdf)
        if not os.path.exists(md_path):
            print(f"  skipped {md} (not found)"); continue
        convert(md_path, pdf_path, title)


if __name__ == "__main__":
    main()
