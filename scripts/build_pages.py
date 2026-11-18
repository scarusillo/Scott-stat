"""Build the GitHub Pages site: site/index.html is written as a page fragment (title, styles, markup,
script), so wrap it in a full HTML document with a charset and viewport, and copy the data file.

Usage: python scripts/build_pages.py  -> _site/index.html, _site/data.json
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "site", ROOT / "_site"


def main():
    page = (SRC / "index.html").read_text(encoding="utf-8")
    cut = page.index("</style>") + len("</style>")
    head, body = page[:cut], page[cut:]
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
           "<meta name=\"description\" content=\"The Scott Stat: how much every NBA player since 1951-52 made his team better, "
           "with coaches, synergy and an all-time legacy ranking.\">\n"
           "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
           "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
           + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(doc, encoding="utf-8")
    shutil.copy(SRC / "data.json", OUT / "data.json")
    (OUT / ".nojekyll").write_text("")
    print("built", OUT, round((OUT / "data.json").stat().st_size / 1e6, 1), "MB data")


if __name__ == "__main__":
    main()
