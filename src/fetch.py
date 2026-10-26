"""Download NBA play-by-play (stats.nba.com playbyplayv2 format) mirrored at
github.com/shufinskiy/nba_data. nba.com / basketball-reference are not
reachable from every environment; this mirror is.

Usage: python src/fetch.py 1996 2024        (season start years; 2024 = 2024-25)
"""
import lzma
import sys
import tarfile
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/{name}.tar.xz"


def fetch(name):
    out = RAW / f"{name}.csv"
    if out.exists():
        return out
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / f"{name}.tar.xz"
    urllib.request.urlretrieve(URL.format(name=name), tmp)
    with tarfile.open(fileobj=lzma.open(tmp)) as t:
        t.extractall(RAW, filter="data")
    tmp.unlink()
    return out


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    for y in range(a, b + 1):
        print(fetch(f"nbastats_{y}"), flush=True)
