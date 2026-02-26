"""
scraper.py — Collect raw documents from Pittsburgh/CMU sources.
Outputs plain text files to data/raw/.
"""

import os
import time
import json
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pdfplumber

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ANLP-HW2-Scraper/1.0)"}

# ── Seed URLs ────────────────────────────────────────────────────────────────
SEED_URLS = [
    # General Pittsburgh
    "https://en.wikipedia.org/wiki/Pittsburgh",
    "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "https://www.britannica.com/place/Pittsburgh",
    "https://www.visitpittsburgh.com",
    # CMU
    "https://www.cmu.edu/about/",
    # Sports
    "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
    # Food festivals
    "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
    "https://www.picklesburgh.com/",
    "https://www.pghtacofest.com/",
    "https://pittsburghrestaurantweek.com/",
    # Culture
    "https://carnegiemuseums.org",
    "https://www.heinzhistorycenter.org",
    "https://www.pittsburghsymphony.org",
    "https://pittsburghopera.org",
    # Events
    "https://events.cmu.edu",
    "https://downtownpittsburgh.com/events/",
]


def fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None


def parse_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove nav, footer, scripts, styles
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = (parsed.netloc + parsed.path).replace("/", "_").strip("_")
    return name[:200] + ".txt"


def scrape_url(url: str):
    html = fetch_html(url)
    if not html:
        return
    text = parse_text(html)
    if len(text) < 100:
        return
    out_path = RAW_DIR / safe_filename(url)
    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] Saved {out_path.name} ({len(text)} chars)")


def scrape_pdf(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        tmp = Path("/tmp/tmp_doc.pdf")
        tmp.write_bytes(r.content)
        with pdfplumber.open(tmp) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        out_path = RAW_DIR / safe_filename(url).replace(".txt", "_pdf.txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"[OK] PDF saved {out_path.name}")
    except Exception as e:
        print(f"[WARN] PDF failed {url}: {e}")


def scrape_all():
    for url in SEED_URLS:
        if url.endswith(".pdf"):
            scrape_pdf(url)
        else:
            scrape_url(url)
        time.sleep(1)  # be polite


if __name__ == "__main__":
    scrape_all()
    print(f"\nDone. {len(list(RAW_DIR.iterdir()))} files in {RAW_DIR}")
