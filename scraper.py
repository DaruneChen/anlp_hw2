"""
scraper.py — Collect raw documents from Pittsburgh/CMU sources.
Outputs plain text files to data/raw/.
"""

import time
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pdfplumber

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ANLP-HW2-Scraper/1.0)"}

# ── Single-page seed URLs ─────────────────────────────────────────────────────
SEED_URLS = [
    # Pittsburgh general
    "https://en.wikipedia.org/wiki/Pittsburgh",
    "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "https://en.wikipedia.org/wiki/Geography_of_Pittsburgh",
    "https://en.wikipedia.org/wiki/Culture_of_Pittsburgh",
    "https://en.wikipedia.org/wiki/Economy_of_Pittsburgh",
    "https://www.britannica.com/place/Pittsburgh",
    "https://www.visitpittsburgh.com",
    "https://www.visitpittsburgh.com/things-to-do/",
    "https://www.visitpittsburgh.com/events-festivals/",
    "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
    "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
    "https://www.visitpittsburgh.com/plan-your-trip/about-pittsburgh/",
    # CMU
    "https://en.wikipedia.org/wiki/Carnegie_Mellon_University",
    "https://en.wikipedia.org/wiki/History_of_Carnegie_Mellon_University",
    "https://en.wikipedia.org/wiki/Mellon_Institute_of_Industrial_Research",
    "https://www.cmu.edu/about/",
    "https://www.cmu.edu/about/history.html",
    "https://www.cmu.edu/about/traditions.html",
    "https://www.cmu.edu/about/schools-colleges.html",
    "https://www.cmu.edu/engage/alumni/events/campus/index.html",
    "https://www.cmu.edu/about/history#:~:text=%22My%20heart%20is%20in%20the,general%20public%20where%20few%20existed.",
    "https://www.cmu.edu/50/#:~:text=A%20Celebration%2050%20Years%20in,own%20initiatives%2C%20creations%20and%20ideas.",
    "https://www.tonyawards.com/press/carnegie-mellon-to-become-first-exclusive-higher-education-partner-of-the-tony-awards/#:~:text=More%20Info-,Carnegie%20Mellon%20to%20Become%20First%2C%20Exclusive%20Higher%20Education%20Partner%20of,one%20of%20the%20world's%20best.",
    
    # Sports
    "https://en.wikipedia.org/wiki/Pittsburgh_Steelers",
    "https://en.wikipedia.org/wiki/Pittsburgh_Pirates",
    "https://en.wikipedia.org/wiki/Pittsburgh_Penguins",
    "https://www.steelers.com/team/history/",
    "https://www.mlb.com/pirates/history",
    "https://www.nhl.com/penguins/team/history",
    # Food festivals
    "https://www.picklesburgh.com/",
    "https://www.pghtacofest.com/",
    "https://pittsburghrestaurantweek.com/",
    "https://littleitalydays.com",
    "https://bananasplitfest.com",
    # Culture & museums
    "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh",
    "https://carnegiemuseums.org",
    "https://www.heinzhistorycenter.org",
    "https://www.thefrickpittsburgh.org",
    "https://www.pittsburghsymphony.org",
    "https://pittsburghopera.org",
    "https://trustarts.org",
    # Events
    "https://events.cmu.edu",
    "https://downtownpittsburgh.com/events/",
    "https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d",
]

# ── URLs to crawl with subpages ───────────────────────────────────────────────
CRAWL_URLS = [
    ("https://www.cmu.edu/about/", "www.cmu.edu", 15),
    ("https://www.pittsburghsymphony.org", "www.pittsburghsymphony.org", 10),
    ("https://pittsburghopera.org", "pittsburghopera.org", 10),
    ("https://carnegiemuseums.org", "carnegiemuseums.org", 10),
    ("https://www.heinzhistorycenter.org", "www.heinzhistorycenter.org", 10),
    ("https://www.picklesburgh.com/", "www.picklesburgh.com", 5),
    ("https://downtownpittsburgh.com/events/", "downtownpittsburgh.com", 10),
    ("https://trustarts.org", "trustarts.org", 10),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    print(f"[OK] {out_path.name} ({len(text)} chars)")


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


def scrape_with_subpages(base_url: str, domain: str, max_pages: int = 10):
    """Crawl a site and follow internal links up to max_pages."""
    visited = set()
    queue = [base_url]

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = fetch_html(url)
        if not html:
            continue

        text = parse_text(html)
        if len(text) > 100:
            out_path = RAW_DIR / safe_filename(url)
            out_path.write_text(text, encoding="utf-8")
            print(f"[OK] {url} ({len(text)} chars)")

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            parsed = urlparse(href)
            # Stay on same domain, skip anchors/query params/non-http
            if (parsed.netloc == domain
                    and parsed.scheme in ("http", "https")
                    and not parsed.fragment
                    and href not in visited):
                queue.append(href)

        time.sleep(0.5)


def scrape_all():
    print("=== Scraping single-page seed URLs ===")
    for url in SEED_URLS:
        if url.endswith(".pdf"):
            scrape_pdf(url)
        else:
            scrape_url(url)
        time.sleep(1)

    print("\n=== Crawling sites with subpages ===")
    for base_url, domain, max_pages in CRAWL_URLS:
        print(f"\n-- Crawling {domain} (max {max_pages} pages) --")
        scrape_with_subpages(base_url, domain, max_pages)


if __name__ == "__main__":
    scrape_all()
    print(f"\nDone. {len(list(RAW_DIR.iterdir()))} files in {RAW_DIR}")