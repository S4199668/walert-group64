"""
Scrape the RMIT assessment-flexibility sources and save each as a .txt file.

Usage:
    pip install requests beautifulsoup4
    python scrape_rmit_sources.py

Output: ./rmit_txt/<SOURCE-ID>.txt (e.g. EXT-01.txt, POL-01.txt)
"""

import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sources = {
    "EXT-01": {
        "category": "Extension",
        "title": "Extensions",
        "url": "https://www.rmit.edu.au/students/my-course/assessment-results/special-consideration-extensions/extensions",
        "type": "webpage",
    },
    "SC-01": {
        "category": "Special Consideration",
        "title": "Special Consideration",
        "url": "https://www.rmit.edu.au/students/my-course/assessment-results/special-consideration-extensions/special-consideration",
        "type": "webpage",
    },
    "EAA-01": {
        "category": "Equitable Assessment Arrangements",
        "title": "Equitable Assessment Arrangements",
        "url": "https://www.rmit.edu.au/students/my-course/assessment-results/special-consideration-extensions/equitable-assessment-arrangements",
        "type": "webpage",
    },
    "EAA-02": {
        "category": "Equitable Assessment Arrangements",
        "title": "Equitable Learning FAQs",
        "url": "https://www.rmit.edu.au/students/support-services/equitable-learning/faqs",
        "type": "webpage",
    },
    "POL-01": {
        "category": "Policy",
        "title": "Assessment and Assessment Flexibility Policy",
        "url": "https://policies.rmit.edu.au/document/view.php?id=7",
        "type": "policy",
    },
}

OUT_DIR = Path("rmit_txt")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; study-notes-scraper/1.0)"}

# Tried in order; first match wins. Falls back to <body> if none match.
CONTENT_SELECTORS = ["#jump-content", "main", "[role=main]", "#content", "body"]

# Page chrome to strip before extracting text.
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "svg", "iframe"]


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    container = None
    for sel in CONTENT_SELECTORS:
        container = soup.select_one(sel)
        if container:
            break

    # Render table rows as pipe-separated lines so they stay readable.
    for row in container.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        row.replace_with(soup.new_string(" | ".join(cells) + "\n"))

    text = container.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def scrape(source_id: str, meta: dict) -> None:
    resp = requests.get(meta["url"], headers=HEADERS, timeout=30)
    resp.raise_for_status()

    body = extract_text(resp.text)

    header = (
        f"SOURCE ID: {source_id}\n"
        f"CATEGORY: {meta['category']}\n"
        f"TITLE: {meta['title']}\n"
        f"TYPE: {meta['type']}\n"
        f"URL: {meta['url']}\n"
        f"SCRAPED: {date.today().isoformat()}\n"
        f"{'=' * 70}\n\n"
    )

    out_path = OUT_DIR / f"{source_id}.txt"
    out_path.write_text(header + body, encoding="utf-8")
    print(f"[ok] {source_id} -> {out_path} ({len(body):,} chars)")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for source_id, meta in sources.items():
        try:
            scrape(source_id, meta)
        except Exception as exc:
            print(f"[fail] {source_id}: {exc}")
        time.sleep(1)  # be polite to the server


if __name__ == "__main__":
    main()
