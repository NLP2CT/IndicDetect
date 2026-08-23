import json
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from transformers import AutoTokenizer
from tqdm import tqdm

# ================= CONFIG =================
OUTPUT_JSONL = Path("telugu_400_500_chunks.jsonl")

# Gutendex
GUTENDEX = "https://gutendex.com/books"

# Blogger
BLOG_BASE = "https://telugupatalalyrics.blogspot.com/"
BLOG_FEED = urljoin(BLOG_BASE, "/feeds/posts/summary")
BLOG_BATCH = 100
BLOG_MAX_POSTS = 8000

YEAR_LIMIT = 2022

MIN_TOKENS = 400
MAX_TOKENS = 500
TARGET_SAMPLES = 1000
# ==========================================


# ---------- Telugu cleaner ----------
TELUGU_RE = re.compile(r"[^\u0C00-\u0C7F\s.!?।॥]+")


def clean_telugu(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = TELUGU_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- Sentence splitter ----------
SENT_RE = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str):
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


# ---------- JSONL writer ----------
def write_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==========================================================
# 400–500 TOKEN MEANINGFUL CHUNK BUILDER
# ==========================================================
def build_chunks_from_sentences(sentences, tokenizer, collected):
    buffer = []

    for sent in sentences:
        trial = " ".join(buffer + [sent])
        tok_len = len(tokenizer.encode(trial, add_special_tokens=False))

        if tok_len <= MAX_TOKENS:
            buffer.append(sent)
            continue

        # finalize previous buffer
        if buffer:
            final_text = " ".join(buffer)
            final_tokens = len(tokenizer.encode(final_text, add_special_tokens=False))

            if MIN_TOKENS <= final_tokens <= MAX_TOKENS and final_text.endswith((".", "।", "॥", "?", "!")):
                collected.append({
                    "text": final_text,
                    "token_count": final_tokens
                })

        buffer = [sent]

        if len(collected) >= TARGET_SAMPLES:
            return collected

    return collected


# ==========================================================
# 1️⃣ GUTENDEX SCRAPER
# ==========================================================
def scrape_gutendex(tokenizer, collected):
    print("📚 Scraping Gutendex...")

    url = GUTENDEX

    while url and len(collected) < TARGET_SAMPLES:
        data = requests.get(url, timeout=30).json()

        for book in data["results"]:
            if "te" not in book["languages"]:
                continue

            txt_url = book["formats"].get("text/plain; charset=utf-8") \
                or book["formats"].get("text/plain")

            if not txt_url:
                continue

            try:
                raw = requests.get(txt_url, timeout=60).text
                text = clean_telugu(raw)

                if len(text) < 1000:
                    continue

                sentences = split_sentences(text)
                collected = build_chunks_from_sentences(sentences, tokenizer, collected)

                if len(collected) >= TARGET_SAMPLES:
                    return collected

            except Exception:
                continue

        url = data["next"]

    return collected


# ==========================================================
# 2️⃣ BLOGGER SCRAPER (before 2022)
# ==========================================================
def scrape_blogger(tokenizer, collected):
    print("📰 Scraping Blogger...")

    start = 1

    while len(collected) < TARGET_SAMPLES:
        feed_url = f"{BLOG_FEED}?start-index={start}&max-results={BLOG_BATCH}"
        soup = BeautifulSoup(requests.get(feed_url, timeout=30).text, "xml")

        entries = soup.find_all("entry")
        if not entries:
            break

        for e in entries:
            date_str = e.published.text[:10]
            year = datetime.strptime(date_str, "%Y-%m-%d").year

            if year >= YEAR_LIMIT:
                continue

            summary = e.summary.text if e.summary else ""
            text = clean_telugu(summary)

            if len(text) < 50:
                continue

            sentences = split_sentences(text)
            collected = build_chunks_from_sentences(sentences, tokenizer, collected)

            if len(collected) >= TARGET_SAMPLES:
                return collected

        start += BLOG_BATCH

    return collected


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Loading XLM-RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    OUTPUT_JSONL.unlink(missing_ok=True)

    collected = []

    collected = scrape_gutendex(tokenizer, collected)
    collected = scrape_blogger(tokenizer, collected)

    write_jsonl(OUTPUT_JSONL, collected)

    print(f"\n✅ Finished. Saved {len(collected)} samples → {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
