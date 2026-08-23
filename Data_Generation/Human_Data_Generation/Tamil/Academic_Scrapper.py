"""
Tamil Academic Writing Corpus Scraper
======================================
Sources (verified working):
  1. Tamil Wikipedia  — encyclopaedic / academic prose on science, history,
                        linguistics, philosophy, geography, biology, etc.
  2. Tamil Virtual Academy (tamilvu.org) — classical scholarship & course texts
  3. Project Madurai  — classical Tamil literary texts
  4. Tamil University Thanjavur — journal pages
  5. TNAU             — agricultural research pages

Quality gates (deliberately lenient — better recall, post-filter in training):
  • ≥ 2 academic-vocabulary keyword hits
  • Average sentence length ≥ 20 chars
  • Minimum 3 valid sentences per article
  • 70 % Tamil character ratio
  • No heavy entertainment / interview marker presence
  • XLM-RoBERTa token window: 400–500 tokens
"""

import argparse
import json
import random
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone
from dateutil import parser as dateparser
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
NUM_SAMPLES       = 1000
MIN_TOK           = 400
MAX_TOK           = 500
RANDOM_SEED       = 42
DEFAULT_MIN_CHARS = 200
LOWER_DATE        = datetime(2000, 1, 1)
UPPER_DATE        = datetime(2022, 1, 1)
MAX_WORKERS       = 10
CRAWL_DELAY       = 0.1   # seconds between requests per thread


def _auto_outfile() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tamil_academic_writing_{NUM_SAMPLES}_{MIN_TOK}-{MAX_TOK}_{ts}.jsonl"


OUTPUT_SAMPLES_FILE = _auto_outfile()
TOKENIZER = None

# ─────────────────────────────────────────────
#  Wikipedia seed articles (academic topics)
# ─────────────────────────────────────────────
WIKI_SEEDS = [
    # Natural Sciences
    "https://ta.wikipedia.org/wiki/இயற்பியல்",
    "https://ta.wikipedia.org/wiki/வேதியியல்",
    "https://ta.wikipedia.org/wiki/உயிரியல்",
    "https://ta.wikipedia.org/wiki/கணிதம்",
    "https://ta.wikipedia.org/wiki/வானியல்",
    "https://ta.wikipedia.org/wiki/புவியியல்",
    "https://ta.wikipedia.org/wiki/சுற்றுச்சூழல்",
    "https://ta.wikipedia.org/wiki/மருத்துவம்",
    "https://ta.wikipedia.org/wiki/வேளாண்மை",
    # Humanities & Social Sciences
    "https://ta.wikipedia.org/wiki/வரலாறு",
    "https://ta.wikipedia.org/wiki/தத்துவம்",
    "https://ta.wikipedia.org/wiki/மொழியியல்",
    "https://ta.wikipedia.org/wiki/இலக்கணம்",
    "https://ta.wikipedia.org/wiki/சமூகவியல்",
    "https://ta.wikipedia.org/wiki/உளவியல்",
    "https://ta.wikipedia.org/wiki/பொருளாதாரம்",
    "https://ta.wikipedia.org/wiki/அரசியல்_அறிவியல்",
    "https://ta.wikipedia.org/wiki/தொல்லியல்",
    # Tamil Studies
    "https://ta.wikipedia.org/wiki/தமிழ்_இலக்கியம்",
    "https://ta.wikipedia.org/wiki/சங்க_இலக்கியம்",
    "https://ta.wikipedia.org/wiki/தமிழ்_இலக்கணம்",
    "https://ta.wikipedia.org/wiki/திருக்குறள்",
    "https://ta.wikipedia.org/wiki/தமிழ்_வரலாறு",
    "https://ta.wikipedia.org/wiki/தமிழ்_நாகரிகம்",
    # Engineering & Technology
    "https://ta.wikipedia.org/wiki/கணினியியல்",
    "https://ta.wikipedia.org/wiki/மின்னணுவியல்",
    "https://ta.wikipedia.org/wiki/பொறியியல்",
    "https://ta.wikipedia.org/wiki/தகவல்_தொழில்நுட்பம்",
]

WIKI_BASE     = "https://ta.wikipedia.org"
WIKI_MAX_FETCH = 600   # crawl up to 600 Wikipedia pages


def wiki_url_valid(url: str) -> bool:
    if "ta.wikipedia.org/wiki/" not in url:
        return False
    # Exclude meta-pages
    bad = [
        "விசேஷம்:", "உதவி:", "விக்கிப்பீடியா:", "வார்ப்புரு:",
        "படிமம்:", "கோப்பு:", "Special:", "Help:", "Wikipedia:",
        "Template:", "File:", "Image:", "Category:", "பகுப்பு:",
        "Talk:", "User:", "Portal:",
    ]
    return not any(b in url for b in bad)


# ─────────────────────────────────────────────
#  Static (non-crawl) sites
# ─────────────────────────────────────────────
STATIC_SITES = [
    {
        "name": "Tamil Virtual Academy",
        "sitemaps": ["https://www.tamilvu.org/sitemap.xml"],
        "archive_pages": [
            "https://www.tamilvu.org/library/",
            "https://www.tamilvu.org/slet/",
            "https://www.tamilvu.org/ta/",
            "https://www.tamilvu.org/courses/",
        ],
        "url_patterns": [r"/library", r"/slet", r"/ta/", r"/courses", r"/article"],
        # tamilvu uses many layouts — try all common containers
        "selectors": [
            "#content-area", "#contentarea", ".content-area",
            "#article-body", ".article-body",
            "#main-content", ".main-content",
            ".col-content", ".course-content",
            "td.content", "td#content",
            "#content", "article", "main",
            # last resort: any <p> on the page
            "p",
        ],
    },
    {
        "name": "Project Madurai",
        "sitemaps": [],
        "archive_pages": ["https://www.projectmadurai.org/"],
        "url_patterns": [r"/pm_etexts", r"/pmuni", r"pmwork"],
        "selectors": ["pre", "body", ".text", "article", "main"],
    },
    {
        "name": "Tamil University Thanjavur",
        "sitemaps": ["https://www.tamiluniversity.ac.in/sitemap.xml"],
        "archive_pages": [
            "https://www.tamiluniversity.ac.in/journal/",
            "https://www.tamiluniversity.ac.in/publications/",
            "https://www.tamiluniversity.ac.in/research/",
        ],
        "url_patterns": [r"/journal", r"/publication", r"/research", r"/article"],
        "selectors": [".content p", "article p", "#main-content p", "main p", "p"],
    },
    {
        "name": "TNAU Research",
        "sitemaps": ["https://www.tnau.ac.in/sitemap.xml"],
        "archive_pages": [
            "https://agritech.tnau.ac.in/",
            "https://www.tnau.ac.in/research/",
        ],
        "url_patterns": [r"/research", r"/publication", r"/article"],
        "selectors": [".content p", "article p", "main p", "#content p", "p"],
    },
]

# ─────────────────────────────────────────────
#  Vocabulary lists
# ─────────────────────────────────────────────

# ≥ 2 of these needed — broad enough to catch encyclopaedic prose
ACADEMIC_KEYWORDS = [
    # Research / scholarship
    "ஆய்வு", "ஆராய்ச்சி", "கட்டுரை", "பகுப்பாய்வு",
    "கோட்பாடு", "கருதுகோள்", "மேற்கோள்", "சான்று",
    "முன்னுரை", "முடிவுரை", "நோக்கம்", "முறைமை",
    # Academic connectives (Wikipedia uses these heavily)
    "எனவே", "ஆகவே", "மேலும்", "இதனால்", "இவ்வாறு",
    "அதாவது", "இதன் அடிப்படையில்", "இதன் படி",
    # Disciplines
    "மொழியியல்", "இலக்கணம்", "இலக்கியம்", "தொல்லியல்",
    "வரலாறு", "பண்பாடு", "நாகரிகம்", "சமூகவியல்",
    "தத்துவம்", "உளவியல்", "பொருளாதாரம்",
    "அறிவியல்", "இயற்பியல்", "வேதியியல்", "உயிரியல்",
    "கணிதம்", "புவியியல்", "கணினியியல்", "பொறியியல்",
    "மருத்துவம்", "வேளாண்மை", "சுற்றுச்சூழல்",
    # Wikipedia-style encyclopaedic markers
    "வரையறை", "விளக்கம்", "வகைப்பாடு", "பண்பு",
    "இனம்", "அமைப்பு", "கட்டமைப்பு", "செயல்முறை",
    "கண்டுபிடிப்பு", "வளர்ச்சி", "தோற்றம்", "தாக்கம்",
    "முக்கியத்துவம்", "பங்களிப்பு", "ஆவணம்",
]

# Reject if ≥ 3 of these appear (entertainment / interview / politics)
NON_ACADEMIC_MARKERS = [
    "திரைப்படம்", "சினிமா", "விமர்சனம்", "ரிலீஸ்",
    "நடிகர்", "நடிகை", "இயக்குனர்",
    "கூறினார்", "தெரிவித்தார்", "சொன்னார்",
    "நேர்காணல்", "பேட்டி",
    "தேர்தல்", "கட்சி", "வாக்கு",
    "வைரல்", "ட்ரெண்டிங்",
]

BLOCKED_HOSTS = [
    "consent.google.com", "translate.google", "facebook.", "twitter.",
    "youtube.", "instagram.", "tiktok.", "linkedin.", "reddit.",
    "accounts.google.com",
]

# ─────────────────────────────────────────────
#  Regex constants
# ─────────────────────────────────────────────
TAMIL_CHAR         = re.compile(r"[\u0B80-\u0BFF]")
HAS_LATIN          = re.compile(r"[A-Za-z]")
HAS_DIGIT          = re.compile(r"[0-9]")
MULTISPACE         = re.compile(r"\s+")
SENTENCE_END_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")
SITEMAP_LOC        = re.compile(r"<loc>\s*(https?://[^<]+)\s*</loc>")
SITEMAP_LASTMOD    = re.compile(r"<lastmod>\s*([^<]+)\s*</lastmod>")
SITEMAP_INDEX_TAG  = re.compile(r"<sitemap>.*?</sitemap>", re.DOTALL)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U0000FE00-\U0000FE0F\U0000200D\U00002640-\U00002642"
    "\U000023CF-\U000023F3\U0000231A-\U0000231B\U00002B05-\U00002B07"
    "\U00002B1B-\U00002B1C\U00002B50\U00002B55\U000025AA-\U000025AB"
    "\U000025FB-\U000025FE\U00003030\U0000303D\U00003297\U00003299"
    "]+", flags=re.UNICODE,
)

FRAGMENT_PATTERNS = [
    re.compile(r"^[\s\.\,\:\;\-\*\#]+$"),
    re.compile(r"^\s*[\.\,\:\;\-\*\#\>\<\+\=\(\)\[\]\{\}]"),
    re.compile(r"^[\s]*\d"),
    re.compile(r"\.\s*\.\s*\."),
]

COOKIE_PRIVACY_MARKERS = [
    "cookie", "cookies", "privacy", "consent", "gdpr",
    "do not sell", "personalized ads", "analytics",
    "strictly necessary", "third parties",
]


# ─────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────
def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        print("[INFO] Loading xlm-roberta-base tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")
        print("[INFO] Tokenizer loaded.")
    return TOKENIZER


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ─────────────────────────────────────────────
#  Text cleaning
# ─────────────────────────────────────────────
def deep_clean_text(text: str) -> str:
    if not text:
        return ""
    text = EMOJI_PATTERN.sub("", text)
    cleaned = []
    for ch in text:
        if "\u0B80" <= ch <= "\u0BFF":
            cleaned.append(ch)
        elif ch in (" ", "\t"):
            cleaned.append(" ")
        elif ch in (".", "!", "?", "\u0964", "\u0965", ",", ";", ":"):
            cleaned.append(ch)
    text = "".join(cleaned)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"^\s*[.,!?]+\s*", "", text)
    return MULTISPACE.sub(" ", text).strip()


def looks_like_cookie_notice(text: str) -> bool:
    low  = text.lower()
    hits = sum(1 for w in COOKIE_PRIVACY_MARKERS if w in low)
    tam  = len(TAMIL_CHAR.findall(text))
    eng  = len(re.findall(r"[A-Za-z]", text))
    return hits >= 2 or (tam == 0 and eng > 50)


def tamil_ratio(text: str) -> float:
    tam         = len(TAMIL_CHAR.findall(text))
    all_letters = len(re.findall(r"[A-Za-z\u0B80-\u0BFF]", text))
    return (tam / all_letters) if all_letters else 0.0


def clean_para(p: str) -> str:
    return MULTISPACE.sub(" ", (p or "").replace("\u00A0", " ")).strip()


def keep_para(p: str) -> bool:
    if not p or len(p) < 40 or looks_like_cookie_notice(p):
        return False
    return tamil_ratio(p) >= 0.60 or len(TAMIL_CHAR.findall(p)) >= 15


# ─────────────────────────────────────────────
#  Sentence validators
# ─────────────────────────────────────────────
def is_valid_sentence(s: str) -> bool:
    if not s or len(s.strip()) < 15:
        return False
    if len(TAMIL_CHAR.findall(s)) < 10:
        return False
    first_char = s.lstrip()[0] if s.strip() else ""
    if first_char and not ("\u0B80" <= first_char <= "\u0BFF"):
        return False
    if HAS_LATIN.search(s) or HAS_DIGIT.search(s):
        return False
    for pat in FRAGMENT_PATTERNS:
        if pat.search(s):
            return False
    return True


# ─────────────────────────────────────────────
#  Academic content gate  (relaxed for recall)
# ─────────────────────────────────────────────
def is_academic_content(text: str) -> bool:
    cleaned = deep_clean_text(text)
    if not cleaned or len(TAMIL_CHAR.findall(cleaned)) < 80:
        return False

    # Gate 1 — must match at least 2 academic vocab words
    kw_hits = sum(1 for kw in ACADEMIC_KEYWORDS if kw in cleaned)
    if kw_hits < 2:
        return False

    # Gate 2 — reject heavy entertainment / interview content
    non_hits = sum(1 for m in NON_ACADEMIC_MARKERS if m in cleaned)
    if non_hits >= 3:
        return False

    # Gate 3 — academic prose has longer sentences than news
    sentences = [s.strip() for s in re.split(r"[.!?।॥]", cleaned) if s.strip()]
    if len(sentences) < 3:
        return False
    avg_len = sum(len(s) for s in sentences) / len(sentences)
    if avg_len < 20:
        return False

    return True


# ─────────────────────────────────────────────
#  Sentence extraction
# ─────────────────────────────────────────────
def extract_clean_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = deep_clean_text(text)
    if not text:
        return []
    sentences, seen = [], set()
    for s in SENTENCE_END_SPLIT.split(text):
        s = s.strip(" \t\r\n\u200c\u200b")
        if not s:
            continue
        s = deep_clean_text(s)
        if not s:
            continue
        if not re.search(r"[.!?।॥]$", s):
            s += "."
        key = s[:120]
        if key in seen:
            continue
        seen.add(key)
        if is_valid_sentence(s):
            sentences.append(s)
    return sentences


# ─────────────────────────────────────────────
#  Date helpers
# ─────────────────────────────────────────────
def parse_date_from_html(html: str):
    soup = BeautifulSoup(html, "lxml")
    for css, attr in [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="pubdate"]',                    "content"),
        ('meta[name="date"]',                       "content"),
        ('meta[name="DC.date"]',                    "content"),
        ('meta[name="citation_publication_date"]',  "content"),
        ('meta[itemprop="datePublished"]',          "content"),
    ]:
        m = soup.select_one(css)
        if m and m.get(attr):
            try:
                return dateparser.parse(m.get(attr), dayfirst=True, fuzzy=True)
            except Exception:
                pass
    return None


def _in_date_range(date_str: str) -> bool:
    try:
        dt = dateparser.parse(date_str.strip(), fuzzy=True)
        if dt:
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return LOWER_DATE <= dt < UPPER_DATE
    except Exception:
        pass
    return False


def _url_date_hint(url: str) -> bool:
    m = re.search(r"/(20\d{2})[/\-_]", url)
    if m:
        return 2000 <= int(m.group(1)) <= 2021
    return True


# ─────────────────────────────────────────────
#  HTML → sentences  (Wikipedia-aware)
# ─────────────────────────────────────────────
def extract_sentences_from_html(
    html: str,
    selectors: list[str],
    min_chars: int,
    is_wikipedia: bool = False,
) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate tags
    for tag in soup.find_all([
        "script", "style", "nav", "footer", "header", "aside",
        "form", "iframe", "noscript", "button", "input",
        "select", "textarea", "svg", "figure", "figcaption",
        "table", "cite", "sup", "sub",
    ]):
        tag.decompose()

    # Wikipedia-specific: remove infoboxes, TOC, references
    if is_wikipedia:
        for tag in soup.find_all(["div", "table"], {"class": [
            "infobox", "navbox", "reflist", "toc", "metadata",
            "thumb", "mw-editsection", "noprint", "hatnote",
        ]}):
            tag.decompose()
        # Get the main content div
        content_root = (
            soup.find("div", {"id": "mw-content-text"})
            or soup.find("div", {"class": "mw-parser-output"})
        )
        if content_root:
            paras = []
            for p in content_root.find_all("p"):
                txt = clean_para(p.get_text(" ", strip=True))
                if keep_para(txt):
                    paras.append(txt)
            if paras:
                full = MULTISPACE.sub(" ", " ".join(paras)).strip()
                if len(full) >= min_chars and is_academic_content(full):
                    return extract_clean_sentences(full)
                return []

    # Non-Wikipedia: try site-specific selectors
    paras: list[str] = []
    for css in selectors:
        nodes = soup.select(css)
        if not nodes:
            continue
        for node in nodes:
            txt = clean_para(node.get_text(" ", strip=True))
            if keep_para(txt):
                paras.append(txt)
        if paras:
            break

    # Generic fallback
    if not paras:
        for css in ["article p", ".content p", "main p", "p"]:
            nodes = soup.select(css)
            if not nodes:
                continue
            for p in nodes:
                txt = clean_para(p.get_text(" ", strip=True))
                if keep_para(txt):
                    paras.append(txt)
            if paras:
                break

    if not paras:
        return []

    full = MULTISPACE.sub(" ", " ".join(paras).replace("\n", " ")).strip()
    if len(full) < min_chars:
        return []
    if not is_academic_content(full):
        return []
    return extract_clean_sentences(full)


# ─────────────────────────────────────────────
#  HTTP session
# ─────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2, connect=2, read=2, backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("http://",  HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ta-IN,ta;q=0.9,en-US;q=0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Cache-Control": "no-cache",
    })
    return s


# ─────────────────────────────────────────────
#  Wikipedia crawler  (BFS from seed pages)
# ─────────────────────────────────────────────
def crawl_wikipedia(
    session: requests.Session,
    seeds: list[str],
    max_fetch: int,
    workers: int,
) -> list[tuple[str, list[str]]]:
    """Return list of (url, sentences) for Wikipedia pages."""
    visited = set(seeds)
    queue   = deque(seeds)
    results: list[tuple[str, list[str]]] = []
    fetched = 0

    print(f"  [WIKI] Starting BFS from {len(seeds)} seed pages (max {max_fetch} pages) …")

    while queue and fetched < max_fetch:
        batch = [queue.popleft() for _ in range(min(workers, len(queue)))]
        fetched += len(batch)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(_fetch_wiki_page, session, url): url for url in batch}
            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    html, linked_urls = future.result()
                except Exception:
                    continue
                if html:
                    sents = extract_sentences_from_html(
                        html, [], 200, is_wikipedia=True
                    )
                    if sents:
                        results.append((url, sents))
                for lu in linked_urls:
                    if lu not in visited and wiki_url_valid(lu):
                        visited.add(lu)
                        queue.append(lu)

        if fetched % 100 == 0:
            print(f"  [WIKI] {fetched} pages fetched, {len(results)} produced sentences")

    print(f"  [WIKI] Done: {fetched} pages → {len(results)} usable articles")
    return results


def _fetch_wiki_page(
    session: requests.Session,
    url: str,
) -> tuple[str, list[str]]:
    """Fetch a Wikipedia page and return (html, list_of_linked_urls)."""
    time.sleep(CRAWL_DELAY)
    try:
        resp = session.get(url, timeout=15, allow_redirects=True)
    except Exception:
        return "", []
    if resp.status_code != 200:
        return "", []

    soup = BeautifulSoup(resp.text, "lxml")
    # Collect internal links for BFS
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/"):
            full = WIKI_BASE + href.split("?")[0].split("#")[0]
            if wiki_url_valid(full):
                links.append(full)
    return resp.text, links


# ─────────────────────────────────────────────
#  Static site URL collection + article fetch
# ─────────────────────────────────────────────
def _is_url_match(url: str, patterns: list[str]) -> bool:
    ul = url.lower()
    return any(re.search(pat, ul) for pat in patterns)


def fetch_sitemap_urls(
    session: requests.Session,
    sitemap_url: str,
    patterns: list[str],
    depth: int = 0,
) -> list[str]:
    if depth > 2:
        return []
    try:
        resp = session.get(sitemap_url, timeout=15)
        if resp.status_code != 200:
            return []
        xml = resp.text
    except Exception as e:
        print(f"    [WARN] Sitemap {sitemap_url}: {e}")
        return []

    if "<sitemapindex" in xml or SITEMAP_INDEX_TAG.search(xml):
        collected = []
        for child in SITEMAP_LOC.findall(xml):
            child = child.strip()
            ym = re.search(r"20(\d{2})", child)
            if ym and not (2000 <= 2000 + int(ym.group(1)) <= 2021):
                continue
            cl = child.lower()
            if any(k in cl for k in [
                "research", "science", "education", "article", "journal",
                "publication", "language", "literature", "library",
            ]) or depth == 0:
                collected.extend(fetch_sitemap_urls(session, child, patterns, depth + 1))
        return collected

    locs     = SITEMAP_LOC.findall(xml)
    lastmods = SITEMAP_LASTMOD.findall(xml)
    urls     = []
    for i, loc in enumerate(locs):
        loc = loc.strip()
        if not loc.startswith("http"):
            continue
        if not _is_url_match(loc, patterns):
            continue
        if lastmods and i < len(lastmods):
            if not _in_date_range(lastmods[i]):
                continue
        elif not _url_date_hint(loc):
            continue
        urls.append(loc)
    return urls


def extract_links_from_archive_page(
    session: requests.Session,
    page_url: str,
    patterns: list[str],
) -> list[str]:
    try:
        resp = session.get(page_url, timeout=12)
        if resp.status_code != 200:
            return []
        soup        = BeautifulSoup(resp.text, "lxml")
        base_domain = urlparse(page_url).netloc
        links       = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/"):
                href = f"{urlparse(page_url).scheme}://{base_domain}{href}"
            elif not href.startswith("http"):
                href = urljoin(page_url, href)
            if not href.startswith("http"):
                continue
            if urlparse(href).netloc != base_domain:
                continue
            path = urlparse(href).path
            if len(path) < 5:
                continue
            if not _is_url_match(href, patterns):
                continue
            if not _url_date_hint(href):
                continue
            links.append(href)
        return list(set(links))
    except Exception as e:
        print(f"    [WARN] Archive {page_url}: {e}")
        return []


def collect_static_site_urls(session: requests.Session, site: dict) -> list[str]:
    patterns = site.get("url_patterns", [])
    urls: list[str] = []
    for sm_url in site.get("sitemaps", []):
        print(f"    [SITEMAP] {sm_url}")
        found = fetch_sitemap_urls(session, sm_url, patterns)
        print(f"    -> {len(found)} URLs")
        urls.extend(found)
        if urls:
            break
    if not urls:
        for arch in site.get("archive_pages", []):
            print(f"    [ARCHIVE] {arch}")
            found = extract_links_from_archive_page(session, arch, patterns)
            print(f"    -> {len(found)} URLs")
            urls.extend(found)
    return list(dict.fromkeys(urls))


def fetch_static_article(
    session: requests.Session,
    url: str,
    selectors: list[str],
    min_chars: int,
) -> list[str] | None:
    host = urlparse(url).netloc.lower()
    if any(b in host for b in BLOCKED_HOSTS):
        return None
    time.sleep(CRAWL_DELAY)
    try:
        resp = session.get(url, timeout=15, allow_redirects=True)
    except Exception:
        return None
    if resp.status_code >= 400 or not resp.text:
        return None
    try:
        pub_dt = parse_date_from_html(resp.text)
        if pub_dt:
            if pub_dt.tzinfo:
                pub_dt = pub_dt.astimezone(timezone.utc).replace(tzinfo=None)
            if pub_dt >= UPPER_DATE:
                return None
    except Exception:
        pass
    sents = extract_sentences_from_html(resp.text, selectors, min_chars)
    return sents if len(sents) >= 3 else None


# ─────────────────────────────────────────────
#  Sample builder
# ─────────────────────────────────────────────
def build_sample(
    sentences: list[str],
    start_idx: int,
    min_tok: int,
    max_tok: int,
) -> tuple[str | None, int]:
    current_text = ""
    current_tok  = 0
    idx          = start_idx
    while idx < len(sentences):
        sent      = sentences[idx]
        idx      += 1
        candidate = (current_text + " " + sent).strip() if current_text else sent
        tok_count = count_tokens(candidate)
        if tok_count > max_tok:
            if min_tok <= current_tok <= max_tok:
                return current_text, idx
            continue
        current_text = candidate
        current_tok  = tok_count
        if min_tok <= current_tok <= max_tok:
            return current_text, idx
    if current_text and min_tok <= current_tok <= max_tok:
        return current_text, idx
    return None, idx


def samples_from_sentences(
    sentences: list[str],
    min_tok: int,
    max_tok: int,
) -> list[str]:
    results = []
    start   = 0
    while start < len(sentences):
        text, start = build_sample(sentences, start, min_tok, max_tok)
        if text is None:
            break
        text = deep_clean_text(text)
        text = MULTISPACE.sub(" ", text).strip()
        if not text:
            continue
        if not re.search(r"[.!?।॥]$", text):
            text += "."
        tok = count_tokens(text)
        if not (min_tok <= tok <= max_tok):
            continue
        if HAS_LATIN.search(text) or HAS_DIGIT.search(text):
            continue
        if tamil_ratio(text) < 0.70:
            continue
        results.append(text)
    return results


def write_sample(fh, text: str, tok: int, sample_count: int, num_samples: int, label: str):
    fh.write(json.dumps({"text": text, "xlm_roberta_tokens": tok}, ensure_ascii=False) + "\n")
    fh.flush()
    print(f"  [SAMPLE {sample_count:>4}/{num_samples}] {tok} tok | {label}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile",     type=str, default=OUTPUT_SAMPLES_FILE)
    ap.add_argument("--min-chars",   type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--min-tok",     type=int, default=MIN_TOK)
    ap.add_argument("--max-tok",     type=int, default=MAX_TOK)
    ap.add_argument("--workers",     type=int, default=MAX_WORKERS)
    ap.add_argument("--wiki-pages",  type=int, default=WIKI_MAX_FETCH,
                    help="Max Wikipedia pages to crawl")
    return ap.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    args        = parse_args()
    random.seed(RANDOM_SEED)
    outfile     = Path(args.outfile)
    num_samples = args.num_samples
    min_tok     = args.min_tok
    max_tok     = args.max_tok

    print("=" * 60)
    print(" Tamil Academic Writing Corpus Scraper")
    print("=" * 60)
    print(f"[CONFIG] Output      : {outfile}")
    print(f"[CONFIG] Target      : {num_samples} samples  ({min_tok}–{max_tok} tokens)")
    print(f"[CONFIG] Wiki pages  : up to {args.wiki_pages}")
    print(f"[CONFIG] Workers     : {args.workers}")
    print("=" * 60)

    get_tokenizer()
    session = make_session()

    # All collected (url, sentences) pairs
    all_article_sentences: list[tuple[str, list[str]]] = []

    # ── Source 1: Tamil Wikipedia (primary) ─────────────────────────────────
    print(f"\n[SOURCE 1/2] Tamil Wikipedia (BFS crawl) …")
    wiki_results = crawl_wikipedia(session, WIKI_SEEDS, args.wiki_pages, args.workers)
    all_article_sentences.extend(wiki_results)
    print(f"  → {len(wiki_results)} Wikipedia articles with valid sentences")

    # ── Source 2: Static academic sites ─────────────────────────────────────
    print(f"\n[SOURCE 2/2] Static academic sites …")
    static_urls_meta: list[tuple[str, list[str]]] = []
    for site in STATIC_SITES:
        print(f"\n  [{site['name']}]")
        urls = collect_static_site_urls(session, site)
        print(f"  → {len(urls)} candidate URLs")
        sel = site.get("selectors", ["p"])
        for url in urls:
            static_urls_meta.append((url, sel))

    deduped  = list(dict.fromkeys(u for u, _ in static_urls_meta))
    meta_map = {u: s for u, s in static_urls_meta}
    random.shuffle(deduped)
    print(f"\n  Fetching {len(deduped)} static URLs with {args.workers} threads …")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_static_article, session, url, meta_map[url], args.min_chars): url
            for url in deduped
        }
        for future in as_completed(futures):
            sents = future.result()
            if sents:
                all_article_sentences.append((futures[future], sents))

    print(f"\n  → {len(all_article_sentences)} total articles with valid sentences")

    # ── Build samples ────────────────────────────────────────────────────────
    print(f"\n[PHASE 2] Building {min_tok}–{max_tok} token samples …")
    print("=" * 60)

    random.shuffle(all_article_sentences)
    sample_count   = 0
    total_articles = 0
    outfile_handle = outfile.open("w", encoding="utf-8")

    try:
        for url, sentences in all_article_sentences:
            if sample_count >= num_samples:
                break
            new_samples = samples_from_sentences(sentences, min_tok, max_tok)
            if not new_samples:
                continue
            total_articles += 1
            for text in new_samples:
                if sample_count >= num_samples:
                    break
                sample_count += 1
                tok = count_tokens(text)
                write_sample(outfile_handle, text, tok, sample_count, num_samples,
                             url[:60])

        # Second pass with shuffled offsets
        if sample_count < num_samples:
            print(f"\n[INFO] {sample_count}/{num_samples} — running second pass …")
            sents_only = [s for _, s in all_article_sentences]
            random.shuffle(sents_only)
            for pass_num in range(3):
                if sample_count >= num_samples:
                    break
                for art_sents in sents_only:
                    if sample_count >= num_samples:
                        break
                    offsets = list(range(0, len(art_sents), max(1, pass_num + 1)))
                    random.shuffle(offsets)
                    for start in offsets:
                        if sample_count >= num_samples:
                            break
                        text, _ = build_sample(art_sents, start, min_tok, max_tok)
                        if text is None:
                            continue
                        text = deep_clean_text(text)
                        text = MULTISPACE.sub(" ", text).strip()
                        if not text:
                            continue
                        if not re.search(r"[.!?।॥]$", text):
                            text += "."
                        tok = count_tokens(text)
                        if not (min_tok <= tok <= max_tok):
                            continue
                        if HAS_LATIN.search(text) or HAS_DIGIT.search(text):
                            continue
                        if tamil_ratio(text) < 0.70:
                            continue
                        sample_count += 1
                        write_sample(outfile_handle, text, tok, sample_count,
                                     num_samples, f"pass {pass_num + 2}")
    finally:
        outfile_handle.close()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Articles used       : {total_articles}")
    print(f"  Samples written     : {sample_count}")
    print(f"  Output file         : {outfile.resolve()}")

    if sample_count > 0:
        tok_counts = [
            json.loads(ln)["xlm_roberta_tokens"]
            for ln in outfile.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if tok_counts:
            avg = sum(tok_counts) / len(tok_counts)
            print(f"  Token stats         : "
                  f"min={min(tok_counts)}  max={max(tok_counts)}  avg={avg:.1f}")

    if sample_count < num_samples:
        print(f"\n  [WARN] Only {sample_count}/{num_samples} samples collected.")
        print(f"  Try: --wiki-pages {args.wiki_pages * 2}")


if __name__ == "__main__":
    main()