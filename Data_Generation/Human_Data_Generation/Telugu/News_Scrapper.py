"""
Telugu News Scraper (requests + BeautifulSoup)
-----------------------------------------------
Scrapes Telugu news articles from Google News (2012-2021),
cleans text thoroughly (no symbols, emoji, stars, fragments),
then assembles samples of 400-500 XLM-RoBERTa tokens.

Key guarantees:
  - Each sample contains ONLY complete, meaningful Telugu sentences.
  - Sentences within a sample come from the SAME article (coherent passages).
  - No emoji, no *, no :-, no bullets, no stray punctuation, no noise.
  - Token count is measured by xlm-roberta-base tokenizer.
  - Each sample is written to the JSONL file immediately (progressive output).

Requirements:
    pip install requests beautifulsoup4 lxml python-dateutil transformers sentencepiece

Usage:
    python telugu_scraper.py [--outfile OUTPUT.jsonl] [--delay 2.0]
"""

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote_plus
from datetime import datetime, timezone
from dateutil import parser as dateparser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

# ──────────────────────────── CONFIGURATION ────────────────────────────

NUM_SAMPLES = 1000
MIN_TOK = 400
MAX_TOK = 500
OUTPUT_SAMPLES_FILE = "telugu_1000_400-500.jsonl"
RANDOM_SEED = 42

TOKENIZER = None  # lazy-loaded

TERMS = [
    # Government Schemes & Welfare
    "రైతు బంధు", "యువ నిధి", "వసతి దీవెన", "నిరుద్యోగ భృతి", "పింఛన్లు",
    "ఎస్సీ, ఎస్టీ సబ్ ప్లాన్", "మిషన్ కాకతీయ", "అమ్మ ఒడి", "నేషనల్ హెల్త్ మిషన్", "ఆరోగ్య లక్ష్మి",
    # Agriculture
    "రైతులు", "రైతు సంక్షేమం", "సాగునీరు", "పంట రుణాలు", "పంట కొనుగోలు",
    "సహజ వ్యవసాయం", "పెట్టుబడి వ్యవసాయం", "పంట నష్టం", "ఎరువులు", "సాగు సామగ్రి",
    # Infrastructure
    "అమరావతి", "హైటెక్ సిటీ", "హైదరాబాద్ మెట్రో", "విశాఖ ఉక్కు కర్మాగారం", "ఇంటి స్థలాలు",
    "రోడ్లు", "ఓవర్ బ్రిడ్జి", "సొరంగాలు", "అండర్ పాస్", "రైల్వే స్టేషన్",
    # Education
    "స్కూల్ విద్య", "పాఠశాల నిర్మాణాలు", "టీఎస్ ఆర్టీసీ నోటిఫికేషన్", "పోలీస్ నోటిఫికేషన్", "ఇంటర్ పరీక్షలు",
    "ఈమెయిన్స్", "నీట్ పరీక్ష", "జేఈఈ పరీక్ష", "పీజీ కౌన్సిలింగ్", "ప్రభుత్వ కళాశాలలు",
    # Health
    "జిల్లా ఆసుపత్రి", "ప్రాథమిక ఆరోగ్య కేంద్రం", "కోవిడ్ వ్యాక్సిన్", "ఆరోగ్యమిత్ర", "మహిళా ఆరోగ్యం",
    "ప్రసూతి సంరక్షణ", "డయాలసిస్ సౌకర్యాలు", "క్యాన్సర్ చికిత్స", "మానసిక ఆరోగ్యం", "పిల్లల ఆరోగ్యం",
    # Industry & Economy
    "విశాఖ పోర్ట్", "కృష్ణా జిల్లా పరిశ్రమలు", "ఫార్మా సెక్టార్", "ఐటి పార్క్", "మైక్రో ఫైనాన్స్",
    "ఎస్బీఐ లోన్లు", "ఆర్థిక సహాయం", "స్టార్టప్ ఇండియా", "స్వయం ఉపాధి", "చిన్న వ్యాపారాలు",
    # Crime & Law
    "పోలీస్ కేసు", "మర్డర్ కేసు", "దొంగతనం", "సైబర్ నేరాలు", "మహిళా సంఘటనలు",
    "పిల్లలపై దాడి", "మాదక దుర్వినియోగం", "కుల ఘర్షణలు", "పోలీస్ చర్య",
    # Social Welfare
    "మహిళా సంక్షేమం", "బాలికల విద్య", "మాతృత్వ మద్దతు", "అనాథ పిల్లలు", "దివ్యాంగులు",
    "పేదరిక నిర్మూలన", "మద్యం నిషేధం", "కుల సమస్యలు", "పెళ్లిళ్లు",
    # Natural Disasters & Environment
    "వరదలు", "ఎడతెరిపి లేని వర్షాలు", "పొడుపు తుగ్గులు", "ప్రకృతి విపత్తు", "పునరుద్ధరణ",
    "అడవులు", "జలాశయాలు", "పర్యావరణ పరిరక్షణ", "పచ్చదనం", "సౌర శక్తి",
    # Technology & Digital
    "డిజిటల్ ఇండియా", "ఆధార్ లింకేజ్", "ఇ-గవర్నెన్స్", "మొబైల్ యాప్",
    "సైబర్ సురక్ష", "ఎమర్జెన్సీ హెల్ప్‌లైన్", "స్మార్ట్ సిటీ", "టెలిమెడిసిన్",
    # Sports & Culture
    "ఐపీఎల్", "క్రికెట్ టోర్నమెంట్", "ఓలంపిక్స్", "సినిమా పరిశ్రమ",
    "సాహిత్య సభ", "పండుగలు", "సంస్కృతి",
    # Cities & Regions
    "విజయవాడ", "గుంటూరు", "రాజమండ్రి", "తిరుపతి", "నెల్లూరు", "కర్నూలు", "అనంతపురం",
    "హైదరాబాద్", "వరంగల్", "నిజామాబాద్", "మెదక్", "రంగారెడ్డి",
    # Disaster Relief
    "తుఫాను", "సుకూన్ తుఫాను", "హారికేన్", "ప్రాంతం పునరుద్ధరణ", "రిలీఫ్ క్యాంప్",
    "ప్రభుత్వ సహాయం", "ఎమర్జెన్సీ సేవలు", "రెడ్ క్రాస్", "స్పందన బృందం", "ప్రమాదం",
    # Religion & Temples
    "తిరుమల ఆలయం", "శ్రీశైలం", "కామాక్షి ఆలయం", "మహాశివరాత్రి", "స్వామి వారి ఉత్సవాలు",
    "పుణ్యక్షేత్రాలు", "పూజలు", "యాత్రలు", "ధార్మిక సేవలు", "మతపరమైన సమావేశాలు",
    # Public Issues & Employment
    "ఉద్యోగ సంఘాలు", "ట్రాన్స్ జెండర్ హక్కులు", "ప్రజా సమస్యలు", "రోడ్డు ప్రమాదాలు", "ప్రభుత్వ ఉద్యోగులు",
    "పెన్షన్ సమస్యలు", "ప్రజా సంఘాలు", "జిల్లా కలెక్టర్",
]

DEFAULT_DELAY = 2.0
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_PER_TERM = 200
DEFAULT_MIN_CHARS = 140

LOWER_DATE = datetime(2012, 1, 1)
UPPER_DATE = datetime(2022, 1, 1)

# ──────────────────────────── REGEX PATTERNS ────────────────────────────

TELUGU_CHAR = re.compile(r"[\u0C00-\u0C7F]")
HAS_LATIN = re.compile(r"[A-Za-z]")
HAS_DIGIT = re.compile(r"[0-9]")
MULTISPACE = re.compile(r"\s+")
SENTENCE_END_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")

# Emoji regex (covers most emoji ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002640-\U00002642"
    "\U000023CF-\U000023F3"
    "\U0000231A-\U0000231B"
    "\U00002B05-\U00002B07"
    "\U00002B1B-\U00002B1C"
    "\U00002B50\U00002B55"
    "\U000025AA-\U000025AB"
    "\U000025FB-\U000025FE"
    "\U00003030\U0000303D"
    "\U00003297\U00003299"
    "]+",
    flags=re.UNICODE
)

# Fragment patterns that indicate noisy/broken sentences
FRAGMENT_PATTERNS = [
    re.compile(r"^[\s\.\,\:\;\-\*\#]+$"),       # only punctuation
    re.compile(r"^\s*[\.\,\:\;\-\*\#\>\<\+\=\(\)\[\]\{\}]"),  # starts with symbol
    re.compile(r"^[\s]*\d"),                      # starts with digit
    re.compile(r"\.\s*\.\s*\."),                  # contains ...
]

COOKIE_PRIVACY_MARKERS = [
    "cookie", "cookies", "privacy", "consent", "gdpr", "cpra", "ccpa",
    "do not sell", "personal information", "advert", "personalized ads",
    "analytics", "performance cookies", "strictly necessary", "third parties",
    "exercise my rights",
]

# ──────────────────────────── TOKENIZER ────────────────────────────


def get_tokenizer():
    """Lazy-load xlm-roberta-base tokenizer."""
    global TOKENIZER
    if TOKENIZER is None:
        print("[INFO] Loading xlm-roberta-base tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")
        print("[INFO] Tokenizer loaded.")
    return TOKENIZER


def count_tokens(text: str) -> int:
    """Count tokens using xlm-roberta-base (no special tokens)."""
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ──────────────────────────── TEXT CLEANING ────────────────────────────


def deep_clean_text(text: str) -> str:
    """
    Aggressively clean text:
    - Remove ALL emoji
    - Remove ALL symbols (*, #, @, ~, bullets, dashes, curly quotes, etc.)
    - Remove ALL Latin letters and Arabic digits
    - Keep ONLY: Telugu characters, spaces, commas, periods, !, ?, Telugu purna virama
    """
    if not text:
        return ""

    # Step 1: Remove emoji
    text = EMOJI_PATTERN.sub("", text)

    # Step 2: Remove characters by Unicode category (symbols, marks we don't want)
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep Telugu letters and marks (Lo, Mc, Mn, Nd in Telugu range)
        if "\u0C00" <= ch <= "\u0C7F":
            cleaned_chars.append(ch)
            continue
        # Keep basic whitespace
        if ch in (" ", "\t"):
            cleaned_chars.append(" ")
            continue
        # Keep sentence-ending punctuation
        if ch in (".", "!", "?", "\u0964", "\u0965", ","):
            cleaned_chars.append(ch)
            continue
        # Drop everything else (Latin, digits, symbols, emoji leftovers)

    text = "".join(cleaned_chars)

    # Step 3: Clean up artifacts
    text = re.sub(r"\.{2,}", ".", text)       # collapse multiple dots
    text = re.sub(r",{2,}", ",", text)        # collapse multiple commas
    text = re.sub(r"^\s*[.,!?]+\s*", "", text)  # remove leading punctuation
    text = MULTISPACE.sub(" ", text).strip()

    return text


def looks_like_cookie_notice(text: str) -> bool:
    low = text.lower()
    hits = sum(1 for w in COOKIE_PRIVACY_MARKERS if w in low)
    tel = len(TELUGU_CHAR.findall(text))
    eng = len(re.findall(r"[A-Za-z]", text))
    return hits >= 2 or (tel == 0 and eng > 50)


def telugu_ratio(text: str) -> float:
    tel = len(TELUGU_CHAR.findall(text))
    all_letters = len(re.findall(r"[A-Za-z\u0C00-\u0C7F]", text))
    return (tel / all_letters) if all_letters else 0.0


def clean_para(p: str) -> str:
    s = (p or "").replace("\u00A0", " ")
    s = MULTISPACE.sub(" ", s).strip()
    return s


def keep_para(p: str) -> bool:
    if not p or looks_like_cookie_notice(p):
        return False
    return telugu_ratio(p) >= 0.70 or len(TELUGU_CHAR.findall(p)) >= 16


def is_valid_sentence(s: str) -> bool:
    """
    A sentence is valid if:
    - Has at least 10 Telugu characters
    - At least 15 characters total
    - Does NOT start with punctuation/symbol
    - Does NOT contain Latin letters or digits (after cleaning)
    - Ends with proper sentence punctuation
    """
    if not s or len(s.strip()) < 15:
        return False

    tel_count = len(TELUGU_CHAR.findall(s))
    if tel_count < 10:
        return False

    # Must start with a Telugu character (not punctuation, not space)
    first_char = s.lstrip()[0] if s.strip() else ""
    if first_char and not ("\u0C00" <= first_char <= "\u0C7F"):
        return False

    # No Latin or digits should remain
    if HAS_LATIN.search(s) or HAS_DIGIT.search(s):
        return False

    # Check fragment patterns
    for pat in FRAGMENT_PATTERNS:
        if pat.search(s):
            return False

    return True


# ──────────────────────────── SENTENCE SPLITTING ────────────────────────────


def extract_clean_sentences(text: str) -> list[str]:
    """
    Clean text and split into valid Telugu sentences.
    Returns only sentences that pass all quality checks.
    """
    if not text:
        return []

    # Deep clean first
    text = deep_clean_text(text)
    if not text:
        return []

    # Split at sentence boundaries
    raw_parts = SENTENCE_END_SPLIT.split(text)

    sentences = []
    seen = set()

    for s in raw_parts:
        s = s.strip(" \t\r\n\u200c\u200b")
        if not s:
            continue

        # Clean again (paranoid)
        s = deep_clean_text(s)
        if not s:
            continue

        # Ensure proper sentence ending
        if not re.search(r"[.!?।॥]$", s):
            s = s + "."

        # Deduplicate by prefix
        key = s[:120]
        if key in seen:
            continue
        seen.add(key)

        # Validate
        if is_valid_sentence(s):
            sentences.append(s)

    return sentences


# ──────────────────────────── HTML PARSING ────────────────────────────


def parse_date_from_html(html: str):
    """Extract publication date from article HTML metadata."""
    soup = BeautifulSoup(html, "lxml")
    meta_selectors = [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[name="publish-date"]', "content"),
        ('meta[name="date"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ('meta[property="og:updated_time"]', "content"),
    ]
    for css, attr in meta_selectors:
        m = soup.select_one(css)
        if m and m.get(attr):
            try:
                return dateparser.parse(m.get(attr), dayfirst=True, fuzzy=True)
            except Exception:
                pass
    for css in [".post-date", ".published-date", ".date", "time", ".article-meta", ".story-date"]:
        n = soup.select_one(css)
        if n:
            txt = n.get("datetime") or n.get_text(" ", strip=True)
            try:
                return dateparser.parse(txt, dayfirst=True, fuzzy=True)
            except Exception:
                pass
    return None


def extract_sentences_from_html(html: str, min_chars: int) -> list[str]:
    """
    Extract clean Telugu sentences from article HTML.
    Returns list of clean sentences from the same article (coherent).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside",
                               "form", "iframe", "noscript", "button", "input",
                               "select", "textarea", "svg", "figure", "figcaption"]):
        tag.decompose()

    containers = [
        "article p", ".article-content p", ".node__content p", ".story-content p",
        ".field--body p", ".content p", "main p", "p",
    ]
    paras = []
    for css in containers:
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

    full_text = " ".join(paras)
    full_text = full_text.replace("\n", " ").replace("\r", " ")
    full_text = MULTISPACE.sub(" ", full_text).strip()

    if len(full_text) < min_chars:
        return []

    return extract_clean_sentences(full_text)


# ──────────────────────────── HTTP SESSION ────────────────────────────


def make_session(lang="te-IN,en-US"):
    s = requests.Session()
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=1.2,
        status_forcelist=[403, 408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"], raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": lang,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Cache-Control": "no-cache",
    })
    return s


def jitter(base: float, pct: float = 0.25) -> float:
    delta = base * pct
    return max(0.2, base + random.uniform(-delta, delta))


# ──────────────────────────── GOOGLE NEWS ────────────────────────────


def google_blocked(html: str) -> bool:
    txt = (html or "").lower()
    markers = [
        "unusual traffic from your computer network",
        "to continue, please verify",
        "detected unusual traffic",
        "sorry, but your computer or network may be sending automated queries",
        "consent.google.com", "recaptcha", "hcaptcha",
    ]
    return any(m in txt for m in markers)


def clean_google_href(href: str) -> str | None:
    if not href:
        return None
    if href.startswith("/url?"):
        qs = parse_qs(urlparse(href).query)
        return qs.get("q", [None])[0]
    return href


def is_google_link(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc
    except Exception:
        return True
    return "google." in netloc or netloc == ""


def extract_links_from_google_html(html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = clean_google_href(a["href"])
        if not href or is_google_link(href):
            continue
        if any(x in href for x in ["accounts.google", "support.google", "policies.google"]):
            continue
        links.add(href)
    return links


def get_next_page_url(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    next_link = soup.find("a", id="pnnext")
    if next_link and next_link.get("href"):
        href = next_link["href"]
        if href.startswith("/"):
            return "https://www.google.com" + href
        return href
    for a in soup.find_all("a", attrs={"aria-label": True}):
        label = a.get("aria-label", "").lower()
        if "next" in label:
            href = a.get("href", "")
            if href.startswith("/"):
                return "https://www.google.com" + href
            return href
    return None


def google_news_links_for_term(session: requests.Session, term: str,
                                delay: float, max_pages: int) -> list[str]:
    base_url = (
        f"https://www.google.com/search?q={quote_plus(term)}"
        f"&tbm=nws&tbs=cdr:1,cd_min:1/1/2012,cd_max:12/31/2021&hl=te"
    )
    all_links = set()
    current_url = base_url

    for page_num in range(max_pages):
        try:
            resp = session.get(current_url, timeout=20)
        except Exception as e:
            print(f"  [WARN] Request failed page {page_num}: {e}")
            break
        if resp.status_code >= 400:
            break

        html = resp.text
        if google_blocked(html):
            print("  [WARN] Google blocked. Waiting 90s...")
            time.sleep(90)
            try:
                resp = session.get(current_url, timeout=20)
                html = resp.text
            except Exception:
                break
            if google_blocked(html):
                print("  [WARN] Still blocked. Skipping term.")
                break

        page_links = extract_links_from_google_html(html)
        all_links |= page_links

        if page_num == 0 and not page_links:
            break

        next_url = get_next_page_url(html)
        if not next_url:
            break
        current_url = next_url
        time.sleep(jitter(delay))

    return list(all_links)


# ──────────────────────── SAMPLE ASSEMBLY ────────────────────────


def build_sample_from_consecutive_sentences(
    sentences: list[str],
    start_idx: int,
    min_tok: int,
    max_tok: int
) -> tuple[str | None, int]:
    """
    Build ONE sample from consecutive sentences of the SAME article.
    Sentences are added one by one. If adding the next sentence exceeds
    max_tok, we stop BEFORE it (no sentence is ever cut).

    Returns:
        (sample_text or None, next_index_to_resume_from)
    """
    current_text = ""
    current_tok_count = 0
    idx = start_idx

    while idx < len(sentences):
        sent = sentences[idx]
        idx += 1

        # Build candidate with this sentence added
        candidate = (current_text + " " + sent).strip() if current_text else sent
        tok_count = count_tokens(candidate)

        if tok_count > max_tok:
            # This sentence would push us over the limit
            if min_tok <= current_tok_count <= max_tok:
                # What we have is valid -- return it
                return current_text, idx
            else:
                # Not enough yet, skip this sentence (it's too long)
                continue

        current_text = candidate
        current_tok_count = tok_count

        # If we're in the valid token range, we can finalize
        if min_tok <= current_tok_count <= max_tok:
            return current_text, idx

    # Ran out of sentences; check if what we accumulated is valid
    if current_text and min_tok <= current_tok_count <= max_tok:
        return current_text, idx

    return None, idx


# ──────────────────────────── BLOCKED HOSTS ────────────────────────────

BLOCKED_HOSTS = [
    "consent.google.com", "translate.google", "facebook.", "twitter.",
    "youtube.", "instagram.", "tiktok.", "linkedin.", "reddit.",
    "accounts.google.com",
]


# ──────────────────────────── MAIN ────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(
        description="Scrape Telugu news (2012-2021), produce 400-500 xlm-roberta-base token samples."
    )
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    ap.add_argument("--outfile", type=str, default=OUTPUT_SAMPLES_FILE)
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--max-per-term", type=int, default=DEFAULT_MAX_PER_TERM)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--min-tok", type=int, default=MIN_TOK)
    ap.add_argument("--max-tok", type=int, default=MAX_TOK)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(RANDOM_SEED)
    delay = max(0.8, float(args.delay))
    outfile = Path(args.outfile)
    num_samples = args.num_samples
    min_tok = args.min_tok
    max_tok = args.max_tok

    print(f"[CONFIG] Output: {outfile}")
    print(f"[CONFIG] Target: {num_samples} samples, {min_tok}-{max_tok} xlm-roberta-base tokens each")
    print(f"[CONFIG] Date range: 2012-01-01 to 2021-12-31")
    print(f"[CONFIG] Delay: {delay}s between pages")

    # Load tokenizer upfront
    get_tokenizer()

    session = make_session()

    sample_count = 0
    seen_urls = set()
    total_articles = 0
    total_sentences = 0

    # Store article sentence pools for second-pass reuse
    all_article_sentences = []

    # Open output file -- samples are written progressively
    outfile_handle = outfile.open("w", encoding="utf-8")

    try:
        print("\n" + "=" * 60)
        print("SCRAPING AND ASSEMBLING (progressive JSONL output)")
        print("=" * 60)

        for term_idx, term in enumerate(TERMS):
            if sample_count >= num_samples:
                break

            print(f"\n[TERM {term_idx + 1}/{len(TERMS)}] {term}")
            links = google_news_links_for_term(session, term, delay, args.max_pages)
            print(f"  Found {len(links)} links")

            saved_for_term = 0

            for url in links:
                if sample_count >= num_samples:
                    break
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                try:
                    resp = session.get(url, timeout=18, allow_redirects=True)
                except Exception:
                    continue
                if resp.status_code >= 400 or not resp.text:
                    continue

                host = urlparse(resp.url).netloc.lower()
                if any(b in host for b in BLOCKED_HOSTS):
                    continue

                # Date filter
                try:
                    pub_dt = parse_date_from_html(resp.text)
                except Exception:
                    pub_dt = None
                if pub_dt is not None:
                    if pub_dt.tzinfo:
                        pub_dt = pub_dt.astimezone(timezone.utc).replace(tzinfo=None)
                    if not (LOWER_DATE <= pub_dt < UPPER_DATE):
                        continue

                # Extract clean sentences from this single article
                sentences = extract_sentences_from_html(resp.text, min_chars=args.min_chars)
                if len(sentences) < 3:
                    continue

                total_articles += 1
                total_sentences += len(sentences)
                saved_for_term += 1

                # Save for potential second pass
                all_article_sentences.append(sentences)

                # Build samples from consecutive sentences of THIS article
                start = 0
                while start < len(sentences) and sample_count < num_samples:
                    sample_text, start = build_sample_from_consecutive_sentences(
                        sentences, start, min_tok, max_tok
                    )
                    if sample_text is None:
                        break

                    # Final paranoid cleanup
                    sample_text = deep_clean_text(sample_text)
                    sample_text = MULTISPACE.sub(" ", sample_text).strip()
                    if not sample_text:
                        continue

                    # Ensure ends with punctuation
                    if not re.search(r"[.!?।॥]$", sample_text):
                        sample_text = sample_text + "."

                    # Final token count verification
                    tok_count = count_tokens(sample_text)
                    if not (min_tok <= tok_count <= max_tok):
                        continue

                    # Validate no noise leaked through
                    if HAS_LATIN.search(sample_text) or HAS_DIGIT.search(sample_text):
                        continue

                    # WRITE to JSONL immediately
                    sample_count += 1
                    obj = {
                        "text": sample_text,
                        "xlm_roberta_tokens": tok_count,
                    }
                    outfile_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    outfile_handle.flush()

                    print(f"  [SAMPLE {sample_count}/{num_samples}] "
                          f"{tok_count} tokens | from article #{total_articles}")

                if saved_for_term >= args.max_per_term:
                    break
                time.sleep(jitter(delay, 0.35))

        # ── Second pass if we still need more samples ──
        if sample_count < num_samples and all_article_sentences:
            print(f"\n[INFO] First pass yielded {sample_count}/{num_samples} samples.")
            print("[INFO] Running second pass over collected articles with different offsets...")

            for pass_num in range(3):
                if sample_count >= num_samples:
                    break

                random.shuffle(all_article_sentences)

                for art_sents in all_article_sentences:
                    if sample_count >= num_samples:
                        break

                    # Try starting at different offsets
                    offsets = list(range(0, len(art_sents), max(1, pass_num + 1)))
                    random.shuffle(offsets)

                    for start in offsets:
                        if sample_count >= num_samples:
                            break

                        sample_text, _ = build_sample_from_consecutive_sentences(
                            art_sents, start, min_tok, max_tok
                        )
                        if sample_text is None:
                            continue

                        sample_text = deep_clean_text(sample_text)
                        sample_text = MULTISPACE.sub(" ", sample_text).strip()
                        if not sample_text:
                            continue

                        if not re.search(r"[.!?।॥]$", sample_text):
                            sample_text = sample_text + "."

                        tok_count = count_tokens(sample_text)
                        if not (min_tok <= tok_count <= max_tok):
                            continue

                        if HAS_LATIN.search(sample_text) or HAS_DIGIT.search(sample_text):
                            continue

                        sample_count += 1
                        obj = {
                            "text": sample_text,
                            "xlm_roberta_tokens": tok_count,
                        }
                        outfile_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        outfile_handle.flush()

                        print(f"  [SAMPLE {sample_count}/{num_samples}] "
                              f"{tok_count} tokens [pass {pass_num + 2}]")

    finally:
        outfile_handle.close()

    # ── Final report ──
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Articles scraped:    {total_articles}")
    print(f"  Sentences extracted: {total_sentences}")
    print(f"  Samples written:     {sample_count}")
    print(f"  Output file:         {outfile.resolve()}")

    if sample_count > 0:
        tok_counts = []
        with outfile.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    tok_counts.append(obj["xlm_roberta_tokens"])
        if tok_counts:
            print(f"  Token stats:         min={min(tok_counts)}, max={max(tok_counts)}, "
                  f"avg={sum(tok_counts)/len(tok_counts):.1f}")


if __name__ == "__main__":
    main()