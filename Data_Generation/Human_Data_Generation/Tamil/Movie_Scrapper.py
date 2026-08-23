import argparse
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
from dateutil import parser as dateparser
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

NUM_SAMPLES       = 1000
MIN_TOK           = 400
MAX_TOK           = 500
RANDOM_SEED       = 42
DEFAULT_MIN_CHARS = 140
LOWER_DATE        = datetime(2012, 1, 1)
UPPER_DATE        = datetime(2022, 1, 1)
MAX_WORKERS       = 10

def _auto_outfile() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tamil_movie_reviews_{NUM_SAMPLES}_{MIN_TOK}-{MAX_TOK}_{ts}.jsonl"

OUTPUT_SAMPLES_FILE = _auto_outfile()
TOKENIZER = None

REVIEW_SITES = [
    {
        "name": "Vikatan Cinema",
        "sitemaps": [
            "https://www.vikatan.com/sitemap.xml",
            "https://www.vikatan.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://www.vikatan.com/entertainment/cinema/movie-review",
        ],
        "review_url_patterns": [
            r"/movie-review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Dinamalar Cinema",
        "sitemaps": [
            "https://www.dinamalar.com/sitemap.xml",
            "https://www.dinamalar.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://www.dinamalar.com/cinema/review",
        ],
        "review_url_patterns": [
            r"/cinema/review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Maalaimalar Cinema",
        "sitemaps": [
            "https://www.maalaimalar.com/sitemap.xml",
            "https://www.maalaimalar.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://www.maalaimalar.com/cinema/review",
        ],
        "review_url_patterns": [
            r"/cinema/review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Galatta",
        "sitemaps": [
            "https://www.galatta.com/sitemap.xml",
            "https://www.galatta.com/sitemap_index.xml",
            "https://www.galatta.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://www.galatta.com/tamil-cinema/movie-review/",
            "https://www.galatta.com/category/movie-review/",
        ],
        "review_url_patterns": [
            r"/movie-review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Indiaglitz Tamil",
        "sitemaps": [
            "https://www.indiaglitz.com/sitemap.xml",
            "https://www.indiaglitz.com/sitemap-index.xml",
        ],
        "archive_pages": [
            "https://www.indiaglitz.com/tamil/movie-review",
            "https://www.indiaglitz.com/channels/tamil/review.asp",
        ],
        "review_url_patterns": [
            r"/movie-review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Behindwoods",
        "sitemaps": [
            "https://behindwoods.com/sitemap.xml",
            "https://behindwoods.com/sitemap_index.xml",
        ],
        "archive_pages": [
            "https://behindwoods.com/tamil-movies/tamil-movie-reviews/",
            "https://behindwoods.com/reviews/",
        ],
        "review_url_patterns": [
            r"/tamil-movie-review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Cinema Express Tamil",
        "sitemaps": [
            "https://www.cinemaexpress.com/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.cinemaexpress.com/reviews/",
        ],
        "review_url_patterns": [
            r"/reviews/",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Filmibeat Tamil",
        "sitemaps": [
            "https://tamil.filmibeat.com/sitemap.xml",
            "https://tamil.filmibeat.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://tamil.filmibeat.com/movies/reviews/",
        ],
        "review_url_patterns": [
            r"/movies/reviews",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Cinesouth",
        "sitemaps": [
            "https://www.cinesouth.com/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.cinesouth.com/tamil-movie-reviews/",
        ],
        "review_url_patterns": [
            r"/tamil-movie-review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
    {
        "name": "Sify Movies Tamil",
        "sitemaps": [],
        "archive_pages": [
            "https://www.sify.com/movies/tamil-reviews/",
            "https://www.sify.com/movies/reviewsmore.php?lang=ta",
        ],
        "review_url_patterns": [
            r"/tamil-review",
            r"movie.*review",
            r"review.*movie",
        ],
    },
]

REVIEW_KEYWORDS = [
    "விமர்சனம்", "திரை விமர்சனம்", "படம் விமர்சனம்",
    "நடிப்பு", "கதை", "திரைக்கதை", "ஒளிப்பதிவு",
    "இசை", "திரைப்படம்", "வசனம்", "காட்சி",
    "பாத்திரம்", "இயக்கம்", "இயக்குனர்",
    "வெற்றிப்படம்", "தோல்விப்படம்",
    "படத்தின்", "படம் பார்க்க", "படம் தரம்",
]

# Any of these appearing heavily → interview, not a review → reject
INTERVIEW_KEYWORDS = [
    "நேர்காணல்", "பேட்டி", "பேசினார்", "கூறினார்", "தெரிவித்தார்",
    "சொன்னார்", "நடிகர் கூற", "நடிகை கூற", "இயக்குனர் கூற",
    "interview", "exclusive", "கேள்வி", "பதில்",
    "என்று சொன்னார்", "என்று கூறினார்", "என்று தெரிவித்தார்",
]

BLOCKED_HOSTS = [
    "consent.google.com", "translate.google", "facebook.", "twitter.",
    "youtube.", "instagram.", "tiktok.", "linkedin.", "reddit.",
    "accounts.google.com",
]

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
    "cookie", "cookies", "privacy", "consent", "gdpr", "cpra", "ccpa",
    "do not sell", "personal information", "advert", "personalized ads",
    "analytics", "performance cookies", "strictly necessary",
    "third parties", "exercise my rights",
]


def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        print("[INFO] Loading xlm-roberta-base tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")
        print("[INFO] Tokenizer loaded.")
    return TOKENIZER


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


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
        elif ch in (".", "!", "?", "\u0964", "\u0965", ","):
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
    if not p or looks_like_cookie_notice(p):
        return False
    return tamil_ratio(p) >= 0.70 or len(TAMIL_CHAR.findall(p)) >= 16


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


def is_review_content(text: str) -> bool:
    tamil_text = deep_clean_text(text)
    if not tamil_text or len(TAMIL_CHAR.findall(tamil_text)) < 50:
        return False

    # Must have at least 3 distinct review vocabulary hits
    review_hits = sum(1 for kw in REVIEW_KEYWORDS if kw in tamil_text)
    if review_hits < 3:
        return False

    # Reject if interview/quote markers dominate the text
    # Count how many interview phrases appear
    interview_hits = sum(1 for kw in INTERVIEW_KEYWORDS if kw in tamil_text)
    if interview_hits >= 3:
        return False

    # Reject if "விமர்சனம்" (review) is absent but interview words are present
    # Pure interviews never use the word review
    has_review_word = "விமர்சனம்" in tamil_text or "விமர்சன" in tamil_text
    if not has_review_word and interview_hits >= 1:
        return False

    return True


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


def parse_date_from_html(html: str):
    soup = BeautifulSoup(html, "lxml")
    for css, attr in [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="pubdate"]',                    "content"),
        ('meta[name="publish-date"]',               "content"),
        ('meta[name="date"]',                       "content"),
        ('meta[itemprop="datePublished"]',          "content"),
        ('meta[property="og:updated_time"]',        "content"),
    ]:
        m = soup.select_one(css)
        if m and m.get(attr):
            try:
                return dateparser.parse(m.get(attr), dayfirst=True, fuzzy=True)
            except Exception:
                pass
    for css in [".post-date", ".published-date", ".date", "time",
                ".article-meta", ".story-date", ".review-date"]:
        n = soup.select_one(css)
        if n:
            txt = n.get("datetime") or n.get_text(" ", strip=True)
            try:
                return dateparser.parse(txt, dayfirst=True, fuzzy=True)
            except Exception:
                pass
    return None


def extract_sentences_from_html(html: str, min_chars: int) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all([
        "script", "style", "nav", "footer", "header", "aside",
        "form", "iframe", "noscript", "button", "input",
        "select", "textarea", "svg", "figure", "figcaption",
    ]):
        tag.decompose()
    paras = []
    for css in [
        ".review-content p", ".movie-review p", ".review-body p",
        "article p", ".article-content p", ".node__content p",
        ".story-content p", ".field--body p", ".content p", "main p", "p",
    ]:
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
    full_text = MULTISPACE.sub(" ", " ".join(paras).replace("\n", " ")).strip()
    if len(full_text) < min_chars:
        return []
    if not is_review_content(full_text):
        return []
    return extract_clean_sentences(full_text)


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
        return 2012 <= int(m.group(1)) <= 2021
    return True


def _is_review_url(url: str, patterns: list[str]) -> bool:
    url_lower = url.lower()
    for pat in patterns:
        if re.search(pat, url_lower):
            return True
    return False


def fetch_sitemap_urls(
    session: requests.Session,
    sitemap_url: str,
    review_patterns: list[str],
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
        child_urls = SITEMAP_LOC.findall(xml)
        collected  = []
        for child_url in child_urls:
            child_url = child_url.strip()
            year_m = re.search(r"20(\d{2})", child_url)
            if year_m and not (2012 <= 2000 + int(year_m.group(1)) <= 2021):
                continue
            if "cinema" in child_url.lower() or "entertainment" in child_url.lower() \
                    or "movie" in child_url.lower() or "review" in child_url.lower() \
                    or depth == 0:
                collected.extend(
                    fetch_sitemap_urls(session, child_url, review_patterns, depth + 1)
                )
        return collected

    locs     = SITEMAP_LOC.findall(xml)
    lastmods = SITEMAP_LASTMOD.findall(xml)
    urls     = []
    for i, loc in enumerate(locs):
        loc = loc.strip()
        if not loc.startswith("http"):
            continue
        if not _is_review_url(loc, review_patterns):
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
    review_patterns: list[str],
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
            if not href.startswith("http"):
                continue
            if urlparse(href).netloc != base_domain:
                continue
            path = urlparse(href).path
            if len(path) < 10 or path.count("/") < 2:
                continue
            if not _is_review_url(href, review_patterns):
                continue
            if not _url_date_hint(href):
                continue
            links.append(href)
        return list(set(links))
    except Exception as e:
        print(f"    [WARN] Archive page {page_url}: {e}")
        return []


def collect_urls_for_site(session: requests.Session, site: dict) -> list[str]:
    patterns = site.get("review_url_patterns", [])
    urls: list[str] = []
    for sm_url in site.get("sitemaps", []):
        print(f"    [SITEMAP] {sm_url}")
        found = fetch_sitemap_urls(session, sm_url, patterns)
        print(f"    -> {len(found)} review URLs")
        urls.extend(found)
        if urls:
            break
    if not urls:
        for arch_url in site.get("archive_pages", []):
            print(f"    [ARCHIVE] {arch_url}")
            found = extract_links_from_archive_page(session, arch_url, patterns)
            print(f"    -> {len(found)} review URLs")
            urls.extend(found)
    return list(dict.fromkeys(urls))


def fetch_and_process_article(
    session: requests.Session,
    url: str,
    min_chars: int,
) -> list[str] | None:
    host = urlparse(url).netloc.lower()
    if any(b in host for b in BLOCKED_HOSTS):
        return None
    try:
        resp = session.get(url, timeout=12, allow_redirects=True)
    except Exception:
        return None
    if resp.status_code >= 400 or not resp.text:
        return None
    try:
        pub_dt = parse_date_from_html(resp.text)
    except Exception:
        pub_dt = None
    if pub_dt is not None:
        if pub_dt.tzinfo:
            pub_dt = pub_dt.astimezone(timezone.utc).replace(tzinfo=None)
        if pub_dt >= UPPER_DATE:
            return None
    sentences = extract_sentences_from_html(resp.text, min_chars=min_chars)
    return sentences if len(sentences) >= 3 else None


def build_sample_from_consecutive_sentences(
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
        text, start = build_sample_from_consecutive_sentences(sentences, start, min_tok, max_tok)
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
        results.append(text)
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile",     type=str, default=OUTPUT_SAMPLES_FILE)
    ap.add_argument("--min-chars",   type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--min-tok",     type=int, default=MIN_TOK)
    ap.add_argument("--max-tok",     type=int, default=MAX_TOK)
    ap.add_argument("--workers",     type=int, default=MAX_WORKERS)
    return ap.parse_args()


def main():
    args        = parse_args()
    random.seed(RANDOM_SEED)
    outfile     = Path(args.outfile)
    num_samples = args.num_samples
    min_tok     = args.min_tok
    max_tok     = args.max_tok

    print(f"[CONFIG] Output     : {outfile}")
    print(f"[CONFIG] Target     : {num_samples} samples, {min_tok}-{max_tok} tokens each")
    print(f"[CONFIG] Date range : 2012-01-01 -> 2021-12-31")
    print(f"[CONFIG] Workers    : {args.workers} parallel threads")
    print(f"[CONFIG] Sources    : {len(REVIEW_SITES)} Tamil movie review sites")

    get_tokenizer()
    session = make_session()

    print("\n[PHASE 1] Collecting review URLs from sitemaps & archive pages ...")
    all_urls: list[str] = []
    for site in REVIEW_SITES:
        print(f"\n  [SITE] {site['name']}")
        urls = collect_urls_for_site(session, site)
        print(f"  -> {len(urls)} candidate review URLs")
        all_urls.extend(urls)

    all_urls = list(dict.fromkeys(all_urls))
    random.shuffle(all_urls)
    print(f"\n[PHASE 1] Total unique review URLs: {len(all_urls)}")

    print(f"\n[PHASE 2] Fetching reviews with {args.workers} parallel threads ...")
    print("=" * 60)

    sample_count                       = 0
    total_articles                     = 0
    all_article_sentences: list[list[str]] = []

    outfile_handle = outfile.open("w", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_and_process_article, session, url, args.min_chars): url
                for url in all_urls
            }
            for future in as_completed(futures):
                if sample_count >= num_samples:
                    break
                sentences = future.result()
                if not sentences:
                    continue

                total_articles += 1
                all_article_sentences.append(sentences)

                for text in samples_from_sentences(sentences, min_tok, max_tok):
                    if sample_count >= num_samples:
                        break
                    sample_count += 1
                    tok = count_tokens(text)
                    outfile_handle.write(
                        json.dumps({"text": text, "xlm_roberta_tokens": tok},
                                   ensure_ascii=False) + "\n"
                    )
                    outfile_handle.flush()
                    print(f"  [SAMPLE {sample_count}/{num_samples}] "
                          f"{tok} tok | review #{total_articles} | {futures[future][:60]}")

        if sample_count < num_samples and all_article_sentences:
            print(f"\n[INFO] Got {sample_count}/{num_samples}. Running second pass ...")
            for pass_num in range(3):
                if sample_count >= num_samples:
                    break
                random.shuffle(all_article_sentences)
                for art_sents in all_article_sentences:
                    if sample_count >= num_samples:
                        break
                    offsets = list(range(0, len(art_sents), max(1, pass_num + 1)))
                    random.shuffle(offsets)
                    for start in offsets:
                        if sample_count >= num_samples:
                            break
                        text, _ = build_sample_from_consecutive_sentences(
                            art_sents, start, min_tok, max_tok
                        )
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
                        sample_count += 1
                        outfile_handle.write(
                            json.dumps({"text": text, "xlm_roberta_tokens": tok},
                                       ensure_ascii=False) + "\n"
                        )
                        outfile_handle.flush()
                        print(f"  [SAMPLE {sample_count}/{num_samples}] "
                              f"{tok} tok [pass {pass_num + 2}]")

    finally:
        outfile_handle.close()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Reviews processed   : {total_articles}")
    print(f"  Samples written     : {sample_count}")
    print(f"  Output file         : {outfile.resolve()}")

    if sample_count > 0:
        tok_counts = [
            json.loads(l)["xlm_roberta_tokens"]
            for l in outfile.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        if tok_counts:
            print(f"  Token stats         : min={min(tok_counts)}, "
                  f"max={max(tok_counts)}, avg={sum(tok_counts)/len(tok_counts):.1f}")


if __name__ == "__main__":
    main()