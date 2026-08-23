import argparse
import json
import random
import re
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
DEFAULT_MIN_CHARS = 100
LOWER_DATE        = datetime(2012, 1, 1)
UPPER_DATE        = datetime(2022, 1, 1)
MAX_WORKERS       = 10

def _auto_outfile() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tamil_poems_{NUM_SAMPLES}_{MIN_TOK}-{MAX_TOK}_{ts}.jsonl"

OUTPUT_SAMPLES_FILE = _auto_outfile()
TOKENIZER = None

POEM_SITES = [
    {
        "name": "Kavithaikal",
        "sitemaps": [
            "https://www.kavithaikal.com/sitemap.xml",
            "https://www.kavithaikal.com/sitemap_index.xml",
        ],
        "archive_pages": [
            "https://www.kavithaikal.com/kavithai/",
            "https://www.kavithaikal.com/love-kavithai/",
            "https://www.kavithaikal.com/friendship-kavithai/",
            "https://www.kavithaikal.com/life-kavithai/",
            "https://www.kavithaikal.com/sad-kavithai/",
            "https://www.kavithaikal.com/motivational-kavithai/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poem", r"/kavithaikal"],
        "poem_selectors": [
            ".kavithai-content", ".poem-content", ".entry-content",
            "article", ".post-content", ".content",
        ],
    },
    {
        "name": "Tamilkavithai",
        "sitemaps": [
            "https://www.tamilkavithai.com/sitemap.xml",
            "https://www.tamilkavithai.com/sitemap_index.xml",
        ],
        "archive_pages": [
            "https://www.tamilkavithai.com/",
            "https://www.tamilkavithai.com/category/kavithai/",
            "https://www.tamilkavithai.com/category/love-kavithai/",
            "https://www.tamilkavithai.com/category/amma-kavithai/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poem"],
        "poem_selectors": [
            ".poem-text", ".kavithai-text", ".entry-content",
            "article", ".post-content",
        ],
    },
    {
        "name": "Vikatan Kavithai",
        "sitemaps": [
            "https://www.vikatan.com/sitemap.xml",
            "https://www.vikatan.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://www.vikatan.com/literature/poetry",
            "https://www.vikatan.com/literature/kavithai",
            "https://www.vikatan.com/entertainment/kavithai",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poetry", r"/literature"],
        "poem_selectors": [
            ".story-content", ".article-content", "article",
            ".content", ".field--body",
        ],
    },
    {
        "name": "Tamilliterature",
        "sitemaps": [
            "https://www.tamilliterature.com/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.tamilliterature.com/kavithai/",
            "https://www.tamilliterature.com/poem/",
            "https://www.tamilliterature.com/modern-poems/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poem"],
        "poem_selectors": [".poem", ".content", "article", "main"],
    },
    {
        "name": "Kavithai Tamil",
        "sitemaps": [
            "https://kavithaitamil.com/sitemap.xml",
            "https://kavithaitamil.com/sitemap_index.xml",
        ],
        "archive_pages": [
            "https://kavithaitamil.com/",
            "https://kavithaitamil.com/kavithai/",
            "https://kavithaitamil.com/love/",
            "https://kavithaitamil.com/life/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/love", r"/poem"],
        "poem_selectors": [".poem-content", ".entry-content", "article"],
    },
    {
        "name": "Kungumam",
        "sitemaps": [
            "https://www.kungumam.co.in/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.kungumam.co.in/kavithai/",
            "https://www.kungumam.co.in/literature/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/literature", r"/poem"],
        "poem_selectors": [".article-content", ".story-content", "article"],
    },
    {
        "name": "Nigazhvugal",
        "sitemaps": [
            "https://www.nigazhvugal.com/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.nigazhvugal.com/kavithai/",
            "https://www.nigazhvugal.com/poems/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poem"],
        "poem_selectors": [".content", "article", ".entry-content"],
    },
    {
        "name": "Tamilsoul",
        "sitemaps": [
            "https://www.tamilsoul.com/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.tamilsoul.com/kavithai/",
            "https://www.tamilsoul.com/poems/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poem", r"/soul"],
        "poem_selectors": [".poem", ".kavithai", ".content", "article"],
    },
    {
        "name": "Dinamalar Kavithai",
        "sitemaps": [
            "https://www.dinamalar.com/sitemap.xml",
            "https://www.dinamalar.com/news-sitemap.xml",
        ],
        "archive_pages": [
            "https://www.dinamalar.com/kavithai/",
            "https://www.dinamalar.com/literature/",
        ],
        "poem_url_patterns": [r"/kavithai", r"/literature", r"/poem"],
        "poem_selectors": [".article-content", ".content", "article"],
    },
    {
        "name": "Padasalai Kavithai",
        "sitemaps": [
            "https://www.padasalai.net/sitemap.xml",
        ],
        "archive_pages": [
            "https://www.padasalai.net/search/label/kavithai",
            "https://www.padasalai.net/search/label/கவிதை",
        ],
        "poem_url_patterns": [r"/kavithai", r"/poem", r"கவிதை"],
        "poem_selectors": [".post-body", ".entry-content", "article"],
    },
]

POEM_KEYWORDS = [
    "கவிதை", "கவிஞர்", "கவி", "பாடல்", "இலக்கியம்",
    "அன்பு", "காதல்", "இரவு", "நிலா", "மலர்",
    "கனவு", "வாழ்க்கை", "மழை", "தாய்", "நட்பு",
    "வானம்", "கடல்", "மண்", "பூமி", "இதயம்",
    "ஒளி", "இருள்", "நேசம்", "பாசம்", "ஆசை",
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
LINE_SPLIT         = re.compile(r"\n+")
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
        elif ch in (" ", "\t", "\n"):
            cleaned.append(ch)
        elif ch in (".", "!", "?", "\u0964", "\u0965", ","):
            cleaned.append(ch)
    text = "".join(cleaned)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"^\s*[.,!?]+\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def is_valid_poem_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 5:
        return False
    if len(TAMIL_CHAR.findall(line)) < 3:
        return False
    if HAS_LATIN.search(line) or HAS_DIGIT.search(line):
        return False
    if looks_like_cookie_notice(line):
        return False
    return True


def is_poem_content(text: str) -> bool:
    cleaned = deep_clean_text(text)
    if not cleaned or len(TAMIL_CHAR.findall(cleaned)) < 30:
        return False
    keyword_hits = sum(1 for kw in POEM_KEYWORDS if kw in cleaned)
    if keyword_hits < 1:
        return False
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    avg_line_len = sum(len(l) for l in lines) / len(lines)
    if avg_line_len > 200:
        return False
    return True


def extract_poem_lines(html: str, poem_selectors: list[str]) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all([
        "script", "style", "nav", "footer", "header", "aside",
        "form", "iframe", "noscript", "button", "input",
        "select", "textarea", "svg", "figure", "figcaption",
    ]):
        tag.decompose()

    raw_text = ""

    # Try poem-specific selectors first
    for css in poem_selectors:
        node = soup.select_one(css)
        if node:
            # Preserve newlines from <br> and <p> tags
            for br in node.find_all("br"):
                br.replace_with("\n")
            for p in node.find_all("p"):
                p.append("\n")
            raw_text = node.get_text("\n")
            if len(TAMIL_CHAR.findall(raw_text)) >= 20:
                break

    if not raw_text:
        for br in soup.find_all("br"):
            br.replace_with("\n")
        raw_text = soup.get_text("\n")

    cleaned = deep_clean_text(raw_text)
    if not cleaned:
        return []

    lines = []
    seen  = set()
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")   # preserve stanza breaks
            continue
        if line in seen:
            continue
        seen.add(line)
        if is_valid_poem_line(line):
            lines.append(line)

    # Collapse multiple blank lines into one stanza break
    collapsed = []
    prev_blank = False
    for l in lines:
        if l == "":
            if not prev_blank:
                collapsed.append("")
            prev_blank = True
        else:
            collapsed.append(l)
            prev_blank = False

    # Strip leading/trailing blanks
    while collapsed and collapsed[0] == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()

    return collapsed


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


def _is_poem_url(url: str, patterns: list[str]) -> bool:
    url_lower = url.lower()
    for pat in patterns:
        if re.search(pat, url_lower):
            return True
    return False


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
                ".article-meta", ".entry-date"]:
        n = soup.select_one(css)
        if n:
            txt = n.get("datetime") or n.get_text(" ", strip=True)
            try:
                return dateparser.parse(txt, dayfirst=True, fuzzy=True)
            except Exception:
                pass
    return None


def fetch_sitemap_urls(
    session: requests.Session,
    sitemap_url: str,
    poem_patterns: list[str],
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
            url_lower = child_url.lower()
            if any(k in url_lower for k in
                   ["kavithai", "poem", "literature", "kavitha", "entertain"]) or depth == 0:
                collected.extend(
                    fetch_sitemap_urls(session, child_url, poem_patterns, depth + 1)
                )
        return collected

    locs     = SITEMAP_LOC.findall(xml)
    lastmods = SITEMAP_LASTMOD.findall(xml)
    urls     = []
    for i, loc in enumerate(locs):
        loc = loc.strip()
        if not loc.startswith("http"):
            continue
        if not _is_poem_url(loc, poem_patterns):
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
    poem_patterns: list[str],
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
            if len(path) < 5 or path.count("/") < 1:
                continue
            if not _is_poem_url(href, poem_patterns):
                continue
            if not _url_date_hint(href):
                continue
            links.append(href)
        return list(set(links))
    except Exception as e:
        print(f"    [WARN] Archive page {page_url}: {e}")
        return []


def collect_urls_for_site(session: requests.Session, site: dict) -> list[str]:
    patterns = site.get("poem_url_patterns", [])
    urls: list[str] = []
    for sm_url in site.get("sitemaps", []):
        print(f"    [SITEMAP] {sm_url}")
        found = fetch_sitemap_urls(session, sm_url, patterns)
        print(f"    -> {len(found)} poem URLs")
        urls.extend(found)
        if urls:
            break
    if not urls:
        for arch_url in site.get("archive_pages", []):
            print(f"    [ARCHIVE] {arch_url}")
            found = extract_links_from_archive_page(session, arch_url, patterns)
            print(f"    -> {len(found)} poem URLs")
            urls.extend(found)
    return list(dict.fromkeys(urls))


def fetch_and_process_poem(
    session: requests.Session,
    url: str,
    poem_selectors: list[str],
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

    lines = extract_poem_lines(resp.text, poem_selectors)
    if not lines:
        return None

    poem_text = "\n".join(lines)
    if not is_poem_content(poem_text):
        return None
    if len(poem_text) < min_chars:
        return None

    return lines


def build_sample_from_poem_lines(
    lines: list[str],
    start_idx: int,
    min_tok: int,
    max_tok: int,
) -> tuple[str | None, int]:
    """
    Accumulates poem lines (preserving stanza breaks) until token
    window is filled. Never splits mid-stanza if possible.
    """
    current_lines: list[str] = []
    current_tok = 0
    idx = start_idx

    while idx < len(lines):
        line = lines[idx]
        idx += 1

        candidate_lines = current_lines + [line]
        candidate_text  = "\n".join(candidate_lines).strip()
        tok_count       = count_tokens(candidate_text)

        if tok_count > max_tok:
            if min_tok <= current_tok <= max_tok:
                return "\n".join(current_lines).strip(), idx
            # Skip this line and continue accumulating
            continue

        current_lines = candidate_lines
        current_tok   = tok_count

        if min_tok <= current_tok <= max_tok:
            return "\n".join(current_lines).strip(), idx

    final = "\n".join(current_lines).strip()
    if final and min_tok <= current_tok <= max_tok:
        return final, idx
    return None, idx


def samples_from_poem_lines(
    lines: list[str],
    min_tok: int,
    max_tok: int,
) -> list[str]:
    results = []
    start   = 0
    while start < len(lines):
        text, start = build_sample_from_poem_lines(lines, start, min_tok, max_tok)
        if text is None:
            break
        text = deep_clean_text(text).strip()
        if not text:
            continue
        tok = count_tokens(text)
        if not (min_tok <= tok <= max_tok):
            continue
        if HAS_LATIN.search(text) or HAS_DIGIT.search(text):
            continue
        if tamil_ratio(text) < 0.80:
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
    print(f"[CONFIG] Sources    : {len(POEM_SITES)} Tamil poetry sites")

    get_tokenizer()
    session = make_session()

    print("\n[PHASE 1] Collecting poem URLs from sitemaps & archive pages ...")
    all_urls_meta: list[tuple[str, list[str]]] = []

    for site in POEM_SITES:
        print(f"\n  [SITE] {site['name']}")
        urls = collect_urls_for_site(session, site)
        print(f"  -> {len(urls)} candidate poem URLs")
        selectors = site.get("poem_selectors", [".content", "article"])
        for url in urls:
            all_urls_meta.append((url, selectors))

    seen_urls = list(dict.fromkeys(u for u, _ in all_urls_meta))
    meta_map  = {u: s for u, s in all_urls_meta}
    random.shuffle(seen_urls)
    print(f"\n[PHASE 1] Total unique poem URLs: {len(seen_urls)}")

    print(f"\n[PHASE 2] Fetching poems with {args.workers} parallel threads ...")
    print("=" * 60)

    sample_count                       = 0
    total_poems                        = 0
    all_poem_lines: list[list[str]]    = []

    outfile_handle = outfile.open("w", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    fetch_and_process_poem, session, url,
                    meta_map[url], args.min_chars
                ): url
                for url in seen_urls
            }
            for future in as_completed(futures):
                if sample_count >= num_samples:
                    break
                lines = future.result()
                if not lines:
                    continue

                total_poems += 1
                all_poem_lines.append(lines)

                for text in samples_from_poem_lines(lines, min_tok, max_tok):
                    if sample_count >= num_samples:
                        break
                    sample_count += 1
                    tok = count_tokens(text)
                    outfile_handle.write(
                        json.dumps(
                            {"text": text, "xlm_roberta_tokens": tok},
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    outfile_handle.flush()
                    url_short = futures[future][:60]
                    print(f"  [SAMPLE {sample_count}/{num_samples}] "
                          f"{tok} tok | poem #{total_poems} | {url_short}")

        if sample_count < num_samples and all_poem_lines:
            print(f"\n[INFO] Got {sample_count}/{num_samples}. Running second pass ...")
            for pass_num in range(3):
                if sample_count >= num_samples:
                    break
                random.shuffle(all_poem_lines)
                for poem_lines in all_poem_lines:
                    if sample_count >= num_samples:
                        break
                    offsets = list(range(0, len(poem_lines), max(1, pass_num + 1)))
                    random.shuffle(offsets)
                    for start in offsets:
                        if sample_count >= num_samples:
                            break
                        text, _ = build_sample_from_poem_lines(
                            poem_lines, start, min_tok, max_tok
                        )
                        if text is None:
                            continue
                        text = deep_clean_text(text).strip()
                        if not text:
                            continue
                        tok = count_tokens(text)
                        if not (min_tok <= tok <= max_tok):
                            continue
                        if HAS_LATIN.search(text) or HAS_DIGIT.search(text):
                            continue
                        if tamil_ratio(text) < 0.80:
                            continue
                        sample_count += 1
                        outfile_handle.write(
                            json.dumps(
                                {"text": text, "xlm_roberta_tokens": tok},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        outfile_handle.flush()
                        print(f"  [SAMPLE {sample_count}/{num_samples}] "
                              f"{tok} tok [pass {pass_num + 2}]")

    finally:
        outfile_handle.close()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Poems processed     : {total_poems}")
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