import re, time, unicodedata
import requests, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ═══════════════════ Config ═══════════════════
TARGET_SAMPLES   = 1000
TARGET_TOKENS    = 400      # soft threshold — closes at sentence boundary
MIN_TOKENS       = 60       # drop samples shorter than this
YEAR_CUTOFF      = 2022     # strict: only year < 2022
CHECKPOINT_EVERY = 100

# Gutenberg
MAX_GUTEN_PAGES  = 100
MAX_GUTEN_BOOKS  = 2000
GUTENDEX         = "https://gutendex.com/books"

# Internet Archive — Hindi texts pre-2022
IA_SEARCH        = "https://archive.org/advancedsearch.php"
IA_TEXT_BASE     = "https://archive.org/download"
MAX_IA_ITEMS     = 800      # items to fetch from search
IA_BATCH         = 100      # items per search page

# Wikisource
WS_API           = "https://hi.wikisource.org/w/api.php"   # ← Hindi Wikisource
MAX_WS_TITLES    = 2000

# Blogger/literary feeds — Hindi poems, kavita, creative writing blogs (pre-2022 only)
# These are well-known Hindi literary blogs focused on poems and creative writing
BLOGGER_FEEDS = [
    "https://kavitakosh.blogspot.com/feeds/posts/summary",
    "https://hindikavita.blogspot.com/feeds/posts/summary",
    "https://hindipoems.blogspot.com/feeds/posts/summary",
    "https://hindisahitya.blogspot.com/feeds/posts/summary",
    "https://kavitarang.blogspot.com/feeds/posts/summary",
    "https://hindikavitayen.blogspot.com/feeds/posts/summary",
    "https://aajkikavita.blogspot.com/feeds/posts/summary",
    "https://hindigeetgaane.blogspot.com/feeds/posts/summary",
    "https://hindishayari.blogspot.com/feeds/posts/summary",
    "https://hindinovelupdates.blogspot.com/feeds/posts/summary",
    "https://hindiliterature.blogspot.com/feeds/posts/summary",
    "https://kahanisansar.blogspot.com/feeds/posts/summary",
    "https://hindikahani.blogspot.com/feeds/posts/summary",
    "https://hindirachna.blogspot.com/feeds/posts/summary",
    "https://hindibhasha.blogspot.com/feeds/posts/summary",
]
BLOG_BATCH       = 100
BLOG_MAX_POSTS   = 5000     # per blog

# HTTP
TIMEOUT          = 45
RETRIES          = 3
BACKOFF          = 1.2
UA               = "HindiCorpusBuilder/1.0 (contact: your@email.com)"

OUT_CSV          = "hindi_samples.csv"
CKPT_CSV         = "hindi_checkpoint.csv"

def log(*a): print(*a, flush=True)

# ═══════════════════ HTTP ═══════════════════
def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    retry = Retry(
        total=RETRIES, backoff_factor=BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

S = make_session()

def http_json(url, **params):
    try:
        r = S.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"  [json-err] {url[:60]} — {e}")
        return None

def http_text(url):
    try:
        r = S.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log(f"  [http-err] {url[:55]} — {e}")
        return ""

def http_soup(url):
    h = http_text(url)
    return BeautifulSoup(h, "html.parser") if h else None

# ═══════════════════ Hindi utilities ═══════════════════
# Devanagari Unicode block: U+0900–U+097F
# We intentionally exclude Vedic Extensions (U+1CD0–U+1CFF) because those
# codepoints appear in OCR garbage but almost never in normal Hindi text.
HINDI_RANGE = r"\u0900-\u097F"
HINDI_TOK   = re.compile(fr"[{HINDI_RANGE}]+")
YEAR_RX     = re.compile(r"\b(1[6-9]\d\d|200\d|201\d|2020|2021)\b")

# ── Devanagari numerals (०–९) — strip these; they are the source of the
#    "numbers everywhere" problem seen in OCR output from scanned books.
DEVA_DIGITS = re.compile(r"[०-९]")

# ── A valid Hindi word must have at least one vowel carrier or matra.
#    Consonants: U+0915–U+0939, U+0958–U+095F, U+0978–U+097F
#    Vowels / matras: U+0904–U+0914, U+093A–U+094F, U+0960–U+0963
#    Halant (virama): U+094D  — needed for conjuncts
#    Anusvara/Visarga/Chandrabindu: U+0900–U+0903
MATRA_RX  = re.compile(r"[\u0904-\u0914\u093A-\u094F\u0960-\u0963]")
CONSONANT = re.compile(r"[\u0915-\u0939\u0958-\u095F\u0978-\u097F]")

def is_real_hindi_word(tok: str) -> bool:
    """
    Return True only if `tok` looks like a genuine Hindi word:
      • contains at least one consonant  AND
      • contains at least one matra/vowel  (rules out bare consonant clusters
        which are the fingerprint of OCR noise like 'ष्एश', 'ज़यपए')
      • is at least 2 characters long
    """
    if len(tok) < 2:
        return False
    return bool(CONSONANT.search(tok)) and bool(MATRA_RX.search(tok))

def is_hindi(s: str, min_ratio=0.55) -> bool:
    """Return True if >= min_ratio of whitespace-tokens contain Devanagari."""
    words = re.findall(r"\S+", s)
    if not words:
        return False
    return len(HINDI_TOK.findall(s)) / len(words) >= min_ratio

def has_latin(s: str) -> bool:
    """True if the string contains 3+ consecutive Latin characters."""
    return bool(re.search(r"[A-Za-z]{3,}", s))

# ── Quality gate thresholds ───────────────────────────────────────────────────
# What fraction of Devanagari tokens must pass the is_real_hindi_word test
# before we accept a sentence.  Set to 0.70 → 70 % of tokens must look real.
MIN_REAL_WORD_RATIO = 0.70

# Maximum allowed ratio of danda/double-danda characters to total characters.
# OCR garbage tends to be full of stray ॥ ॥ ॥ markers.
MAX_DANDA_RATIO     = 0.08   # e.g. 8 dandas in 100 chars → reject

# Maximum allowed ratio of Devanagari digit characters (०–९) to total chars.
# Real poetry rarely has more than ~3 % digits; garbled scans have 15-40 %.
MAX_DIGIT_RATIO     = 0.05

def quality_ok(s: str) -> bool:
    """
    Multi-signal quality gate.  Returns False if any of:
      1. Too many Devanagari numerals  (OCR scanned number columns)
      2. Too many dandas per character (symbol soup)
      3. Too few real Hindi words among Devanagari tokens
         (rules out 'ष्एश', 'ज़यपए', 'रिक्षग्रीब्राग' style OCR garbage)
    """
    if not s:
        return False
    n_chars = len(s)

    # Signal 1 — digit density
    digits = len(DEVA_DIGITS.findall(s))
    if digits / n_chars > MAX_DIGIT_RATIO:
        return False

    # Signal 2 — danda density
    dandas = s.count("।") + s.count("॥")
    if dandas / n_chars > MAX_DANDA_RATIO:
        return False

    # Signal 3 — real-word ratio among Devanagari tokens
    toks = HINDI_TOK.findall(s)
    if not toks:
        return False
    real = sum(1 for t in toks if is_real_hindi_word(t))
    if real / len(toks) < MIN_REAL_WORD_RATIO:
        return False

    return True

def clean_hindi(s: str) -> str:
    """
    Normalise to NFC, strip non-Devanagari characters, collapse whitespace.
    • Strips Devanagari numerals (०–९) — these are noise in OCR output.
    • Strips repeated dandas: ॥॥॥ → single ॥
    • Preserves single danda (।) and double-danda (॥) as sentence markers.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2026", " ").replace("...", " ")
    # Strip Latin letters and ASCII digits
    s = re.sub(r"[A-Za-z0-9]", " ", s)
    # Strip Devanagari digits (०–९)
    s = DEVA_DIGITS.sub(" ", s)
    # Keep only Devanagari block + dandas + whitespace
    s = re.sub(fr"[^{HINDI_RANGE}।॥\s]", " ", s)
    # Collapse repeated dandas: ॥ ॥ ॥ → ॥  and  ।।। → ।
    s = re.sub(r"(॥\s*){2,}", "॥ ", s)
    s = re.sub(r"(।\s*){2,}", "। ", s)
    return re.sub(r"\s+", " ", s).strip()

def hindi_tokens(s: str):
    """Return list of Devanagari word-tokens (whitespace-split after cleaning)."""
    return HINDI_TOK.findall(s)

def extract_year(s: str):
    m = YEAR_RX.search(s)
    return int(m.group()) if m else None

def year_ok(y) -> bool:
    if y is None:
        return True    # unknown year → assume pre-2022
    return y < YEAR_CUTOFF

# ═══════════════════ Sentence splitter ═══════════════════
def split_sentences(text: str):
    """
    Split raw text into cleaned Hindi sentences.
    Splits on:
      • Devanagari danda   (।  U+0964)
      • double-danda       (॥  U+0965)
      • ASCII full stop, ?, !
      • blank lines (paragraph breaks, common in poetry)
    Each candidate sentence is:
      1. Cleaned   (strip digits, extra dandas, Latin, non-Devanagari)
      2. Language-checked  (>= 55 % Devanagari tokens)
      3. Quality-gated     (real-word ratio, digit density, danda density)
    Only sentences passing all three checks are kept.
    """
    parts = re.split(r'(?<=[।॥\.?!])\s+|\n\s*\n', text)
    out = []
    for p in parts:
        p = p.strip()
        if not p or has_latin(p):
            continue
        c = clean_hindi(p)
        # Gate 1: must still be recognised as Hindi after cleaning
        if not c or not is_hindi(c):
            continue
        # Gate 2: multi-signal quality check (real words, digit density, dandas)
        if not quality_ok(c):
            continue
        out.append(c)
    return out

# ═══════════════════ Fine chunk splitter ═══════════════════
PARA_SPLIT = re.compile(r"\n\s*\n")

# Document-level quality check: sample the first 1000 characters of the raw
# document.  If the sample fails quality_ok we skip the entire document.
# This catches badly OCR-scanned PDFs before we waste time splitting them.
def doc_quality_ok(raw: str) -> bool:
    sample = clean_hindi(raw[:1500])
    if not sample:
        return False
    # Use slightly relaxed thresholds at document level since the sample
    # may include headers / metadata; we tighten at sentence level.
    toks = HINDI_TOK.findall(sample)
    if not toks:
        return False
    real = sum(1 for t in toks if is_real_hindi_word(t))
    real_ratio = real / len(toks)
    digit_ratio = len(DEVA_DIGITS.findall(sample)) / max(len(sample), 1)
    return real_ratio >= 0.55 and digit_ratio <= 0.10

def fine_chunks(text: str, chunk_chars: int = 3000):
    """
    Split text first by paragraphs (blank lines), then group into
    ~chunk_chars windows.  Poetry collections often have very short
    stanzas; grouping prevents tiny samples.
    Returns empty list if the document fails the quality pre-check.
    """
    if not doc_quality_ok(text):
        return []   # discard entire document early
    paras = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    if not paras:
        return [text]
    chunks, buf, buf_len = [], [], 0
    for p in paras:
        buf.append(p)
        buf_len += len(p)
        if buf_len >= chunk_chars:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [], 0
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

# ═══════════════════ Sentence-aware packer ═══════════════════
MAX_TOKENS = 500   # hard ceiling — no sample ever exceeds this

def pack(doc_iter, target=TARGET_SAMPLES):
    """
    Accumulate whole sentences into a buffer.
    Flush rules (in priority order):
      1. If adding the next sentence would push total > MAX_TOKENS (500),
         flush the current buffer FIRST, then start fresh with that sentence.
      2. If the buffer has reached TARGET_TOKENS (400) at a sentence boundary,
         flush immediately.
    Guarantees: MIN_TOKENS <= total_tokens <= MAX_TOKENS for every row.
    """
    rows      = []
    buf_sents = []
    buf_toks  = 0
    made      = 0

    def flush():
        nonlocal made
        if not buf_sents: return
        toks = hindi_tokens(" ".join(buf_sents))
        if len(toks) >= MIN_TOKENS:
            rows.append({"text": " ".join(toks), "total_tokens": len(toks)})
            made += 1
            if made % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(CKPT_CSV, index=False, encoding="utf-8")
                log(f"  [ckpt] {made} samples → {CKPT_CSV}")

    for doc in doc_iter:
        y = doc.get("year")
        if y is not None and y >= YEAR_CUTOFF:
            continue
        for sent in split_sentences(doc.get("text") or ""):
            toks = hindi_tokens(sent)
            if not toks: continue
            sent_len = len(toks)

            # Rule 1: adding this sentence would exceed the hard ceiling → flush first
            if buf_toks > 0 and buf_toks + sent_len > MAX_TOKENS:
                flush()
                buf_sents, buf_toks = [], 0
                if made >= target:
                    return pd.DataFrame(rows)

            # Oversized single sentence → emit as fixed-size token windows
            if sent_len > MAX_TOKENS:
                words = toks
                for start in range(0, len(words), MAX_TOKENS):
                    chunk = words[start:start + MAX_TOKENS]
                    if len(chunk) >= MIN_TOKENS:
                        rows.append({"text": " ".join(chunk),
                                     "total_tokens": len(chunk)})
                        made += 1
                        if made % CHECKPOINT_EVERY == 0:
                            pd.DataFrame(rows).to_csv(CKPT_CSV, index=False,
                                                       encoding="utf-8")
                            log(f"  [ckpt] {made} samples → {CKPT_CSV}")
                        if made >= target:
                            return pd.DataFrame(rows)
                continue  # don't add oversized sentence to buf

            buf_sents.append(sent)
            buf_toks += sent_len

            # Rule 2: soft target reached at a clean sentence boundary → flush
            if buf_toks >= TARGET_TOKENS:
                flush()
                buf_sents, buf_toks = [], 0
                if made >= target:
                    return pd.DataFrame(rows)

    if buf_sents:
        flush()
    return pd.DataFrame(rows)

# ═══════════════════ S1: Project Gutenberg ═══════════════════
def gutenberg_docs():
    """
    Fetch ALL Hindi books from Gutendex (lang=hi).
    Also fetches books tagged with Devanagari script that may be
    miscategorised under 'sa' (Sanskrit) but written in Hindi.
    Split each into fine paragraph chunks (~3000 chars each).

    Poetry / creative content filter:
      Prefer books whose title or subject contains keywords like
      कविता (poetry), काव्य (kavya), गीत (song/lyric), कहानी (story),
      उपन्यास (novel), नाटक (play), etc.
      Falls back to all Hindi books if the filtered set is too small.
    """
    POETRY_KEYWORDS = re.compile(
        r"कवित|काव्य|गीत|भजन|कहानी|उपन्यास|नाटक|रचना|पद्य|छंद|दोहा|"
        r"poem|lyric|song|novel|fiction|story|creative|literary",
        re.IGNORECASE
    )

    def fetch_books(lang):
        books, page = [], 1
        while page <= MAX_GUTEN_PAGES and len(books) < MAX_GUTEN_BOOKS:
            js = http_json(GUTENDEX, languages=lang, page=page)
            if not js:
                time.sleep(2); page += 1; continue
            for b in js.get("results", []):
                fmts  = b.get("formats", {})
                url   = (fmts.get("text/plain; charset=utf-8")
                         or fmts.get("text/plain")
                         or fmts.get("text/html; charset=utf-8")
                         or fmts.get("text/html"))
                if url:
                    title    = (b.get("title") or "").strip()
                    subjects = " ".join(b.get("subjects") or [])
                    books.append({
                        "title":    title,
                        "subjects": subjects,
                        "url":      url,
                    })
            if not js.get("next"): break
            page += 1
            time.sleep(0.25)
        return books

    # Primary: Hindi-tagged books
    books = fetch_books("hi")
    # Supplement: Sanskrit-tagged books often include classical Hindi poetry
    books += fetch_books("sa")

    # Prefer poetry/creative titles — filter first, fall back to all
    preferred = [b for b in books
                 if POETRY_KEYWORDS.search(b["title"] + " " + b["subjects"])]
    if len(preferred) >= 10:
        books_to_use = preferred
        log(f"[Gutenberg] {len(preferred)} poetry/creative books (filtered from "
            f"{len(books)} hi+sa books)")
    else:
        books_to_use = books
        log(f"[Gutenberg] {len(books)} hi+sa books (no filter — few poetry matches)")

    for b in books_to_use:
        raw = http_text(b["url"])
        if not raw: continue
        # Skip if the content is overwhelmingly non-Devanagari
        if not is_hindi(raw[:2000]):
            log(f"  skip (non-Hindi content): {b['title'][:50]}")
            continue
        chunks = fine_chunks(raw, chunk_chars=2500)
        log(f"  {b['title'][:55]} — {len(chunks)} chunks")
        for ch in chunks:
            yield {"text": ch, "year": None}  # classic = always pre-2022
        time.sleep(0.1)

# ═══════════════════ S2: Internet Archive Hindi texts ═══════════════════
def ia_search_ids():
    """
    Search archive.org for Hindi text items published before 2022.
    Narrows to poetry/creative content via subject keywords.
    Returns list of item identifiers.
    """
    # Run two queries: generic Hindi texts + specifically poetry/kavita
    queries = [
        'language:Hindi AND mediatype:texts',
        'language:Hindi AND mediatype:texts AND (subject:kavya OR subject:kavita '
        'OR subject:poetry OR subject:novel OR subject:kahani)',
    ]
    ids = []
    seen = set()
    for q in queries:
        page = 1
        while len(ids) < MAX_IA_ITEMS:
            params = {
                "q":      q,
                "fl[]":   "identifier,year,title",
                "sort[]": "downloads desc",
                "rows":   IA_BATCH,
                "page":   page,
                "output": "json",
            }
            js = http_json(IA_SEARCH, **params)
            if not js: break
            docs = js.get("response", {}).get("docs", [])
            if not docs: break
            for d in docs:
                iid = d["identifier"]
                if iid in seen: continue
                seen.add(iid)
                yr = extract_year(str(d.get("year", "")))
                if year_ok(yr):
                    ids.append(iid)
            page += 1
            time.sleep(0.3)
            if len(docs) < IA_BATCH: break

    log(f"[Archive.org] {len(ids)} Hindi text items")
    return ids[:MAX_IA_ITEMS]

def ia_text_url(identifier: str):
    """Find a plain-text file inside an Archive.org item."""
    meta_url = f"https://archive.org/metadata/{identifier}/files"
    js = http_json(meta_url)
    if not js: return None
    for f in js.get("result", []):
        name = f.get("name", "")
        if name.endswith(".txt") or name.endswith("_djvu.txt"):
            return f"{IA_TEXT_BASE}/{identifier}/{name}"
    return None

def ia_docs():
    ids = ia_search_ids()
    fetched = 0
    for identifier in ids:
        url = ia_text_url(identifier)
        if not url: continue
        raw = http_text(url)
        if not raw: continue
        if not is_hindi(raw[:2000]):
            continue
        yr = extract_year(identifier + raw[:500])
        for ch in fine_chunks(raw, chunk_chars=2500):
            yield {"text": ch, "year": yr}
        fetched += 1
        if fetched % 50 == 0:
            log(f"  [Archive.org] {fetched} items fetched")
        time.sleep(0.2)

# ═══════════════════ S3: Hindi Wikisource ═══════════════════
def ws_all_titles():
    """
    Scan hi.wikisource.org for all main-namespace pages.
    Also walk known poetry/literature categories.
    """
    titles = set()

    # Method A: allpages scan
    cont = None
    params = dict(action="query", list="allpages",
                  aplimit=500, apnamespace=0, format="json")
    while len(titles) < MAX_WS_TITLES:
        if cont: params["apcontinue"] = cont
        js = http_json(WS_API, **params)
        if not js: break
        for p in js.get("query", {}).get("allpages", []):
            titles.add(p["title"])
        cont = (js.get("continue") or {}).get("apcontinue")
        if not cont: break
        time.sleep(0.15)

    # Method B: Hindi literary/poetry category names on hi.wikisource.org
    cats_to_try = [
        # Hindi category names (श्रेणी = Category in Hindi)
        "श्रेणी:हिन्दी कविता",
        "श्रेणी:हिन्दी कहानियाँ",
        "श्रेणी:हिन्दी उपन्यास",
        "श्रेणी:हिन्दी नाटक",
        "श्रेणी:हिन्दी साहित्य",
        "श्रेणी:हिन्दी गीत",
        "श्रेणी:काव्य",
        "श्रेणी:भजन",
        # English category names also present on the site
        "Category:Hindi poetry",
        "Category:Hindi literature",
        "Category:Hindi novels",
        "Category:Hindi songs",
    ]
    for cat in cats_to_try:
        p2 = dict(action="query", list="categorymembers",
                  cmtitle=cat, cmlimit=500,
                  cmtype="page", format="json")
        js2 = http_json(WS_API, **p2)
        if not js2: continue
        for m in js2.get("query", {}).get("categorymembers", []):
            titles.add(m["title"])
        time.sleep(0.1)

    log(f"[Wikisource-hi] {len(titles)} unique pages")
    return list(titles)[:MAX_WS_TITLES]

def ws_extract(title: str):
    js = http_json(WS_API,
                   action="query", prop="extracts|revisions",
                   explaintext=True, rvprop="timestamp", rvlimit=1,
                   titles=title, format="json")
    if not js: return "", None
    for p in js.get("query", {}).get("pages", {}).values():
        txt  = p.get("extract", "") or ""
        revs = p.get("revisions", [])
        ts   = revs[0].get("timestamp", "") if revs else ""
        yr   = int(ts[:4]) if ts and len(ts) >= 4 else None
        return txt, yr
    return "", None

def wikisource_docs():
    titles = ws_all_titles()
    done = 0
    for t in titles:
        txt, yr = ws_extract(t)
        if txt and is_hindi(txt[:500]) and year_ok(yr):
            for ch in fine_chunks(txt, chunk_chars=2500):
                yield {"text": ch, "year": yr}
            done += 1
        time.sleep(0.1)
    log(f"[Wikisource-hi] yielded from {done} pages")

# ═══════════════════ S4: Multiple Hindi literary/poetry blogs ═══════════════════
def single_blog_docs(feed_url: str):
    """Yield docs from one Blogger feed, only posts published < 2022."""
    start = 1
    fetched = 0
    while fetched < BLOG_MAX_POSTS:
        js = http_json(feed_url, **{
            "alt":         "json",
            "start-index": start,
            "max-results": BLOG_BATCH
        })
        if not js: break
        entries = js.get("feed", {}).get("entry", []) or []
        if not entries: break

        for e in entries:
            pub = e.get("published", {}).get("$t", "")[:10]
            yr  = int(pub[:4]) if pub else None
            if yr is not None and yr >= YEAR_CUTOFF:
                continue
            link = next(
                (l["href"] for l in e.get("link", []) if l.get("rel") == "alternate"),
                ""
            )
            if not link: continue
            html = http_text(link)
            if not html: continue
            soup = BeautifulSoup(html, "html.parser")
            el   = (soup.select_one(".post-body")
                    or soup.select_one("article")
                    or soup.select_one("#post-body"))
            txt  = (el.get_text("\n", strip=True) if el
                    else soup.get_text("\n", strip=True))
            if txt and is_hindi(txt[:500]):
                yield {"text": txt, "year": yr}
            fetched += 1
            time.sleep(0.12)

        start += BLOG_BATCH
        time.sleep(0.2)

def all_blog_docs():
    for feed_url in BLOGGER_FEEDS:
        blog_name = feed_url.split("//")[1].split(".")[0]
        log(f"  [Blog] scraping: {blog_name}")
        count = 0
        for doc in single_blog_docs(feed_url):
            yield doc
            count += 1
        log(f"  [Blog] {blog_name} → {count} posts")

# ═══════════════════ Combined iterator ═══════════════════
def all_docs():
    log("── S1: Project Gutenberg (Hindi + Sanskrit, poetry-preferred) ──")
    yield from gutenberg_docs()
    log("── S2: Internet Archive (Hindi texts, pre-2022) ────────────────")
    yield from ia_docs()
    log("── S3: Hindi Wikisource (allpages + category scan) ─────────────")
    yield from wikisource_docs()
    log("── S4: Hindi literary/poetry Blogger feeds (15 blogs) ──────────")
    yield from all_blog_docs()

# ═══════════════════ Main ═══════════════════
if __name__ == "__main__":
    log(f"Building {TARGET_SAMPLES} Hindi poem/creative samples "
        f"(pre-{YEAR_CUTOFF}, ~{TARGET_TOKENS} tok each, sentence-aware)\n")

    final_df = pack(all_docs(), target=TARGET_SAMPLES)

    if final_df.empty:
        log("No samples collected.")
    else:
        log(f"\n{'='*55}")
        log(f"Total samples  : {len(final_df)}")
        log(f"Token counts   — "
            f"min={final_df['total_tokens'].min()}, "
            f"max={final_df['total_tokens'].max()}, "
            f"mean={final_df['total_tokens'].mean():.1f}, "
            f"median={final_df['total_tokens'].median():.0f}")
        final_df[["text", "total_tokens"]].to_csv(
            OUT_CSV, index=False, encoding="utf-8"
        )
        log(f"Saved → {OUT_CSV}")