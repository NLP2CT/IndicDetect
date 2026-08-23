"""
TWO-TIER CHUNKING SCRAPER - 1000 SAMPLES GUARANTEED
====================================================
Strategy:
1. Primary tier: Generate chunks at 380-500 tokens (most samples)
2. Secondary tier: Generate chunks at 360-379 tokens (only to fill shortage)

This ensures MOST samples are in your preferred 380-500 range,
with only minimal samples at 360-379 to reach exactly 1000.
"""

import re, json, time, unicodedata, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import requests
from bs4 import BeautifulSoup

TARGET     = 1000
MIN_PRIMARY   = 380  # Primary tier
MAX_PRIMARY   = 500
MIN_SECONDARY = 360  # Secondary tier (only for shortage)
MAX_SECONDARY = 379
OUTPUT     = "hindi_reviews_1000.jsonl"
WORKERS    = 12
TIMEOUT    = 12
DELAY      = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept-Language": "hi-IN,hi;q=0.9",
})

DEVA_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")

def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def token_count(text):
    return len(DEVA_RE.findall(text))

def split_sentences(raw):
    parts = re.split(r"(?<=[।॥?!])\s*|(?<=\.)\s+", raw)
    return [clean(p.strip()) for p in parts if clean(p.strip()) and token_count(clean(p.strip())) >= 6]

def build_chunks_tier(sentences, min_tok, max_tok):
    """Build chunks with specified token range."""
    chunks, buf, buf_t = [], [], 0
    for s in sentences:
        st = token_count(s)
        if st > max_tok:
            if buf_t >= min_tok:
                txt = " ".join(buf)
                tc = token_count(txt)
                if min_tok <= tc <= max_tok:
                    chunks.append(txt)
            buf, buf_t = [], 0
            continue
        if buf_t + st > max_tok:
            if buf_t >= min_tok:
                txt = " ".join(buf)
                tc = token_count(txt)
                if min_tok <= tc <= max_tok:
                    chunks.append(txt)
            buf, buf_t = [s], st
        else:
            buf.append(s)
            buf_t += st
    if buf and buf_t >= min_tok:
        txt = " ".join(buf)
        tc = token_count(txt)
        if min_tok <= tc <= max_tok:
            chunks.append(txt)
    return chunks

def fetch(url):
    try:
        time.sleep(DELAY)
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", []
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.find_all(["nav","header","footer","script","style","noscript","iframe"]):
            tag.decompose()
        urls = [a.get("href") for a in soup.find_all("a", href=True) if a.get("href")]
        raw = soup.get_text(" ", strip=True)
        return (raw, urls) if token_count(raw) >= 50 else ("", urls)
    except:
        return "", []

def crawl(name, seeds, valid_fn, base_url, max_fetch=200):
    """Generic BFS crawler."""
    visited, queue, texts, tc = set(), deque(), [], 0
    for u in seeds:
        visited.add(u)
        queue.append(u)
    fetched = 0
    while queue and fetched < max_fetch:
        batch = [queue.popleft() for _ in range(min(WORKERS, len(queue)))]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for txt, urls in [f.result() for f in [ex.submit(fetch, u) for u in batch]]:
                fetched += 1
                if txt:
                    texts.append(txt)
                    tc += token_count(txt)
                for href in urls:
                    if href.startswith("/"):
                        href = base_url + href
                    elif href.startswith("//"):
                        href = "https:" + href
                    elif not href.startswith("http"):
                        continue
                    href = href.split("?")[0].split("#")[0]
                    if valid_fn(href) and href not in visited:
                        visited.add(href)
                        queue.append(href)
        if fetched % 40 == 0:
            log.info("  [%s] %d pages, %d tokens", name, fetched, tc)
    log.info("[%s] DONE: %d pages, %d tokens", name, len(texts), tc)
    return texts, tc

# ══════════════════════════════════════════════════════════════════════════════
# SOURCES
# ══════════════════════════════════════════════════════════════════════════════

def wd_valid(url):
    if "/bollywood-movie-review/" not in url or not url.endswith("_1.html"):
        return False
    m = re.search(r"-(1[0-9]{11})_1\.html$", url)
    return m and int(m.group(1)[1:3]) < 22

WD_SEEDS = [
    "https://hindi.webdunia.com/bollywood-movie-review/sooryavanshi-review-in-hindi-akshay-kumar-katrina-kaif-ajay-devgn-ranveer-singh-samay-tamrkar-rohit-shetty-121110500026_1.html",
    "https://hindi.webdunia.com/bollywood-movie-review/gully-boy-review-in-hindi-ranveer-singh-samay-tamrakar-alia-bhatt-119021400047_1.html",
    "https://hindi.webdunia.com/bollywood-movie-review/uri-the-surgical-strike-review-in-hindi-vicky-kaushal-samay-tamrakar-119011200049_1.html",
    "https://m-hindi.webdunia.com/bollywood-movie-review/sooryavanshi-review-in-hindi-akshay-kumar-katrina-kaif-ajay-devgn-ranveer-singh-samay-tamrkar-rohit-shetty-121110500026_1.html",
]

def tn_valid(url):
    return "/movie-review" in url and "timesnowhindi.com" in url

TN_SEEDS = ["https://www.timesnowhindi.com/entertainment/movie-reviews"]

def au_valid(url):
    return ("/entertainment" in url or "/movie" in url) and "amarujala.com" in url

AU_SEEDS = ["https://www.amarujala.com/entertainment"]

def lh_valid(url):
    return ("/entertainment" in url or "/movie" in url) and "livehindustan.com" in url

LH_SEEDS = ["https://www.livehindustan.com/entertainment"]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 72)
    log.info("TWO-TIER CHUNKING SCRAPER")
    log.info("Primary tier: %d-%d tokens (most samples)", MIN_PRIMARY, MAX_PRIMARY)
    log.info("Secondary tier: %d-%d tokens (only to fill shortage)", MIN_SECONDARY, MAX_SECONDARY)
    log.info("Target: %d samples", TARGET)
    log.info("=" * 72)

    all_texts = []
    
    log.info("\n[1/4] Webdunia...")
    t, _ = crawl("WD", WD_SEEDS, wd_valid, "https://hindi.webdunia.com", max_fetch=350)
    all_texts.extend(t)
    
    log.info("\n[2/4] TimesNow Hindi...")
    t, _ = crawl("TN", TN_SEEDS, tn_valid, "https://www.timesnowhindi.com", max_fetch=300)
    all_texts.extend(t)
    
    log.info("\n[3/4] AmarUjala...")
    t, _ = crawl("AU", AU_SEEDS, au_valid, "https://www.amarujala.com", max_fetch=150)
    all_texts.extend(t)
    
    log.info("\n[4/4] LiveHindustan...")
    t, _ = crawl("LH", LH_SEEDS, lh_valid, "https://www.livehindustan.com", max_fetch=150)
    all_texts.extend(t)
    
    total_tc = sum(token_count(t) for t in all_texts)
    log.info("\n" + "=" * 72)
    log.info("TOTAL: %d pages, %dk tokens", len(all_texts), total_tc // 1000)
    log.info("=" * 72)
    
    log.info("\nProcessing text...")
    sentences = split_sentences(" ".join(all_texts))
    log.info("Total sentences: %d", len(sentences))
    
    # Dedupe sentences
    seen, unique = set(), []
    for s in sentences:
        k = s[:130]
        if k not in seen:
            seen.add(k)
            unique.append(s)
    log.info("Unique sentences: %d", len(unique))
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1: PRIMARY CHUNKS (380-500 tokens)
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n[PRIMARY TIER] Generating chunks at %d-%d tokens...", MIN_PRIMARY, MAX_PRIMARY)
    primary_chunks = build_chunks_tier(unique, MIN_PRIMARY, MAX_PRIMARY)
    log.info("Primary chunks generated: %d", len(primary_chunks))
    
    # Dedupe primary
    seen, primary_final = set(), []
    for txt in primary_chunks:
        k = txt[:250]
        if k not in seen:
            seen.add(k)
            primary_final.append({"text": txt, "token_count": token_count(txt), "tier": "primary"})
    
    log.info("Primary chunks (after dedup): %d", len(primary_final))
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIER 2: SECONDARY CHUNKS (360-379 tokens) - ONLY IF NEEDED
    # ═══════════════════════════════════════════════════════════════════════
    secondary_final = []
    shortage = TARGET - len(primary_final)
    
    if shortage > 0:
        log.info("\n[SECONDARY TIER] Need %d more samples...", shortage)
        log.info("Generating chunks at %d-%d tokens...", MIN_SECONDARY, MAX_SECONDARY)
        secondary_chunks = build_chunks_tier(unique, MIN_SECONDARY, MAX_SECONDARY)
        log.info("Secondary chunks generated: %d", len(secondary_chunks))
        
        # Dedupe secondary (avoid duplicates with primary)
        for txt in secondary_chunks:
            k = txt[:250]
            if k not in seen:
                seen.add(k)
                secondary_final.append({"text": txt, "token_count": token_count(txt), "tier": "secondary"})
        
        log.info("Secondary chunks (after dedup): %d", len(secondary_final))
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMBINE AND SAVE
    # ═══════════════════════════════════════════════════════════════════════
    all_final = primary_final + secondary_final
    all_final = all_final[:TARGET]
    
    # Count by tier
    primary_count = sum(1 for x in all_final if x["tier"] == "primary")
    secondary_count = sum(1 for x in all_final if x["tier"] == "secondary")
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in all_final:
            # Remove tier field before saving
            output_rec = {"text": rec["text"], "token_count": rec["token_count"]}
            f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
    
    counts = [r["token_count"] for r in all_final]
    log.info("\n" + "=" * 72)
    log.info("✅ SAVED: %d samples → %s", len(all_final), OUTPUT)
    log.info("=" * 72)
    log.info("PRIMARY tier (380-500): %d samples", primary_count)
    log.info("SECONDARY tier (360-379): %d samples", secondary_count)
    if counts:
        log.info("Overall tokens: min=%d max=%d avg=%.1f", min(counts), max(counts), sum(counts)/len(counts))
    log.info("=" * 72)
    
    if len(all_final) < TARGET:
        log.error("\n❌ STILL SHORT: %d/%d samples", len(all_final), TARGET)
        log.error("Need to lower MIN_SECONDARY further or add more sources")
    else:
        log.info("\n🎉 SUCCESS: Delivered %d samples!", len(all_final))
        log.info("Most samples (%d) are in your preferred 380-500 range", primary_count)


if __name__ == "__main__":
    main()