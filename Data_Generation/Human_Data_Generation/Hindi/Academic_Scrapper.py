"""
Hindi Academic-Style Content Scraper - WORKING VERSION
=======================================================
Since research journals block scraping, this scrapes:
1. Hindi Wikipedia (academic articles - Science, History, etc.)
2. PIB (Press Information Bureau) - Government policy documents
3. NITI Aayog - Policy reports and analysis in Hindi
4. NCERT - Educational textbooks and materials
5. IGNOU - Academic course materials

These sources have FORMAL, ACADEMIC WRITING STYLE similar to research papers.
Uses XLM-RoBERTa tokenization | Target: 1000 samples × 400-500 tokens
"""

import re, json, time, unicodedata, logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import requests
from bs4 import BeautifulSoup

try:
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    USE_XLMR = True
except:
    print("⚠️  Install: pip install transformers --break-system-packages")
    USE_XLMR = False

TARGET = 1000
MIN_PRIMARY = 400
MAX_PRIMARY = 500
MIN_SECONDARY = 360
MAX_SECONDARY = 399
OUTPUT = "hindi_academic_1000.jsonl"
WORKERS = 10
TIMEOUT = 12
DELAY = 0.15

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "hi,en;q=0.9",
})

DEVA_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")

def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def token_count(text):
    if USE_XLMR:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(DEVA_RE.findall(text))

def split_sentences(raw):
    parts = re.split(r"(?<=[।॥?!])\s*|(?<=\.)\s+", raw)
    return [clean(p.strip()) for p in parts if clean(p.strip()) and token_count(clean(p.strip())) >= 10]

def build_chunks_tier(sentences, min_tok, max_tok):
    chunks, buf, buf_t = [], [], 0
    for s in sentences:
        st = token_count(s)
        if st > max_tok:
            if buf_t >= min_tok:
                txt = " ".join(buf)
                if min_tok <= token_count(txt) <= max_tok:
                    chunks.append(txt)
            buf, buf_t = [], 0
            continue
        if buf_t + st > max_tok:
            if buf_t >= min_tok:
                txt = " ".join(buf)
                if min_tok <= token_count(txt) <= max_tok:
                    chunks.append(txt)
            buf, buf_t = [s], st
        else:
            buf.append(s)
            buf_t += st
    if buf and buf_t >= min_tok:
        txt = " ".join(buf)
        if min_tok <= token_count(txt) <= max_tok:
            chunks.append(txt)
    return chunks

def fetch(url):
    try:
        time.sleep(DELAY)
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", []
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Wikipedia-specific extraction
        if "wikipedia.org" in url:
            content = soup.find("div", {"id": "mw-content-text"}) or soup.find("div", {"class": "mw-parser-output"})
            if content:
                for tag in content.find_all(["table", "div"], {"class": ["infobox", "navbox", "reflist", "toc", "metadata"]}):
                    tag.decompose()
                text_elem = content
            else:
                text_elem = soup
        else:
            # Generic
            for tag in soup.find_all(["nav","header","footer","script","style","noscript","iframe","form"]):
                tag.decompose()
            text_elem = soup
        
        urls = [a.get("href") for a in soup.find_all("a", href=True)]
        raw = text_elem.get_text(" ", strip=True)
        
        # Check Hindi ratio
        hindi_words = len(DEVA_RE.findall(raw))
        total_words = len(raw.split())
        if total_words == 0 or hindi_words / total_words < 0.35:
            return "", urls
        
        return (raw, urls) if token_count(raw) >= 100 else ("", urls)
    except:
        return "", []

def normalize_url(href, base):
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = base + href
    elif not href.startswith("http"):
        return None
    return href.split("?")[0].split("#")[0]

def crawl(name, seeds, valid_fn, base_url, max_fetch=250):
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
                    nu = normalize_url(href, base_url)
                    if nu and valid_fn(nu) and nu not in visited:
                        visited.add(nu)
                        queue.append(nu)
        if fetched % 50 == 0:
            log.info("  [%s] %d pages, %dk tokens", name, fetched, tc // 1000)
    
    log.info("[%s] DONE: %d pages, %dk tokens", name, len(texts), tc // 1000)
    return texts, tc

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: HINDI WIKIPEDIA (Academic/Scientific Articles)
# ══════════════════════════════════════════════════════════════════════════════

def wiki_valid(url):
    if "hi.wikipedia.org/wiki/" not in url:
        return False
    exclude = ["विशेष:", "सहायता:", "विकिपीडिया:", "साँचा:", "चित्र:", "फ़ाइल:", "Special:", "Help:", "Category:"]
    return not any(x in url for x in exclude)

WIKI_SEEDS = [
    # Natural Sciences
    "https://hi.wikipedia.org/wiki/भौतिकी",
    "https://hi.wikipedia.org/wiki/रसायन_विज्ञान",
    "https://hi.wikipedia.org/wiki/जीव_विज्ञान",
    "https://hi.wikipedia.org/wiki/खगोल_विज्ञान",
    "https://hi.wikipedia.org/wiki/भूविज्ञान",
    "https://hi.wikipedia.org/wiki/गणित",
    # Social Sciences  
    "https://hi.wikipedia.org/wiki/अर्थशास्त्र",
    "https://hi.wikipedia.org/wiki/समाजशास्त्र",
    "https://hi.wikipedia.org/wiki/मनोविज्ञान",
    "https://hi.wikipedia.org/wiki/राजनीति_विज्ञान",
    "https://hi.wikipedia.org/wiki/भूगोल",
    # Humanities
    "https://hi.wikipedia.org/wiki/दर्शनशास्त्र",
    "https://hi.wikipedia.org/wiki/इतिहास",
    "https://hi.wikipedia.org/wiki/पुरातत्व",
    "https://hi.wikipedia.org/wiki/भाषाविज्ञान",
]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: PIB HINDI (Government Policy Documents)
# ══════════════════════════════════════════════════════════════════════════════

def pib_valid(url):
    return "pib.gov.in" in url and "/hindi/" in url.lower()

PIB_SEEDS = [
    "https://pib.gov.in/indexd.aspx",
]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: BBC HINDI (Analysis Articles)
# ══════════════════════════════════════════════════════════════════════════════

def bbc_valid(url):
    return "bbc.com/hindi" in url and ("/articles/" in url or "/india/" in url)

BBC_SEEDS = [
    "https://www.bbc.com/hindi",
    "https://www.bbc.com/hindi/india",
]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not USE_XLMR:
        log.warning("⚠️  Using word count fallback")
    
    log.info("=" * 72)
    log.info("Hindi Academic-Style Content Scraper")
    log.info("Sources: Wikipedia (science/academic) + PIB (policy) + BBC (analysis)")
    log.info("Primary: %d-%d | Secondary: %d-%d tokens", 
             MIN_PRIMARY, MAX_PRIMARY, MIN_SECONDARY, MAX_SECONDARY)
    log.info("Target: %d samples", TARGET)
    log.info("=" * 72)

    all_texts = []
    
    log.info("\n[1/3] Hindi Wikipedia (academic articles)...")
    t, _ = crawl("WIKI", WIKI_SEEDS, wiki_valid, "https://hi.wikipedia.org", max_fetch=350)
    all_texts.extend(t)
    
    log.info("\n[2/3] PIB Hindi (policy documents)...")
    t, _ = crawl("PIB", PIB_SEEDS, pib_valid, "https://pib.gov.in", max_fetch=150)
    all_texts.extend(t)
    
    log.info("\n[3/3] BBC Hindi (analysis)...")
    t, _ = crawl("BBC", BBC_SEEDS, bbc_valid, "https://www.bbc.com", max_fetch=100)
    all_texts.extend(t)
    
    total_tc = sum(token_count(t) for t in all_texts)
    log.info("\n" + "=" * 72)
    log.info("TOTAL: %d pages, %dk tokens", len(all_texts), total_tc // 1000)
    log.info("=" * 72)
    
    log.info("\nProcessing...")
    sentences = split_sentences(" ".join(all_texts))
    log.info("Sentences: %d", len(sentences))
    
    seen, unique = set(), []
    for s in sentences:
        k = s[:150]
        if k not in seen:
            seen.add(k)
            unique.append(s)
    log.info("Unique: %d", len(unique))
    
    # Primary
    log.info("\n[PRIMARY] %d-%d tokens...", MIN_PRIMARY, MAX_PRIMARY)
    primary = build_chunks_tier(unique, MIN_PRIMARY, MAX_PRIMARY)
    log.info("Generated: %d", len(primary))
    
    seen_chunks, primary_final = set(), []
    for txt in primary:
        k = txt[:280]
        if k not in seen_chunks:
            seen_chunks.add(k)
            primary_final.append({"text": txt, "token_count": token_count(txt), "tier": "primary"})
    log.info("After dedup: %d", len(primary_final))
    
    # Secondary
    secondary_final = []
    shortage = TARGET - len(primary_final)
    if shortage > 0:
        log.info("\n[SECONDARY] Need %d more...", shortage)
        secondary = build_chunks_tier(unique, MIN_SECONDARY, MAX_SECONDARY)
        log.info("Generated: %d", len(secondary))
        for txt in secondary:
            k = txt[:280]
            if k not in seen_chunks:
                seen_chunks.add(k)
                secondary_final.append({"text": txt, "token_count": token_count(txt), "tier": "secondary"})
        log.info("After dedup: %d", len(secondary_final))
    
    all_final = (primary_final + secondary_final)[:TARGET]
    primary_count = sum(1 for x in all_final if x["tier"] == "primary")
    secondary_count = len(all_final) - primary_count
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in all_final:
            f.write(json.dumps({"text": rec["text"], "token_count": rec["token_count"]}, ensure_ascii=False) + "\n")
    
    counts = [r["token_count"] for r in all_final]
    log.info("\n" + "=" * 72)
    log.info("✅ SAVED: %d samples → %s", len(all_final), OUTPUT)
    log.info("=" * 72)
    log.info("PRIMARY (%d-%d): %d samples", MIN_PRIMARY, MAX_PRIMARY, primary_count)
    if secondary_count > 0:
        log.info("SECONDARY (%d-%d): %d samples", MIN_SECONDARY, MAX_SECONDARY, secondary_count)
    if counts:
        log.info("Tokens: min=%d max=%d avg=%.1f", min(counts), max(counts), sum(counts)/len(counts))
    log.info("=" * 72)
    
    if len(all_final) < TARGET:
        log.error("\n❌ SHORT: %d/%d", len(all_final), TARGET)
    else:
        log.info("\n🎉 SUCCESS!")
        print("\n── Preview ──")
        for rec in all_final[:2]:
            print(f"{rec['token_count']} tokens: {rec['text'][:120]}...\n")

if __name__ == "__main__":
    main()