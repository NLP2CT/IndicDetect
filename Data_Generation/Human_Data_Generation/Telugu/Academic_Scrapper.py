"""
Telugu Text Scraper for Auchithyam.com
======================================
Scrapes Telugu articles, extracts clean sentence-boundary samples,
tokenizes with XLM-RoBERTa (sentencepiece BPE), and writes JSONL output.

Token range: 350–500 XLM-RoBERTa tokens per sample.
Output: telugu_samples.jsonl  (fields: text, token_count, source_url, article_title)

Requirements:
    pip install requests beautifulsoup4 sentencepiece certifi
    (The XLM-RoBERTa sentencepiece model is auto-downloaded on first run)
"""

import re
import json
import time
import os
import ssl
import urllib.request
import logging
from pathlib import Path
from typing import Generator

import certifi
import requests
from bs4 import BeautifulSoup
import sentencepiece as spm

# ─── Configuration ───────────────────────────────────────────────────────────
# Phase 1 — first 800 samples:  token range [350, 500]  (longer, richer samples)
# Phase 2 — next  200 samples:  token range [300, 349]  (slightly shorter)
PHASE1_COUNT = 800
PHASE1_MIN   = 350
PHASE1_MAX   = 500

PHASE2_COUNT = 200
PHASE2_MIN   = 300
PHASE2_MAX   = 349

TARGET_COUNT = PHASE1_COUNT + PHASE2_COUNT   # = 1000 exactly
OUTPUT_FILE  = "telugu_samples.jsonl"
SPM_MODEL_PATH = "xlm_roberta.spm"
SPM_MODEL_URL = (
    "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"
)
REQUEST_DELAY = 1.5          # seconds between HTTP requests (be polite)
REQUEST_TIMEOUT = 20         # seconds
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─── URL index (from the provided document) ──────────────────────────────────
ARTICLES = {
    "అక్టోబర్-2020 సంపాదకీయం": "https://auchithyam.com/articles/editorial.php",
    "అక్టోబర్-2020 అభినందనలు": "https://auchithyam.com/articles/bbreddy.php",
    "అక్టోబర్-2020 గాయత్రి - భాగవతం": "https://auchithyam.com/articles/gayatri.php",
    "అక్టోబర్-2020 పురాణాల పరిచయము": "https://auchithyam.com/articles/drmaddulapalli.php",
    "అక్టోబర్-2020 దళిత స్త్రీ వాద నవలా తత్త్వం": "https://auchithyam.com/articles/profdarla.php",
    "అక్టోబర్-2020 శ్రీనాథయుగసాహిత్యం - రామకథ": "https://auchithyam.com/articles/srinatha.php",
    "అక్టోబర్-2020 మొల్ల రామాయణం మానవీయ విలువలు": "https://auchithyam.com/articles/molla.php",
    "నవంబర్-2020 తెలుగు సాహిత్యంలో ఆధునికత": "https://auchithyam.com/articles/drpmukundarao.php",
    "నవంబర్-2020 అమెరికా డయాస్ఫోరా తెలుగు కథలు": "https://auchithyam.com/articles/drpcr.php",
    "నవంబర్-2020 ఆధునికత దిశగా తెలుగు కావ్యాలు": "https://auchithyam.com/articles/drseshakala.php",
    "నవంబర్-2020 డా తిరుమల రామచంద్ర": "https://auchithyam.com/articles/jayap.php",
    "నవంబర్-2020 వేటూరి రసమాధురి": "https://auchithyam.com/articles/madhuriinguva.php",
    "డిశంబర్-2020 నరబలి జంతుహింస": "https://auchithyam.com/articles/drjmarkad.php",
    "డిశంబర్-2020 ఉత్తరాంధ్ర కవితా సంకలనాల్లో స్త్రీ జీవనం": "https://auchithyam.com/articles/drpsrinivas.php",
    "డిశంబర్-2020 తెలుగుసాహిత్యం మానవతావిలువలు": "https://auchithyam.com/articles/druvsssastry.php",
    "డిశంబర్-2020 సుభాషితసాహిత్యం": "https://auchithyam.com/articles/drvvsarma.php",
    "డిశంబర్-2020 సుందరీ నాగమణి కథలు": "https://auchithyam.com/articles/brajeswari.php",
    "జనవరి-జూన్-2021 వ్యాఖ్యానాల్లో అర్థ నిర్ణయం": "https://auchithyam.com/articles/profvnr.php",
    "జనవరి-జూన్-2021 తెలుగు సాహిత్యం భవిష్యత్తు": "https://auchithyam.com/articles/drykameswari.php",
    "జనవరి-జూన్-2021 తిలక్ సాహిత్యం": "https://auchithyam.com/articles/drsdhileeswar.php",
    "జనవరి-జూన్-2021 తెలంగాణ శిల్పకారుల సేవ": "https://auchithyam.com/articles/chsrilakshmi.php",
    "జనవరి-జూన్-2021 ఓం నమశ్శివాయ": "https://auchithyam.com/articles/vvls.php",
    "జులై-డిశంబర్-2021 నా సాహితీ జీవితం": "https://auchithyam.com/articles/drvsimmanna.php",
    "జులై-డిశంబర్-2021 నీతి సీస శతకం": "https://auchithyam.com/articles/drpcrneetisatakam.php",
    "జులై-డిశంబర్-2021 జాషువా ఫిరదౌసి కావ్యం": "https://auchithyam.com/articles/acharavi.php",
    "జులై-డిశంబర్-2021 జాషువా కవిత్వంలో కాల్పనికత": "https://auchithyam.com/articles/bhgs.php",
    "జులై-డిశంబర్-2021 ఓం నమశ్శివాయ 2": "https://auchithyam.com/articles/madhuinguvaom.php",
    "జనవరి-2022 బంజారా సంస్కృతి వైభవం": "https://auchithyam.com/articles/bmohannaik.php",
    "జనవరి-2022 రంధి నవల": "https://auchithyam.com/articles/pkrupakar.php",
    "జనవరి-2022 అభినవభాసుని రూపక నిర్మాణం": "https://auchithyam.com/articles/pseshaabhinava.php",
    "జనవరి-2022 కుమ్మరమొల్ల": "https://auchithyam.com/articles/gprasadrao.php",
    "జనవరి-2022 మల్లంపల్లి వీరేశ్వరశర్మ": "https://auchithyam.com/articles/drchdurgaprasad.php",
    "ఫిబ్రవరి-2022 శ్రీరామ లీలా విలాసము": "https://auchithyam.com/articles/drtvk.php",
    "ఫిబ్రవరి-2022 నన్నయ్య వృత్తౌచిత్యం": "https://auchithyam.com/articles/drchdurgaprasadnannayya.php",
    "ఫిబ్రవరి-2022 మనసాహిత్యం స్పర్శ సిద్ధాంతం": "https://auchithyam.com/articles/drrgrtouch.php",
    "ఫిబ్రవరి-2022 విజయ విలాసం": "https://auchithyam.com/articles/akellabalabhanu.php",
    "ఫిబ్రవరి-2022 శ్రీనాథుని కవిత్వాదర్శం": "https://auchithyam.com/articles/drelchuri.php",
    "మార్చి-2022 కర్మతత్త్వ విచారం": "https://auchithyam.com/articles/acharyavedulanannayya.php",
    "మార్చి-2022 సంస్కృత సాహిత్యము సంస్కరణాత్మక దృష్టి": "https://auchithyam.com/articles/mdsamsk.php",
    "మార్చి-2022 తెలుగు సాహిత్య పరిశోధకులు": "https://auchithyam.com/articles/drvnrpari.php",
    "మార్చి-2022 శ్రీనాథయుగసాహిత్యం ప్రక్రియావైవిధ్యం": "https://auchithyam.com/articles/drrvrsharmaprakriya.php",
    "మార్చి-2022 మబ్బుల్లో బొమ్మ నాటకం": "https://auchithyam.com/articles/drchsuseelamb.php",
    "ఏప్రిల్-2022 ఆంధ్ర ప్రయోగ రత్నాకరము": "https://auchithyam.com/articles/drchsavitri.php",
    "ఏప్రిల్-2022 ఇంటింటి భాగోతం నాటకం": "https://auchithyam.com/articles/drchsuseelaini.php",
    "ఏప్రిల్-2022 కరుణాసౌగతము": "https://auchithyam.com/articles/drjmrkadkaruna.php",
    "ఏప్రిల్-2022 అవధాన కవిత్వం": "https://auchithyam.com/articles/drrvrsharmaavadhanam.php",
    "ఏప్రిల్-2022 బాల సాహిత్య హితైషి": "https://auchithyam.com/articles/kandukuribhaskar.php",
    "మే-2022 రాజశేఖరచరిత్ర సమీక్ష": "https://auchithyam.com/advanced/may22_01.html",
    "మే-2022 మన గోవులు స్థితిగతులు": "https://auchithyam.com/advanced/may22_02.html",
    "మే-2022 తెలుగు వార్త ఛానళ్లు": "https://auchithyam.com/advanced/may22_03.html",
    "మే-2022 కందుకూరి శాకుంతల": "https://auchithyam.com/advanced/may22_04.html",
    "మే-2022 మడికి సింగన కృతులు": "https://auchithyam.com/advanced/may22_05.html",
    "జూన్-2022 జానపద సాహిత్యం నైవేద్యసంస్కృతి": "https://auchithyam.com/advanced/june22_01.html",
    "జూన్-2022 ఆదివాసి కళారూపాల సౌందర్యం": "https://auchithyam.com/advanced/june22_02.html",
    "జూన్-2022 పోతన భాగవతం భక్తితత్త్వం": "https://auchithyam.com/advanced/june22_03.html",
    "జూన్-2022 ఉత్తరాంధ్ర పత్రికల పాత్ర": "https://auchithyam.com/advanced/june22_04.html",
    "జూన్-2022 పార్వతీ శతకం": "https://auchithyam.com/advanced/june22_05.html",
    "జూలై-2022 వాసిలి వసంతకుమార్ కవిత్వం": "https://auchithyam.com/advanced/july22_01.html",
    "జూలై-2022 ఆధునిక జానపదగేయం": "https://auchithyam.com/advanced/vol3_issue7_july2022.html",
    "జూలై-2022 పింగళి లక్ష్మీకాంతం": "https://auchithyam.com/advanced/july22_04.html",
    "జూలై-2022 వసుచరిత్రకారునిపై కాళిదాసు": "https://auchithyam.com/advanced/july22_05.html",
    "ఆగస్ట్-2022 ఆదిశంకరుల అపరోక్షానుభూతి": "https://auchithyam.com/advanced/august22_01.html",
    "ఆగస్ట్-2022 శ్రీ శివభారతం": "https://auchithyam.com/advanced/august22_02.html",
    "ఆగస్ట్-2022 ప్రాచీనభారతం మహిళాసాధికారత": "https://auchithyam.com/advanced/august22_03.html",
    "ఆగస్ట్-2022 మహాభారతం చార్వాక వధ": "https://auchithyam.com/advanced/august22_04.html",
    "ఆగస్ట్-2022 జానపద సాహిత్యం ప్రక్రియలు": "https://auchithyam.com/advanced/august22_05.html",
    "సెప్టెంబర్-2022 మన సామెతలు": "https://auchithyam.com/advanced/september22_01.html",
    "సెప్టెంబర్-2022 సంస్కృత రిపార్టీ కవిత": "https://auchithyam.com/advanced/september22_02.html",
    "సెప్టెంబర్-2022 సురవరం మొగ్గలు": "https://auchithyam.com/advanced/september22_03.html",
    "సెప్టెంబర్-2022 వట్టికోట జైలు కథలు": "https://auchithyam.com/advanced/september22_04.html",
    "సెప్టెంబర్-2022 కవితా పూరణం": "https://auchithyam.com/advanced/september22_05.html",
    "అక్టోబర్-2022 తెలుగు కవిత్వంలో రెక్కల ప్రస్థానం": "https://auchithyam.com/advanced/october22_01.html",
    "అక్టోబర్-2022 రాయలసీమ ఆకలికేకలు": "https://auchithyam.com/advanced/october22_02.html",
    "అక్టోబర్-2022 అప్సరసలు వంచిత మానవకాంతలు": "https://auchithyam.com/advanced/october22_03.html",
    "అక్టోబర్-2022 నండూరి సుందరీనాగమణి కుటుంబవిలువలు": "https://auchithyam.com/advanced/october22_04.html",
    "అక్టోబర్-2022 తెన్నేటి లక్ష్మీ నరసింహమూర్తి": "https://auchithyam.com/advanced/october22_05.html",
    "అక్టోబర్-2022 కందుకూరి వీరేశలింగం": "https://auchithyam.com/advanced/october22_06.html",
    "అక్టోబర్-2022 మహాభారత ఔచిత్యం": "https://auchithyam.com/advanced/october22_Special_Issue_01.html",
    "అక్టోబర్-2022 కవిత్రయభారతం ఆధునికజీవనం": "https://auchithyam.com/advanced/october22_Special_Issue_02.html",
    "అక్టోబర్-2022 భారతంలో కుటుంబజీవనచిత్రణ": "https://auchithyam.com/advanced/october22_Special_Issue_03.html",
    "అక్టోబర్-2022 శతకసారాంశం": "https://auchithyam.com/advanced/october22_Special_Issue_04.html",
    "నవంబర్-2022 శిల్పకావ్య కావ్యశిల్పం": "https://auchithyam.com/advanced/latest/november22_01.php",
    "నవంబర్-2022 ధర్మజుని ఔన్నత్యం": "https://auchithyam.com/advanced/november22_02.html",
    "నవంబర్-2022 తెన్నేటి సంఘటనాత్మక కవిత్వం": "https://auchithyam.com/advanced/november22_03.html",
    "నవంబర్-2022 ద్రౌపది నవల": "https://auchithyam.com/advanced/november22_04.html",
    "నవంబర్-2022 మౌసలత్రయ పర్వములు": "https://auchithyam.com/advanced/november22_05.html",
    "నవంబర్-2022 గురజాడ మహాకవి": "https://auchithyam.com/advanced/november22_06.html",
    "డిసెంబర్-2022 ఆదిశంకరుల అయిదు ప్రకరణాలు": "https://auchithyam.com/advanced/latest/december22_01.php",
    "డిసెంబర్-2022 ఔచిత్యమ్": "https://auchithyam.com/advanced/latest/december22_02.php",
    "డిసెంబర్-2022 తల్లాప్రగడ విశ్వసుందరమ్మ": "https://auchithyam.com/advanced/latest/december22_03.php",
    "డిసెంబర్-2022 దార్లమాట శతకం": "https://auchithyam.com/advanced/latest/december22_04.php",
    "డిసెంబర్-2022 పోతన భాగవతం రసౌచిత్యాలు": "https://auchithyam.com/advanced/latest/december22_05.php",
    "డిసెంబర్-2022 చెంచుల పాత్ర స్వాతంత్య్ర పోరాటం": "https://auchithyam.com/advanced/latest/december22_06.php",
    "డిసెంబర్-2022 రాయలసీమ కథాసాహిత్యం": "https://auchithyam.com/advanced/latest/december22_07.php",
}


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Tokenizer setup ──────────────────────────────────────────────────────────
def ensure_spm_model() -> spm.SentencePieceProcessor:
    """
    Download XLM-RoBERTa sentencepiece model if not present, then load it.

    Uses `requests` with the `certifi` CA bundle so it works correctly on
    macOS, where the system Python SSL store is not available by default.
    Falls back to an unverified SSL context only if certifi is unavailable.
    """
    if not Path(SPM_MODEL_PATH).exists():
        log.info("Downloading XLM-RoBERTa sentencepiece model (~5 MB)…")
        try:
            # Primary: use requests which ships its own certifi CA bundle
            resp = requests.get(
                SPM_MODEL_URL,
                headers=HEADERS,
                timeout=60,
                stream=True,
                verify=certifi.where(),
            )
            resp.raise_for_status()
            with open(SPM_MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as primary_err:
            log.warning(
                "Primary download failed (%s). Retrying without SSL verification…",
                primary_err,
            )
            # Fallback: bypass SSL verification (safe here — we only read a
            # public model file and verify its integrity via sentencepiece load)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            with opener.open(SPM_MODEL_URL, timeout=60) as src, \
                 open(SPM_MODEL_PATH, "wb") as dst:
                dst.write(src.read())

        log.info("Model saved to %s", SPM_MODEL_PATH)

    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL_PATH)
    log.info("Sentencepiece model loaded (%d vocab)", sp.GetPieceSize())
    return sp


def count_tokens(sp: spm.SentencePieceProcessor, text: str) -> int:
    """
    Token count as XLM-RoBERTa would produce:
      [CLS] + subword_pieces + [SEP]  → len(pieces) + 2
    We add +2 to match the actual model input length.
    """
    return len(sp.EncodeAsPieces(text)) + 2


# ─── Scraping helpers ─────────────────────────────────────────────────────────

# Telugu Unicode block U+0C00–U+0C7F
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

# English words (2+ ASCII letters)
ENGLISH_WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")

# ── Sentence-level noise that disqualifies the entire sentence ──────────────
SENTENCE_REJECT_RE = re.compile(
    r"(AUCHITHYAM|UGC[\s\-]?CARE|ISSN|DOI\s*:|Coverage\s+Period"
    r"|Save\s+as\s+PDF|Volume[-\s]\d|Issue[-\s]\d"
    r"|http[s]?://|www\."
    r"|@[a-zA-Z]|\d{10}"          # email/phone fragments
    r"|ఫోను\s*:|ఈమెయిల్\s*:)",   # author contact lines in Telugu
    re.IGNORECASE,
)

# ── Characters/sequences to scrub from text before processing ───────────────
SCRUB_PATTERNS = [
    (re.compile(r"http\S+"),                           ""),   # URLs
    (re.compile(r"www\.\S+"),                          ""),   # www
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), ""),  # emails
    (re.compile(r"\b\d{10,}\b"),                       ""),   # phone numbers
    (re.compile(r"@+"),                                ""),   # @ symbols
    (re.compile(r"#{2,}"),                             ""),   # ##
    (re.compile(r"\*{2,}"),                            ""),   # **
    (re.compile(r"_{2,}"),                             ""),   # __
    (re.compile(r"-{3,}"),                             ""),   # ---
    (re.compile(r"={3,}"),                             ""),   # ===
    (re.compile(r"\|{2,}"),                            ""),   # |||
    (re.compile(r"\[\d+\]"),                           ""),   # [1] citations
    (re.compile(r"\(\d{4}\)"),                         ""),   # (2022)
    (re.compile(r"Vol\.\s*\d+",   re.IGNORECASE),     ""),   # Vol.3
    (re.compile(r"pp\.\s*\d+[-–]\d+", re.IGNORECASE), ""),  # pp.12-14
    (re.compile(r"[\u0000-\u001F\u007F-\u009F]"),     " "),  # control chars
    (re.compile(r"[ \t]{2,}"),                         " "),  # extra spaces
]


def is_mostly_telugu(text: str, threshold: float = 0.60) -> bool:
    """True if ≥ threshold of non-space characters are in the Telugu Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 10:
        return False
    telugu_count = sum(1 for c in chars if TELUGU_RE.match(c))
    return (telugu_count / len(chars)) >= threshold


def english_word_ratio(text: str) -> float:
    """Fraction of whitespace-separated tokens that are pure ASCII words."""
    tokens = text.split()
    if not tokens:
        return 1.0
    eng = sum(1 for t in tokens if re.fullmatch(r"[a-zA-Z]{2,}", t))
    return eng / len(tokens)


def scrub(text: str) -> str:
    """Apply all SCRUB_PATTERNS and normalise whitespace."""
    for pattern, replacement in SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_article_paragraphs(html: str) -> list[str]:
    """
    Site-specific extractor for auchithyam.com.

    The page structure is always:
        <header banner>
        <journal metadata line>   ← AUCHITHYAM | Volume-X | ...
        <h1>  Article title       ← drop
        <h2>  Author name         ← drop
        <p>   Author affiliation  ← drop
        <hr>  ──────────────────  ← article body starts HERE
        <p>   Paragraph 1         ✓ keep
        <p>   Paragraph 2         ✓ keep
        ...
        <h2>  Section heading     ← drop (short, not a sentence)
        <p>   Paragraph N         ✓ keep

    Strategy:
      1. Remove all script/style/nav/footer/img tags.
      2. Find the <hr> that separates metadata from body; take only tags after it.
      3. From those, keep only <p> tags whose text is predominantly Telugu.
      4. Scrub each paragraph and return as a list.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags completely
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "header", "aside", "form", "img", "figure", "table"]):
        tag.decompose()

    # ── Find the <hr> separator ──────────────────────────────────────────────
    hr = soup.find("hr")
    if hr:
        # Collect all sibling/descendant <p> tags that come AFTER the <hr>
        body_paragraphs = []
        for tag in hr.find_all_next():
            if tag.name == "p":
                body_paragraphs.append(tag)
    else:
        # Fallback: grab every <p> on the page
        body_paragraphs = soup.find_all("p")

    # ── Filter and clean each paragraph ─────────────────────────────────────
    clean_paragraphs = []
    for p in body_paragraphs:
        text = p.get_text(separator=" ").strip()

        # Skip very short fragments (headings, labels, etc.)
        if len(text) < 40:
            continue

        # Skip paragraphs that are mostly English/metadata
        if not is_mostly_telugu(text):
            continue

        # Skip paragraphs with any hard disqualifying content
        if SENTENCE_REJECT_RE.search(text):
            continue

        # Scrub noise characters
        text = scrub(text)

        # Re-check after scrubbing
        if len(text) < 40 or not is_mostly_telugu(text):
            continue

        clean_paragraphs.append(text)

    return clean_paragraphs


def paragraphs_to_sentences(paragraphs: list[str]) -> list[str]:
    """
    Split each paragraph into individual sentences on Telugu/Devanagari
    full-stops. Each sentence passes a final quality gate:
      - length ≥ 30 chars
      - ≥ 60% Telugu characters
      - ≤ 15% English word ratio
      - no residual noise markers
      - starts with a Telugu character (not a digit, symbol, or Latin letter)
    """
    TELUGU_START_RE = re.compile(r"^[\u0C00-\u0C7F]")

    sentences = []
    for para in paragraphs:
        # Split on full-stop followed by whitespace
        parts = re.split(r"(?<=[.।])\s+", para)
        for part in parts:
            part = part.strip()

            # Minimum length
            if len(part) < 30:
                continue

            # Ensure ends with full stop
            if part[-1] not in ".।":
                part += "."

            # Must start with a Telugu character
            if not TELUGU_START_RE.match(part):
                continue

            # Telugu ratio gate
            if not is_mostly_telugu(part, threshold=0.60):
                continue

            # English word ratio gate
            if english_word_ratio(part) > 0.15:
                continue

            # Hard reject patterns
            if SENTENCE_REJECT_RE.search(part):
                continue

            sentences.append(part)

    return sentences


def build_samples(
    sentences: list[str],
    sp: spm.SentencePieceProcessor,
    min_tok: int,
    max_tok: int,
) -> Generator[tuple[str, int], None, None]:
    """
    Strict non-overlapping greedy windowing.

    Accumulates consecutive sentences until the token count is in
    [min_tok, max_tok], then emits and starts a fresh window.

    Rules:
    - NEVER yield a sample with token_count < min_tok
    - NEVER yield a sample with token_count > max_tok
    - Single sentence alone exceeds max_tok  -> skip it entirely
    - Tail of article never reaches min_tok  -> discard silently
    """
    i = 0
    n = len(sentences)

    while i < n:
        window: list[str] = []
        j = i

        while j < n:
            candidate = " ".join(window + [sentences[j]])
            tc = count_tokens(sp, candidate)

            if tc > max_tok:
                if not window:
                    # Single sentence exceeds max_tok — skip it
                    i = j + 1
                else:
                    # Check window without this sentence
                    tc_now = count_tokens(sp, " ".join(window))
                    if min_tok <= tc_now <= max_tok:
                        yield " ".join(window), tc_now
                    i = j
                break

            # Sentence fits — add it
            window.append(sentences[j])

            if tc >= min_tok:
                # Strictly in range — emit and move on
                yield " ".join(window), tc
                i = j + 1
                break

            j += 1

        else:
            # Exhausted all sentences — only emit if strictly in range
            if window:
                tc_now = count_tokens(sp, " ".join(window))
                if min_tok <= tc_now <= max_tok:
                    yield " ".join(window), tc_now
            i = n


# ─── Main pipeline ────────────────────────────────────────────────────────────
def fetch_html(url: str) -> str | None:
    """Fetch URL and return HTML string, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except requests.RequestException as exc:
        log.warning("Failed to fetch %s — %s", url, exc)
        return None


def run_phase(
    phase_name: str,
    need: int,
    min_tok: int,
    max_tok: int,
    sp: spm.SentencePieceProcessor,
    out_f,
) -> int:
    """
    Iterate over all articles and collect exactly `need` samples
    in the token range [min_tok, max_tok].
    Returns the number of samples actually written.
    """
    collected = 0
    for title, url in ARTICLES.items():
        if collected >= need:
            break

        log.info("[%s] Fetching: %s", phase_name, url)
        html = fetch_html(url)
        if html is None:
            continue

        paragraphs = extract_article_paragraphs(html)
        if not paragraphs:
            log.warning("  ↳ No usable paragraphs: %s", title)
            time.sleep(REQUEST_DELAY)
            continue

        sentences = paragraphs_to_sentences(paragraphs)
        log.info("  ↳ %d clean sentences", len(sentences))

        article_samples = 0
        for sample_text, token_count in build_samples(sentences, sp, min_tok, max_tok):
            if collected >= need:
                break
            record = {
                "text": sample_text,
                "token_count": token_count,
                "source_url": url,
                "article_title": title,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            article_samples += 1
            collected += 1

        log.info("  ↳ %d samples written [%s]", article_samples, phase_name)
        time.sleep(REQUEST_DELAY)

    return collected


def main():
    sp = ensure_spm_model()
    out_path = Path(OUTPUT_FILE)

    with out_path.open("w", encoding="utf-8") as out_f:

        # ── Phase 1: 800 samples with token range [350, 500] ─────────────────
        log.info("=" * 60)
        log.info("PHASE 1 — collecting %d samples in range [%d, %d]",
                 PHASE1_COUNT, PHASE1_MIN, PHASE1_MAX)
        log.info("=" * 60)
        p1 = run_phase("Phase-1", PHASE1_COUNT, PHASE1_MIN, PHASE1_MAX, sp, out_f)
        log.info("Phase 1 done: %d / %d samples collected.", p1, PHASE1_COUNT)

        # ── Phase 2: 200 samples with token range [300, 349] ─────────────────
        log.info("=" * 60)
        log.info("PHASE 2 — collecting %d samples in range [%d, %d]",
                 PHASE2_COUNT, PHASE2_MIN, PHASE2_MAX)
        log.info("=" * 60)
        p2 = run_phase("Phase-2", PHASE2_COUNT, PHASE2_MIN, PHASE2_MAX, sp, out_f)
        log.info("Phase 2 done: %d / %d samples collected.", p2, PHASE2_COUNT)

    total_written = p1 + p2
    log.info("=" * 60)
    log.info("Total samples written: %d → %s", total_written, out_path)
    if total_written == TARGET_COUNT:
        log.info("✓ Exactly %d samples collected.", TARGET_COUNT)
    else:
        log.warning("⚠ Expected %d but got %d.", TARGET_COUNT, total_written)
    log.info("=" * 60)

    # ── Validation ───────────────────────────────────────────────────────────
    log.info("Running validation…")
    range_errors = 0
    noise_warnings = 0
    p1_ok = p2_ok = True

    with out_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            rec = json.loads(line)
            tc  = rec["token_count"]
            txt = rec["text"]

            # Phase-aware range check
            if lineno <= PHASE1_COUNT:
                in_range = PHASE1_MIN <= tc <= PHASE1_MAX
                if not in_range:
                    p1_ok = False
            else:
                in_range = PHASE2_MIN <= tc <= PHASE2_MAX
                if not in_range:
                    p2_ok = False

            if not in_range:
                log.error("Line %d token_count=%d out of expected range", lineno, tc)
                range_errors += 1

            if SENTENCE_REJECT_RE.search(txt):
                log.warning("Line %d residual noise: %.80s", lineno, txt)
                noise_warnings += 1

            if english_word_ratio(txt) > 0.15:
                log.warning("Line %d too much English: %.80s", lineno, txt)
                noise_warnings += 1

            if not txt.strip().endswith((".", "।")):
                log.warning("Line %d no full-stop at end", lineno)

            if not re.match(r"^[\u0C00-\u0C7F]", txt.strip()):
                log.warning("Line %d does not start with Telugu char", lineno)

    log.info("─" * 60)
    log.info("Phase 1 (lines   1–800) range [%d,%d]: %s",
             PHASE1_MIN, PHASE1_MAX, "✓ OK" if p1_ok else "✗ ERRORS")
    log.info("Phase 2 (lines 801–1000) range [%d,%d]: %s",
             PHASE2_MIN, PHASE2_MAX, "✓ OK" if p2_ok else "✗ ERRORS")
    if range_errors == 0 and noise_warnings == 0:
        log.info("✓ All %d samples clean.", total_written)
    else:
        if range_errors:
            log.error("✗ %d out-of-range samples.", range_errors)
        if noise_warnings:
            log.warning("⚠ %d samples flagged for noise/English.", noise_warnings)


if __name__ == "__main__":
    main()