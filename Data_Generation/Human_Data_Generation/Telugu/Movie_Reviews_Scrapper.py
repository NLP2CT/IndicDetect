"""
Telugu Movie Review Scraper — 123telugu.com/telugu
===================================================

URL Sources (two-stage, combined & deduplicated):
  1. CURATED_URLS  — 260 handpicked URLs (2014-2022) from the provided
                     famous-movies list. All on 123telugu.com only.
  2. Wayback CDX   — discovers extra review URLs from 2014-2021 that
                     actually existed (HTTP-200 verified by CDX).
                     Used for URL discovery ONLY.
                     ALL content is fetched LIVE from 123telugu.com.

Sampler: STRICT NON-OVERLAPPING greedy walk.
  • Once sentences [i..j] are yielded as one sample, next sample starts
    at sentence [j+1]. Sentences are NEVER reused across samples.
  • Every sample starts at the first char of a sentence  (→ Telugu char)
  • Every sample ends   at the last  char of a sentence  (→ full-stop)
  • 380 ≤ token_count ≤ 420

Output: 1000 samples as JSONL  { "text": "...", "token_count": N }

Requirements:
    pip install requests beautifulsoup4 sentencepiece certifi
"""

import re, json, time, ssl, urllib.request, logging
from pathlib import Path
from typing import Generator

import certifi, requests
from bs4 import BeautifulSoup
import sentencepiece as spm

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_COUNT    = 1000
TOKEN_MIN       = 380
TOKEN_MAX       = 420
OUTPUT_FILE     = "telugu_review_samples.jsonl"
SPM_MODEL_PATH  = "xlm_roberta.spm"
SPM_MODEL_URL   = ("https://huggingface.co/xlm-roberta-base/resolve/main/"
                   "sentencepiece.bpe.model")
TELUGU_DOMAIN   = "123telugu.com"
REQUEST_DELAY   = 1.5      # seconds between requests (be polite)
REQUEST_TIMEOUT = 25
MAX_RETRIES     = 2

# Wayback CDX — URL DISCOVERY ONLY, content fetched live from 123telugu.com
CDX_API = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=123telugu.com/telugu/reviews/*"
    "&output=json"
    "&fl=original"
    "&filter=statuscode:200"
    "&filter=original:.*review.*"
    "&collapse=urlkey"
    "&from={from_year}0101"
    "&to={to_year}1231"
    "&limit=800"
)
CDX_YEAR_RANGES = [(2014, 2017), (2018, 2021)]   # two CDX calls

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "te-IN,te;q=0.9,en;q=0.5",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

B = "https://www.123telugu.com"

# ── Curated URLs (260 URLs, 2014-2022) ───────────────────────────────────────
CURATED_URLS = [
    # ══════════════ 2014 (50) ════════════════════════════════════════════════
    B+"/telugu/reviews/stylish-commercially-entertainer.html",
    B+"/telugu/reviews/review-legend-explosive-emotional-entertainer.html",
    B+"/telugu/reviews/a-true-celebration-of-togetherness.html",
    B+"/telugu/reviews/review-1-nenokkadine-too-complex-for-common-man.html",
    B+"/telugu/reviews/drishyam-telugu-movie-review.html",
    B+"/telugu/reviews/power-movie-review-in-telugu.html",
    B+"/telugu/reviews/loukyam-telugu-review.html",
    B+"/telugu/news/govindudu-andarivadele-movie-review-in-telugu.html",
    B+"/telugu/reviews/run-raja-run-telugu-movie-review.html",
    B+"/telugu/reviews/karthikeya-movie-review-in-telugu.html",
    B+"/telugu/reviews/review-oohalu-gusagusalade-refreshing-romance-with-comedy.html",
    B+"/telugu/reviews/geethanjali-telugu-movie-review.html",
    B+"/telugu/reviews/heart-attack-review.html",
    B+"/telugu/reviews/review-kotha-janta-selfish-love-story.html",
    B+"/telugu/reviews/alludu-sreenu-movie-review-in-telugu.html",
    B+"/telugu/reviews/lovers-telugu-movie-review.html",
    B+"/telugu/reviews/dikkulu-chudaku-ramayya-movie-review-in-telugu.html",
    B+"/telugu/reviews/oka-laila-kosam-movie-review-in-telugu.html",
    B+"/telugu/reviews/current-theega-movie-review.html",
    B+"/telugu/reviews/pilla-nuvvu-leni-jeevitham-movie-review.html",
    B+"/telugu/reviews/mukunda-telugu-review.html",
    B+"/telugu/reviews/review-prathinidhi-good-concept-average-execution.html",
    B+"/telugu/reviews/mahesh-babu-aagadu-movie-review.html",
    B+"/telugu/reviews/ntr-rabhasa-movie-review-in-telugu.html",
    B+"/telugu/reviews/review-bangaru-kodipetta-this-hen-is-too-slow.html",
    B+"/telugu/reviews/paisa-no-paisa-vasool.html",
    B+"/telugu/reviews/telugu-review-autonagar-surya-surya-ok.html",
    B+"/telugu/reviews/pandavulu-pandavulu-thummedha-review.html",
    B+"/telugu/reviews/vikramasimha-movie-review-in-telugu.html",
    B+"/telugu/reviews/jaihind-2-movie-review-in-telugu.html",
    B+"/telugu/reviews/naa-bangaaru-talli-movie-review-in-telugu.html",
    B+"/telugu/reviews/bheemavaram-bullodu-stale-comedy.html",
    B+"/telugu/reviews/review-hrudaya-kaleyam-sampoos-show-all-the-way.html",
    B+"/telugu/reviews/review-chandamama-kathalu-boring-stories.html",
    B+"/telugu/reviews/review-anamika-gripping-thriller.html",
    B+"/telugu/reviews/review-ulava-charu-biryani-flovourless-recipe.html",
    B+"/telugu/reviews/bhale-bhale-magadivoy-review-in-telugu.html",
    B+"/telugu/reviews/review-aaha-kalyanam.html",
    B+"/telugu/reviews/review-jump-jilani-routine-and-stale-comedy.html",
    B+"/telugu/reviews/review-rowdy-mohan-babus-masterclass.html",
    B+"/telugu/reviews/movie-review-pratighatana-lacklustre-effort.html",
    B+"/telugu/reviews/malupu-movie-review-in-telugu.html",
    B+"/telugu/reviews/nen-local-movie-review-in-telugu.html",
    B+"/telugu/reviews/kalavathi-telugu-movie-review.html",
    B+"/telugu/reviews/janda-pai-kapiraju-movie-review-in-telugu.html",
    B+"/telugu/reviews/galipatam-telugu-movie-youthful-entertainer.html",
    B+"/telugu/reviews/iddarammayilatho-puris-stylish-offering.html",
    B+"/telugu/reviews/review-seethamma-vakitlo-sirimalle-chettu-a-beautifully-woven-emotional-tale.html",
    B+"/telugu/reviews/gunde-jaari-gallanthayyinde-movie-review-in-telugu.html",
    B+"/telugu/reviews/dk-bose-movie-review-in-telugu.html",

    # ══════════════ 2015 (38) ════════════════════════════════════════════════
    B+"/telugu/reviews/bahubali-telugu-review.html",
    B+"/telugu/reviews/srimanthudu-movie-review-in-telugu.html",
    B+"/telugu/reviews/temper-telugu-review.html",
    B+"/telugu/reviews/son-of-satyamurthy-movie-review-in-telugu.html",
    B+"/telugu/reviews/rudhramadevi-movie-review-in-telugu.html",
    B+"/telugu/reviews/patas-movie-review-in-telugu.html",
    B+"/telugu/reviews/kumari-21f-movie-review-in-telugu.html",
    B+"/telugu/reviews/gopala-gopala-movie-review-in-telugu.html",
    B+"/telugu/reviews/kanche-movie-review-in-telugu.html",
    B+"/telugu/reviews/malli-malli-idi-rani-roju-movie-telugu-review.html",
    B+"/telugu/reviews/yevade-subramanyam-movie-review-in-telugu.html",
    B+"/telugu/reviews/raju-gari-gadhi-movie-review-in-telugu.html",
    B+"/telugu/reviews/kerintha-telugu-review.html",
    B+"/telugu/reviews/kick-2-review-in-telugu.html",
    B+"/telugu/reviews/subramanyam-for-sale-review-in-telugu.html",
    B+"/telugu/reviews/pandaga-chesko-review.html",
    B+"/telugu/reviews/bengal-tiger-movie-review-in-telugu.html",
    B+"/telugu/reviews/loafer-movie-review-in-telugu.html",
    B+"/telugu/reviews/bhale-manchi-roju-movie-review-in-telugu.html",
    B+"/telugu/reviews/raghuvaran-b-tech-movie-review-in-telugu.html",
    B+"/telugu/reviews/best-actors-review-in-telugu.html",
    B+"/telugu/reviews/size-zero-movie-review-in-telugu.html",
    B+"/telugu/reviews/ladies-and-gentleman-movie-telugu-review.html",
    B+"/telugu/reviews/jakkanna-movie-review-in-telugu.html",
    B+"/telugu/reviews/rey-telugu-review.html",
    B+"/telugu/reviews/nuvve-naa-pranam-movie-review-in-telugu.html",
    B+"/telugu/reviews/bandipotu-telugu-review.html",
    B+"/telugu/reviews/okka-ammayi-thappa-movie-review-in-telugu.html",
    B+"/telugu/reviews/ganga-movie-review.html",
    B+"/telugu/reviews/chinnadana-nee-kosam-telugu-review.html",
    B+"/telugu/reviews/review-prema-katha-chitam-a-very-entertaining-thriller.html",
    B+"/telugu/reviews/dohchay-movie-review-in-telugu.html",
    B+"/telugu/reviews/sahasam-movie-review-in-telugu.html",
    B+"/telugu/reviews/hyper-movie-review-in-telugu.html",
    B+"/telugu/reviews/maa-abbayi-movie-review-in-telugu.html",
    B+"/telugu/reviews/majnu-movie-review-in-telugu.html",
    B+"/telugu/reviews/soggade-chinni-nayana-movie-review-in-telugu.html",
    B+"/telugu/reviews/krishnamma-kalipindi-iddarini-movie-review-in-telugu.html",

    # ══════════════ 2016 (38) ════════════════════════════════════════════════
    B+"/telugu/reviews/nannaku-prematho-movie-review-in-telugu.html",
    B+"/telugu/reviews/janatha-garage-movie-review-in-telugu.html",
    B+"/telugu/reviews/sarrainodu-movie-review-in-telugu.html",
    B+"/telugu/reviews/oopiri-movie-review-in-telugu.html",
    B+"/telugu/reviews/a-aa-movie-review-in-telugu.html",
    B+"/telugu/reviews/pelli-choopulu-movie-review-in-telugu.html",
    B+"/telugu/reviews/nenu-sailaja-movie-review-in-telugu.html",
    B+"/telugu/reviews/brahmotsavam-movie-review-in-telugu.html",
    B+"/telugu/reviews/supreme-movie-review-in-telugu.html",
    B+"/telugu/reviews/dhruva-movie-review-in-telugu.html",
    B+"/telugu/reviews/ekkadiki-pothavu-chinnavada-movie-review-in-telugu.html",
    B+"/telugu/reviews/kshanam-movie-review-in-telugu.html",
    B+"/telugu/reviews/express-raja-movie-review-in-telugu.html",
    B+"/telugu/reviews/kalyana-vaibhogame-movie-review-in-telugu.html",
    B+"/telugu/reviews/surya-vs-surya-movie-review-in-telugu.html",
    B+"/telugu/reviews/srirastu-subhamastu-movie-review-in-telugu.html",
    B+"/telugu/reviews/gentleman-movie-review-in-telugu.html",
    B+"/telugu/reviews/krishna-gaadi-veera-prema-gaadha-movie-review-in-telugu.html",
    B+"/telugu/reviews/babu-bangaram-movie-review-in-telugu.html",
    B+"/telugu/reviews/dictator-movie-review-in-telugu.html",
    B+"/telugu/reviews/manyam-puli-movie-review-in-telugu.html",
    B+"/telugu/reviews/bethaludu-movie-review-in-telugu.html",
    B+"/telugu/reviews/ism-movie-review-in-telugu.html",
    B+"/telugu/reviews/speedunnodu-movie-review-in-telugu.html",
    B+"/telugu/reviews/shourya-movie-review-in-telugu.html",
    B+"/telugu/reviews/saahasam-swaasaga-saagipo-movie-review-in-telugu.html",
    B+"/telugu/reviews/mental-madhilo-movie-review-in-telugu.html",
    B+"/telugu/reviews/jyo-achyutananda-movie-review-in-telugu.html",
    B+"/telugu/reviews/dwaraka-movie-review-in-telugu.html",
    B+"/telugu/reviews/thikka-movie-review-in-telugu.html",
    B+"/telugu/reviews/mister-movie-review-in-telugu.html",
    B+"/telugu/reviews/sardaar-gabbar-singh-movie-review-in-telugu.html",
    B+"/telugu/reviews/eedo-rakam-aado-rakam-movie-review-in-telugu.html",
    B+"/telugu/reviews/nenu-rowdy-ne-movie-review-in-telugu.html",
    B+"/telugu/reviews/okkadu-migiladu-movie-review-in-telugu.html",
    B+"/telugu/reviews/abhinetri-movie-review-in-telugu.html",
    B+"/telugu/reviews/rarandoi-veduka-chudham-movie-review-in-telugu.html",
    B+"/telugu/reviews/krishnashtami-movie-review-in-telugu.html",

    # ══════════════ 2017 (22) ════════════════════════════════════════════════
    B+"/telugu/reviews/baahubali-2-the-conclusion-movie-review-in-telugu.html",
    B+"/telugu/reviews/arjun-reddy-movie-review-in-telugu.html",
    B+"/telugu/reviews/khaidi-no-150-movie-review-in-telugu.html",
    B+"/telugu/reviews/gautamiputra-satakarni-movie-review-in-telugu.html",
    B+"/telugu/reviews/fidaa-movie-review-in-telugu.html",
    B+"/telugu/reviews/jai-lava-kusa-movie-review-in-telugu.html",
    B+"/telugu/reviews/spyder-movie-review-in-telugu.html",
    B+"/telugu/reviews/ghazi-attack-movie-review-in-telugu.html",
    B+"/telugu/reviews/nakshatram-movie-review-in-telugu.html",
    B+"/telugu/reviews/vunnadhi-okate-zindagi-movie-review-in-telugu.html",
    B+"/telugu/reviews/winner-movie-review-in-telugu.html",
    B+"/telugu/reviews/oxygen-movie-review-in-telugu.html",
    B+"/telugu/reviews/anando-brahma-movie-review-in-telugu.html",
    B+"/telugu/reviews/prema-katha-chitram-2-movie-review-in-telugu.html",
    B+"/telugu/reviews/duvvada-jagannadham-movie-review-in-telugu.html",
    B+"/telugu/reviews/bichagadu-movie-review-in-telugu.html",
    B+"/telugu/reviews/paisa-vasool-movie-review-in-telugu.html",
    B+"/telugu/reviews/yuddham-sharanam-movie-review-in-telugu.html",
    B+"/telugu/reviews/tholi-prema-movie-review-in-telugu.html",
    B+"/telugu/reviews/needi-naadi-oke-katha-movie-review-in-telugu.html",
    B+"/telugu/reviews/o-manishi-movie-review-in-telugu.html",
    B+"/telugu/reviews/rarandoi-veduka-chudham-movie-review-in-telugu.html",

    # ══════════════ 2018 (36) ════════════════════════════════════════════════
    B+"/telugu/reviews/rangasthalam-movie-review-in-telugu.html",
    B+"/telugu/reviews/mahanati-movie-review-in-telugu.html",
    B+"/telugu/reviews/bharat-ane-nenu-movie-review-in-telugu.html",
    B+"/telugu/reviews/geetha-govindam-movie-review-in-telugu.html",
    B+"/telugu/reviews/aravindha-sametha-veera-raghava-movie-review-in-telugu.html",
    B+"/telugu/reviews/taxiwaala-movie-review-in-telugu.html",
    B+"/telugu/reviews/goodachari-movie-review-in-telugu.html",
    B+"/telugu/reviews/rx-100-movie-review-in-telugu.html",
    B+"/telugu/reviews/co-kancharapalem-movie-review-in-telugu.html",
    B+"/telugu/reviews/bhaagamathie-movie-review-in-telugu.html",
    B+"/telugu/reviews/jai-simha-movie-review-in-telugu.html",
    B+"/telugu/reviews/chalo-movie-review-in-telugu.html",
    B+"/telugu/reviews/awe-movie-review-in-telugu.html",
    B+"/telugu/reviews/pawan-kalyan-agnyaathavaasi-movie-review-in-telugu.html",
    B+"/telugu/reviews/husharu-movie-review-in-telugu.html",
    B+"/telugu/reviews/sammohanam-movie-review-in-telugu.html",
    B+"/telugu/reviews/devadas-movie-review-in-telugu.html",
    B+"/telugu/reviews/u-turn-movie-review-in-telugu.html",
    B+"/telugu/reviews/antariksham-9000-kmph-movie-review-in-telugu.html",
    B+"/telugu/reviews/hello-guru-prema-kosame-movie-review-in-telugu.html",
    B+"/telugu/reviews/srinivasa-kalyanam-movie-review-in-telugu.html",
    B+"/telugu/reviews/pantham-movie-review-in-telugu.html",
    B+"/telugu/reviews/kirrak-party-movie-review-in-telugu.html",
    B+"/telugu/reviews/chi-la-sow-movie-review-in-telugu.html",
    B+"/telugu/reviews/bluff-master-movie-review-in-telugu.html",
    B+"/telugu/reviews/naa-nuvve-movie-review-in-telugu.html",
    B+"/telugu/reviews/intelligent-movie-review-in-telugu.html",
    B+"/telugu/reviews/padi-padi-leche-manasu-movie-review-in-telugu.html",
    B+"/telugu/reviews/sailaja-reddy-alludu-movie-review-in-telugu.html",
    B+"/telugu/reviews/next-enti-movie-review-in-telugu.html",
    B+"/telugu/reviews/savyasachi-movie-review-in-telugu.html",
    B+"/telugu/reviews/neevevaro-movie-review-in-telugu.html",
    B+"/telugu/reviews/silly-fellows-movie-review-in-telugu.html",
    B+"/telugu/reviews/happy-wedding-movie-review-in-telugu.html",
    B+"/telugu/reviews/nartanasala-movie-review-in-telugu.html",
    B+"/telugu/reviews/tholi-prema-movie-review-in-telugu.html",

    # ══════════════ 2019 (25) ════════════════════════════════════════════════
    B+"/telugu/reviews/maharshi-movie-review-in-telugu.html",
    B+"/telugu/reviews/saaho-movie-review-in-telugu.html",
    B+"/telugu/reviews/sye-raa-narasimha-reddy-movie-review-in-telugu.html",
    B+"/telugu/reviews/dear-comrade-movie-review-in-telugu.html",
    B+"/telugu/reviews/jersey-movie-review-in-telugu.html",
    B+"/telugu/reviews/majili-movie-review-in-telugu.html",
    B+"/telugu/reviews/vinaya-vidheya-rama-movie-review-in-telugu.html",
    B+"/telugu/reviews/f2-fun-and-frustration-movie-review-in-telugu.html",
    B+"/telugu/reviews/gaddalakonda-ganesh-movie-review-in-telugu.html",
    B+"/telugu/reviews/ismart-shankar-movie-review-in-telugu.html",
    B+"/telugu/reviews/evaru-movie-review-in-telugu.html",
    B+"/telugu/reviews/mr-majnu-movie-review-in-telugu.html",
    B+"/telugu/reviews/agent-sai-srinivasa-athreyya-movie-review-in-telugu.html",
    B+"/telugu/reviews/ruler-movie-review-in-telugu.html",
    B+"/telugu/reviews/ninu-veedani-needanu-nene-movie-review-in-telugu.html",
    B+"/telugu/reviews/gang-leader-movie-review-in-telugu.html",
    B+"/telugu/reviews/chitralahari-movie-review-in-telugu.html",
    B+"/telugu/reviews/mathu-vadalara-movie-review-in-telugu.html",
    B+"/telugu/reviews/middle-class-melodies-movie-review-in-telugu.html",
    B+"/telugu/reviews/prati-roju-pandaage-movie-review-in-telugu.html",
    B+"/telugu/reviews/ranarangam-movie-review-in-telugu.html",
    B+"/telugu/reviews/valmiki-movie-review-in-telugu.html",
    B+"/telugu/reviews/oh-baby-movie-review-in-telugu.html",
    B+"/telugu/reviews/entha-manchivadavuraa-movie-review-in-telugu.html",
    B+"/telugu/reviews/raju-gari-gadhi-3-movie-review-in-telugu.html",

    # ══════════════ 2020 (17) ════════════════════════════════════════════════
    B+"/telugu/reviews/ala-vaikunthapurramuloo-movie-review-in-telugu.html",
    B+"/telugu/reviews/sarileru-neekevvaru-movie-review-in-telugu.html",
    B+"/telugu/reviews/uppena-movie-review-in-telugu.html",
    B+"/telugu/reviews/bheeshma-movie-review-in-telugu.html",
    B+"/telugu/reviews/aakaasam-nee-haddhu-ra-movie-review-in-telugu.html",
    B+"/telugu/reviews/jaanu-movie-review-in-telugu.html",
    B+"/telugu/reviews/palasa-1978-movie-review-in-telugu.html",
    B+"/telugu/reviews/kanulu-kanulanu-dhochaayante-movie-review-in-telugu.html",
    B+"/telugu/reviews/colour-photo-movie-review-in-telugu.html",
    B+"/telugu/reviews/uma-maheswara-ugra-roopasya-movie-review-in-telugu.html",
    B+"/telugu/reviews/v-movie-review-in-telugu.html",
    B+"/telugu/reviews/world-famous-lover-movie-review-in-telugu.html",
    B+"/telugu/reviews/naandhi-movie-review-in-telugu.html",
    B+"/telugu/reviews/sreekaram-movie-review-in-telugu.html",
    B+"/telugu/reviews/solo-brathuke-so-better-movie-review-in-telugu.html",
    B+"/telugu/reviews/zombie-reddy-movie-review-in-telugu.html",
    B+"/telugu/reviews/mallesham-movie-review-in-telugu.html",

    # ══════════════ 2021 (16) ════════════════════════════════════════════════
    B+"/telugu/reviews/pushpa-movie-review-in-telugu.html",
    B+"/telugu/reviews/vakeel-saab-movie-review-in-telugu.html",
    B+"/telugu/reviews/krack-movie-review-in-telugu.html",
    B+"/telugu/reviews/jathi-ratnalu-movie-review-in-telugu.html",
    B+"/telugu/reviews/most-eligible-bachelor-movie-review-in-telugu.html",
    B+"/telugu/reviews/tuck-jagadish-movie-review-in-telugu.html",
    B+"/telugu/reviews/republic-movie-review-in-telugu.html",
    B+"/telugu/reviews/wild-dog-movie-review-in-telugu.html",
    B+"/telugu/reviews/rang-de-movie-review-in-telugu.html",
    B+"/telugu/reviews/narappa-movie-review-in-telugu.html",
    B+"/telugu/reviews/a1-express-movie-review-in-telugu.html",
    B+"/telugu/reviews/love-story-movie-review-in-telugu.html",
    B+"/telugu/reviews/check-movie-review-in-telugu.html",
    B+"/telugu/reviews/maestro-movie-review-in-telugu.html",
    B+"/telugu/reviews/drushyam-2-movie-review-in-telugu.html",
    B+"/telugu/reviews/samajavaragamana-movie-review-in-telugu.html",

    # ══════════════ 2022 (20) ════════════════════════════════════════════════
    B+"/telugu/reviews/rrr-movie-review-in-telugu.html",
    B+"/telugu/reviews/bheemla-nayak-movie-review-in-telugu.html",
    B+"/telugu/reviews/sarkaru-vaari-paata-movie-review-in-telugu.html",
    B+"/telugu/reviews/radhe-shyam-movie-review-in-telugu.html",
    B+"/telugu/reviews/sita-ramam-movie-review-in-telugu.html",
    B+"/telugu/reviews/major-movie-review-in-telugu.html",
    B+"/telugu/reviews/dj-tillu-movie-review-in-telugu.html",
    B+"/telugu/reviews/ante-sundaraniki-movie-review-in-telugu.html",
    B+"/telugu/reviews/karthikeya-2-movie-review-in-telugu.html",
    B+"/telugu/reviews/hit-2-movie-review-in-telugu.html",
    B+"/telugu/reviews/godfather-movie-review-in-telugu.html",
    B+"/telugu/reviews/bangarraju-movie-review-in-telugu.html",
    B+"/telugu/reviews/viraata-parvam-movie-review-in-telugu.html",
    B+"/telugu/reviews/f3-movie-review-in-telugu.html",
    B+"/telugu/reviews/dhamaka-movie-review-in-telugu.html",
    B+"/telugu/reviews/aadavallu-meeku-johaarlu-movie-review-in-telugu.html",
    B+"/telugu/reviews/aa-ammayi-gurinchi-meeku-cheppali-movie-review-in-telugu.html",
    B+"/telugu/reviews/liger-movie-review-in-telugu.html",
    B+"/telugu/reviews/good-luck-sakhi-movie-review-in-telugu.html",
    B+"/telugu/reviews/bimbisara-movie-review-in-telugu.html",
]

# ── Telugu helpers ─────────────────────────────────────────────────────────────
TELUGU_RE    = re.compile(r"[\u0C00-\u0C7F]")
TELUGU_START = re.compile(r"^[\u0C00-\u0C7F]")

REJECT_RE = re.compile(
    r"(123తెలుగు|రేటింగ్\s*:|నటీనటులు\s*:|దర్శకత్వం\s*:|నిర్మాత\s*:"
    r"|సంగీతం?\s*:|సినిమాటోగ్రఫీ\s*:|ఎడిటింగ్\s*:|విడుదల తేదీ\s*:"
    r"|http[s]?://|www\."
    r"|WhatsApp|Google\s*News|Facebook|Instagram|Twitter|Youtube"
    r"|ABOUT\s+US|CONTACT\s+US|DISCLAIMER|PRIVACY\s+POLICY"
    r"|Join Us|Follow Us|Add as a preferred"
    r"|హోమ్\s*\||\bవార్తలు\s*\||గ్యాలరీ\s*\||సమీక్షలు\s*\|"
    r"|వీడియోలు\s*\||ముఖాముఖి\s*\||English Version|Load more)",
    re.IGNORECASE,
)
SECTION_HEADER_RE = re.compile(
    r"^(కథ|స్టోరీ|ప్లస్\s*పాయింట్స్?|మైనస్\s*పాయింట్స్?"
    r"|విశ్లేషణ|మొత్తం\s*మీద|నటనలు|సాంకేతిక\s*విభాగం"
    r"|సంగీతం|ఛాయాగ్రహణం|దర్శకత్వం|నేపథ్యం)\s*:?\s*$"
)
SCRUB_RE = [
    (re.compile(r"http\S+"),                                            ""),
    (re.compile(r"www\.\S+"),                                           ""),
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), ""),
    (re.compile(r"\[\d+\]"),                                            ""),
    (re.compile(r"\(\d{4}\)"),                                          ""),
    (re.compile(r"[\u0000-\u001F\u007F-\u009F]"),                       " "),
    (re.compile(r"[ \t]{2,}"),                                          " "),
]

def is_mostly_telugu(text: str, threshold: float = 0.55) -> bool:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 10: return False
    return sum(1 for c in chars if TELUGU_RE.match(c)) / len(chars) >= threshold

def english_word_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens: return 1.0
    return sum(1 for t in tokens if re.fullmatch(r"[a-zA-Z]{2,}", t)) / len(tokens)

def scrub(text: str) -> str:
    for pat, rep in SCRUB_RE: text = pat.sub(rep, text)
    return re.sub(r"[ \t]+", " ", text).strip()

# ── Sentencepiece ──────────────────────────────────────────────────────────────
def ensure_spm_model() -> spm.SentencePieceProcessor:
    model_path = Path(SPM_MODEL_PATH)
    if not model_path.exists():
        log.info("Downloading XLM-RoBERTa sentencepiece model…")
        try:
            resp = requests.get(SPM_MODEL_URL, stream=True,
                                verify=certifi.where(), timeout=60)
            resp.raise_for_status()
            with model_path.open("wb") as f:
                for chunk in resp.iter_content(8192): f.write(chunk)
        except Exception:
            log.warning("certifi failed — retrying unverified…")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
            urllib.request.urlretrieve(SPM_MODEL_URL, model_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    log.info("Sentencepiece loaded  vocab=%d", sp.GetPieceSize())
    return sp

def count_tokens(sp: spm.SentencePieceProcessor, text: str) -> int:
    return len(sp.EncodeAsPieces(text)) + 2   # +2 for [CLS]/[SEP]

# ── HTTP ───────────────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)

def safe_get(url: str) -> tuple[str | None, str]:
    """Fetch URL. Returns (None, final_url) if redirected off 123telugu.com."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            final_url = resp.url
            if TELUGU_DOMAIN not in final_url:
                log.warning("  REDIRECT-SKIP %s → %s", url, final_url)
                return None, final_url
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text, final_url
        except requests.RequestException as exc:
            log.warning("  Attempt %d/%d failed %s : %s",
                        attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES: time.sleep(REQUEST_DELAY * 2)
    return None, ""

# ── Wayback CDX — URL discovery only ──────────────────────────────────────────
def discover_cdx_urls(seen: set[str]) -> list[str]:
    """
    Query Wayback CDX API to find additional review URLs from 123telugu.com/telugu.
    Returns only live 123telugu.com URLs — content is NEVER fetched from Wayback.
    Filters to Telugu review pages only, deduplicates against already-known URLs.
    """
    discovered: list[str] = []
    review_slug_re = re.compile(
        r"/telugu/(reviews|news)/[a-z0-9\-]+-"
        r"(review|samiksha|telugu-review|movie-review)"
        r"[a-z0-9\-]*\.html$",
        re.IGNORECASE,
    )
    for from_year, to_year in CDX_YEAR_RANGES:
        api_url = CDX_API.format(from_year=from_year, to_year=to_year)
        log.info("CDX query %d-%d …", from_year, to_year)
        try:
            resp = requests.get(api_url, timeout=60, verify=certifi.where())
            resp.raise_for_status()
            rows = resp.json()
        except Exception as exc:
            log.warning("CDX fetch failed (%s) — skipping", exc)
            continue

        count = 0
        for row in rows[1:]:   # row[0] is header
            raw_url = row[0]
            # Normalise: always use https://www.123telugu.com/...
            path = re.sub(r"^https?://(?:www\.)?123telugu\.com", "", raw_url)
            if not path.startswith("/telugu/"): continue
            if not review_slug_re.search(path): continue
            live_url = "https://www.123telugu.com" + path
            if live_url in seen: continue
            seen.add(live_url)
            discovered.append(live_url)
            count += 1

        log.info("  CDX %d-%d → %d new URLs", from_year, to_year, count)
        time.sleep(1)

    return discovered

# ── Extraction ─────────────────────────────────────────────────────────────────
def extract_paragraphs(html: str) -> list[str]:
    """
    Page structure (confirmed from screenshot):
      <table>             → metadata (rating/director/cast) → REMOVE
      <p><strong>కథ :</strong></p>   → section header      → SKIP
      <p>Telugu text…</p>            → review body         → KEEP
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","nav","footer",
                     "header","aside","form","img","figure","iframe"]):
        tag.decompose()
    for table in soup.find_all("table"): table.decompose()
    for tag in soup(["td","tr","th","tbody","thead"]): tag.decompose()

    body = (soup.find("div", class_="entry-content")
            or soup.find("div", class_="post-content")
            or soup.find("div", class_="td-post-content")
            or soup.find("article")
            or soup.find("div", id=re.compile(r"post-\d+"))
            or soup.find("body") or soup)

    paragraphs = []
    for p in body.find_all("p"):
        raw = p.get_text(separator=" ").strip()
        if len(raw) < 40: continue
        # Skip all-bold section headers (కథ:, ప్లస్ పాయింట్స్:, etc.)
        bold_text = " ".join(b.get_text() for b in p.find_all(["strong","b"]))
        if bold_text.strip() and len(bold_text.strip()) >= len(raw) * 0.80: continue
        if SECTION_HEADER_RE.match(raw): continue
        if REJECT_RE.search(raw): continue
        if not is_mostly_telugu(raw, 0.55): continue
        if english_word_ratio(raw) > 0.22: continue
        text = scrub(raw)
        if len(text) < 40 or not is_mostly_telugu(text): continue
        paragraphs.append(text)
    return paragraphs

def split_sentences(paragraph: str) -> list[str]:
    sentences = []
    for part in re.split(r"(?<=[.।])\s+", paragraph):
        part = part.strip()
        if len(part) < 25: continue
        if part[-1] not in ".।": part += "."
        if not TELUGU_START.match(part): continue
        if not is_mostly_telugu(part, 0.55): continue
        if english_word_ratio(part) > 0.18: continue
        if REJECT_RE.search(part): continue
        sentences.append(part)
    return sentences

def page_sentences(html: str) -> list[str]:
    sentences = []
    for para in extract_paragraphs(html):
        sentences.extend(split_sentences(para))
    return sentences

# ── STRICT NON-OVERLAPPING sampler ────────────────────────────────────────────
def build_samples(
    sentences: list[str],
    sp: spm.SentencePieceProcessor,
) -> Generator[tuple[str, int], None, None]:
    """
    STRICT NON-OVERLAPPING greedy walk.

    Rule: once sentences[i..j] are yielded as one sample, the NEXT sample
          starts at sentences[j+1].  Sentences are NEVER reused.

    Every sample:
      - starts at sentences[i][0]  → always a Telugu char  (checked in collector)
      - ends   at sentences[j][-1] → always '.' or '।'     (checked in collector)
      - TOKEN_MIN ≤ token_count ≤ TOKEN_MAX
    """
    i = 0
    n = len(sentences)

    while i < n:
        window: list[str] = []
        j = i

        while j < n:
            candidate = " ".join(window + [sentences[j]])
            tc = count_tokens(sp, candidate)

            if tc > TOKEN_MAX:
                # adding sentences[j] would exceed the ceiling
                if window:
                    tc_now = count_tokens(sp, " ".join(window))
                    if TOKEN_MIN <= tc_now <= TOKEN_MAX:
                        yield " ".join(window), tc_now
                    # whether or not we yielded, advance past this whole window
                    i = j      # next sample starts at sentences[j]
                else:
                    # single sentence already exceeds TOKEN_MAX — skip it
                    i = j + 1
                break

            window.append(sentences[j])

            if tc >= TOKEN_MIN:
                # window is exactly in [TOKEN_MIN, TOKEN_MAX]
                yield " ".join(window), tc
                i = j + 1   # ← advance past ALL sentences used
                break        # start a completely fresh window

            j += 1

        else:
            # reached end of list without hitting TOKEN_MAX or TOKEN_MIN
            if window:
                tc_now = count_tokens(sp, " ".join(window))
                if TOKEN_MIN <= tc_now <= TOKEN_MAX:
                    yield " ".join(window), tc_now
            break   # exhausted all sentences

# ── Collector ──────────────────────────────────────────────────────────────────
def collect(sp: spm.SentencePieceProcessor, out_path: Path) -> int:
    collected = skip_redir = skip_empty = 0

    # Step 1: build deduplicated curated list
    seen: set[str] = set()
    url_list: list[str] = []
    for u in CURATED_URLS:
        if u not in seen:
            seen.add(u); url_list.append(u)
    log.info("Curated URLs : %d unique", len(url_list))

    # Step 2: top up with Wayback CDX discovery if needed
    cdx_urls = discover_cdx_urls(seen)
    log.info("CDX extra    : %d new URLs", len(cdx_urls))
    url_list.extend(cdx_urls)
    log.info("Total pool   : %d URLs", len(url_list))

    with out_path.open("w", encoding="utf-8") as out_f:
        for url in url_list:
            if collected >= TARGET_COUNT: break

            log.info("[%d/%d]  %s", collected, TARGET_COUNT, url)
            html, final_url = safe_get(url)
            time.sleep(REQUEST_DELAY)

            if html is None:
                if TELUGU_DOMAIN not in final_url: skip_redir += 1
                continue

            sentences = page_sentences(html)
            if not sentences:
                log.info("  ↳ no usable sentences — skip")
                skip_empty += 1
                continue

            log.info("  ↳ %d sentences", len(sentences))
            page_count = 0

            for sample_text, token_count in build_samples(sentences, sp):
                if collected >= TARGET_COUNT: break
                s = sample_text.strip()
                # Final guard: must start with Telugu, end with full-stop
                if not TELUGU_START.match(s): continue
                if s[-1] not in ".।": continue
                out_f.write(
                    json.dumps({"text": s, "token_count": token_count},
                               ensure_ascii=False) + "\n"
                )
                out_f.flush()
                page_count += 1
                collected  += 1

            if page_count:
                log.info("  ↳ +%d samples  total=%d/%d",
                         page_count, collected, TARGET_COUNT)

    log.info("─" * 60)
    log.info("Collected=%d  redirect_skip=%d  empty_skip=%d",
             collected, skip_redir, skip_empty)
    return collected

# ── Validator ──────────────────────────────────────────────────────────────────
def validate(out_path: Path, total: int) -> None:
    log.info("Validating %s …", out_path)
    range_err = warns = 0

    with out_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            rec = json.loads(line)
            tc  = rec["token_count"]
            txt = rec["text"].strip()

            if not (TOKEN_MIN <= tc <= TOKEN_MAX):
                log.error("  L%d  tc=%d  OUT OF RANGE [%d,%d]",
                          lineno, tc, TOKEN_MIN, TOKEN_MAX)
                range_err += 1
            if not TELUGU_START.match(txt):
                log.warning("  L%d  does not start with Telugu char", lineno)
                warns += 1
            if txt[-1] not in ".।":
                log.warning("  L%d  does not end with full-stop", lineno)
                warns += 1
            if REJECT_RE.search(txt):
                log.warning("  L%d  noise/metadata detected", lineno)
                warns += 1
            if english_word_ratio(txt) > 0.18:
                log.warning("  L%d  high English word ratio", lineno)
                warns += 1

    log.info("═" * 60)
    log.info("Samples : %d / %d", total, TARGET_COUNT)
    log.info("Range   : [%d, %d] tokens", TOKEN_MIN, TOKEN_MAX)
    log.info("Sampler : STRICT NON-OVERLAPPING (zero sentence reuse)")
    if range_err == 0 and warns == 0:
        log.info("✓ ALL %d samples passed validation.", total)
    else:
        if range_err: log.error("✗ %d range error(s)", range_err)
        if warns:     log.warning("⚠  %d warning(s)", warns)

# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    sp = ensure_spm_model()
    log.info("═" * 60)
    log.info("Curated URLs : 260  (2014-2022, famous Telugu films)")
    log.info("CDX fallback : up to 800 extra URLs per year range")
    log.info("Sampler      : STRICT NON-OVERLAPPING (no duplicate sentences)")
    log.info("Target       : %d samples  tokens=[%d,%d]",
             TARGET_COUNT, TOKEN_MIN, TOKEN_MAX)
    log.info("═" * 60)
    out_path = Path(OUTPUT_FILE)
    total = collect(sp, out_path)
    validate(out_path, total)

if __name__ == "__main__":
    main()