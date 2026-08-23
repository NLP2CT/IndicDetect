import json
import os
import re
import time
from collections import Counter
from openai import OpenAI

INPUT_FILE: str   = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/In_Distribution/Hindi/Test.json"

QWEN_MODEL: str   = "qwen-plus"
LABEL_KEY: str    = "label"
TEXT_KEY: str     = "text"
TARGET_LABEL: str = "LLM"
BATCH_SIZE: int   = 30
MIN_WORD_LEN: int = 2
COVERAGE_TARGET: float = 80.0   # auto-select TOP_N that hits this % avg coverage

_HINDI_RE: re.Pattern = re.compile(r'[\u0900-\u097F]{2,}')

_CLIENT = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)


def make_output_dir() -> str:
    base = os.path.dirname(INPUT_FILE)
    out  = os.path.join(base, "Dictionaries")
    os.makedirs(out, exist_ok=True)
    print(f"📁  Output directory ready: {out}")
    return out


def compute_coverage(llm_samples: list, top_set: set) -> float:
    coverages = [
        sum(1 for w in words if w in top_set) / len(words) * 100
        for words in llm_samples if words
    ]
    return sum(coverages) / len(coverages) if coverages else 0.0


def select_top_words(data: list) -> list[str]:
    print("\n🔍  Extracting Hindi words from LLM samples...")
    freq: Counter = Counter()
    llm_samples   = []
    total_entries = len(data)
    llm_count     = 0

    for idx, entry in enumerate(data, 1):
        print(f"\r   Scanning entry {idx}/{total_entries} ...", end="", flush=True)
        if isinstance(entry, dict) and entry.get(LABEL_KEY) == TARGET_LABEL:
            llm_count += 1
            text = entry.get(TEXT_KEY, "")
            if isinstance(text, str):
                words = [w for w in _HINDI_RE.findall(text) if len(w) >= MIN_WORD_LEN]
                if words:
                    llm_samples.append(words)
                    for w in words:
                        freq[w] += 1

    total_unique = len(freq)
    print(f"\r   Scanned {total_entries} entries | LLM samples: {llm_count} | Unique words: {total_unique}")

    # Auto-select TOP_N to hit COVERAGE_TARGET
    print(f"\n📊  Finding minimum word count to achieve {COVERAGE_TARGET}% avg coverage...")
    print(f"{'Top-N':<10} {'Avg Coverage':>14}")
    print("-" * 26)

    ranked_words = [w for w, _ in freq.most_common()]
    chosen_n = total_unique  # fallback: use all

    for top_n in [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000, 50000, total_unique]:
        if top_n > total_unique:
            top_n = total_unique
        top_set  = set(ranked_words[:top_n])
        avg_cov  = compute_coverage(llm_samples, top_set)
        marker   = "  ◀ selected" if avg_cov >= COVERAGE_TARGET and chosen_n == total_unique else ""
        print(f"{top_n:<10} {avg_cov:>13.1f}%{marker}")
        if avg_cov >= COVERAGE_TARGET and chosen_n == total_unique:
            chosen_n = top_n
            break

    print(f"\n✅  Using TOP_N_WORDS = {chosen_n} ({COVERAGE_TARGET}% coverage target)")

    top_words = ranked_words[:chosen_n]

    print(f"\n📋  Top words selected (showing first 20):")
    for rank, (word, count) in enumerate(freq.most_common(20), 1):
        print(f"   {rank:>4}. {word:<25} (freq: {count})")
    if chosen_n > 20:
        print(f"   ... and {chosen_n - 20} more words")

    return top_words


def call_qwen(prompt: str) -> str:
    try:
        completion = _CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "आप एक हिंदी भाषा विशेषज्ञ हैं। हमेशा केवल सही JSON में उत्तर दें।"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        print(f"\n  ⚠️  API error: {exc}")
        return ""


def parse_json_response(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


def build_dict(words: list[str], dict_type: str) -> dict:
    prompts = {
        "alternative_spelling": (
            "नीचे दिए गए प्रत्येक हिंदी शब्द के लिए वैकल्पिक वर्तनी रूपों की सूची दें "
            "(एक ही शब्द के विभिन्न लिखित रूप — जैसे: बिंदी vs चंद्रबिंदु, नुक्ता का प्रयोग, "
            "अर्ध-व्यंजन vs पूर्ण रूप, मे/में जैसे स्वर भेद, अनुस्वार के भिन्न रूप, "
            "पुरानी vs आधुनिक वर्तनी). "
            "केवल वे रूप शामिल करें जो वास्तव में प्रचलित हों और स्वाभाविक लगें।\n"
            "केवल JSON दें: {\"शब्द\": [\"रूप1\", \"रूप2\"], ...}\n"
            "यदि किसी शब्द का कोई अर्थपूर्ण वैकल्पिक रूप नहीं है तो उसे खाली सूची से मैप करें [].\n\n"
            "शब्द:\n"
        ),
        "misspelling": (
            "नीचे दिए गए प्रत्येक हिंदी शब्द के लिए वह एकमात्र सबसे सामान्य गलत वर्तनी बताएं "
            "जो एक हिंदी मातृभाषी टाइप करते समय कर सकता है — जैसे: मिलते-जुलते अक्षरों की अदला-बदली "
            "(अ/आ, इ/ई, उ/ऊ, ब/व, श/ष/स), गलत या छूटी हुई मात्रा, "
            "गलत अनुस्वार/अनुनासिक, या सामान्य उच्चारण भ्रम।\n"
            "केवल JSON दें: {\"सही_शब्द\": \"गलत_रूप\", ...}\n"
            "जिस शब्द की कोई सामान्य गलत वर्तनी संभव न हो उसे छोड़ दें।\n\n"
            "शब्द:\n"
        ),
        "synonym": (
            "नीचे दिए गए प्रत्येक हिंदी शब्द के लिए पर्यायवाची शब्दों की सूची दें:\n"
            "1. समान या बहुत मिलता-जुलता अर्थ हो\n"
            "2. एक ही शब्द-भेद हो (संज्ञा है तो संज्ञा, क्रिया है तो क्रिया)\n"
            "3. वाक्य में स्वाभाविक रूप से प्रतिस्थापित किया जा सके\n"
            "4. मूल रूप में हो (विभक्ति के बिना)\n"
            "केवल JSON दें: {\"शब्द\": [\"पर्याय1\", \"पर्याय2\"], ...}\n"
            "यदि किसी शब्द का कोई अच्छा पर्याय नहीं है तो खाली सूची दें [].\n\n"
            "शब्द:\n"
        ),
    }

    result        = {}
    total_batches = (len(words) + BATCH_SIZE - 1) // BATCH_SIZE
    icons         = {"alternative_spelling": "✏️", "misspelling": "❌", "synonym": "🔄"}
    icon          = icons.get(dict_type, "•")

    print(f"\n{icon}  Building {dict_type.replace('_', ' ').title()} dictionary ({len(words)} words, {total_batches} batches)...")

    for batch_num, i in enumerate(range(0, len(words), BATCH_SIZE), 1):
        batch = words[i: i + BATCH_SIZE]
        print(f"\n   Batch {batch_num}/{total_batches} — words: {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")
        print(f"   ⏳  Calling Qwen API...", end="", flush=True)

        prompt = prompts[dict_type] + "\n".join(batch)
        raw    = call_qwen(prompt)
        parsed = parse_json_response(raw)

        batch_added = 0
        for w, val in parsed.items():
            if dict_type == "misspelling":
                if isinstance(val, str) and val and val != w:
                    result[w] = val
                    batch_added += 1
            else:
                if isinstance(val, list) and val:
                    result[w] = val
                    batch_added += 1

        print(f"\r   ✅  Batch {batch_num}/{total_batches} done — {batch_added}/{len(batch)} words got entries | Total so far: {len(result)}")
        time.sleep(0.5)

    return result


def main() -> None:
    print("=" * 60)
    print("  Hindi Dictionary Builder (Qwen API)")
    print("=" * 60)
    print(f"  Input          : {INPUT_FILE}")
    print(f"  Model          : {QWEN_MODEL}")
    print(f"  Coverage target: {COVERAGE_TARGET}%")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"⚠️  Input file not found: {INPUT_FILE}")
        return

    output_dir = make_output_dir()

    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"\n📂  Loaded {len(data)} total entries from file")

    words = select_top_words(data)

    for dict_type, filename in [
        ("alternative_spelling", "hindi_alternative_spelling.json"),
        ("misspelling",          "hindi_misspelling.json"),
        ("synonym",              "hindi_synonym.json"),
    ]:
        d        = build_dict(words, dict_type)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        print(f"\n💾  Saved {len(d)} entries -> {out_path}")

    print("\n" + "=" * 60)
    print("  ✅  All Hindi dictionaries built successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()