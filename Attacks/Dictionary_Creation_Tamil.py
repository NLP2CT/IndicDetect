import json
import os
import re
import time
from collections import Counter
from openai import OpenAI

INPUT_FILE: str = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/In_Distribution/Tamil/Test.json"

QWEN_MODEL: str   = "qwen-plus"
LABEL_KEY: str    = "label"
TEXT_KEY: str     = "text"
TARGET_LABEL: str = "LLM"
BATCH_SIZE: int   = 30
MIN_WORD_LEN: int = 2
COVERAGE_TARGET: float = 80.0

_TAMIL_RE: re.Pattern = re.compile(r'[\u0B80-\u0BFF]{2,}')

_CLIENT = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)


def make_output_dir() -> str:
    base = os.path.dirname(INPUT_FILE)
    out  = os.path.join(base, "Dictionaries")
    os.makedirs(out, exist_ok=True)
    print(f"Output directory ready: {out}")
    return out


def compute_coverage(llm_samples: list, top_set: set) -> float:
    coverages = [
        sum(1 for w in words if w in top_set) / len(words) * 100
        for words in llm_samples if words
    ]
    return sum(coverages) / len(coverages) if coverages else 0.0


def select_top_words(data: list) -> list[str]:
    print("\nExtracting Tamil words from LLM samples...")
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
                words = [w for w in _TAMIL_RE.findall(text) if len(w) >= MIN_WORD_LEN]
                if words:
                    llm_samples.append(words)
                    for w in words:
                        freq[w] += 1

    total_unique = len(freq)
    print(f"\r   Scanned {total_entries} entries | LLM samples: {llm_count} | Unique words: {total_unique}")

    print(f"\nFinding minimum word count to achieve {COVERAGE_TARGET}% avg coverage...")
    print(f"{'Top-N':<10} {'Avg Coverage':>14}")
    print("-" * 26)

    ranked_words = [w for w, _ in freq.most_common()]
    chosen_n = total_unique

    for top_n in [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000, 50000, total_unique]:
        if top_n > total_unique:
            top_n = total_unique
        top_set = set(ranked_words[:top_n])
        avg_cov = compute_coverage(llm_samples, top_set)
        marker  = "  <- selected" if avg_cov >= COVERAGE_TARGET and chosen_n == total_unique else ""
        print(f"{top_n:<10} {avg_cov:>13.1f}%{marker}")
        if avg_cov >= COVERAGE_TARGET and chosen_n == total_unique:
            chosen_n = top_n
            break

    print(f"\nUsing TOP_N_WORDS = {chosen_n} ({COVERAGE_TARGET}% coverage target)")

    top_words = ranked_words[:chosen_n]
    print(f"\nTop words selected (showing first 20):")
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
                    "content": "நீங்கள் தமிழ் மொழி நிபுணர். பதில் எப்போதும் சரியான JSON மட்டுமே தரவும்."
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
        print(f"\n  API error: {exc}")
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
            "கீழே கொடுக்கப்பட்ட ஒவ்வொரு தமிழ் சொல்லுக்கும் மாற்று எழுத்துரு வடிவங்களின் பட்டியல் தரவும் "
            "(ஒரே சொல்லுக்கு வெவ்வேறு எழுத்துரு முறைகள் - எ.கா: புள்ளி/அனுஸ்வார வித்தியாசம், "
            "குறில்/நெடில் மாற்றம், பழைய vs புதிய எழுத்துமுறை). "
            "உண்மையில் பயன்படுத்தப்படும் மற்றும் இயல்பாகத் தோன்றும் வடிவங்களை மட்டும் சேர்க்கவும்.\n"
            "JSON மட்டும் தரவும்: {\"சொல்\": [\"வடிவம்1\", \"வடிவம்2\"], ...}\n"
            "அர்த்தமுள்ள மாற்று வடிவம் இல்லாத சொல்லை காலி பட்டியலுடன் மேப் செய்யவும் [].\n\n"
            "சொற்கள்:\n"
        ),
        "misspelling": (
            "கீழே கொடுக்கப்பட்ட ஒவ்வொரு தமிழ் சொல்லுக்கும், தமிழ் தாய்மொழி பேசுபவர் தட்டச்சு செய்யும்போது "
            "செய்யும் மிகவும் பொதுவான ஒரே ஒரு தவறான எழுத்துரு தரவும் - எ.கா: ஒத்த எழுத்துகளை மாற்றுவது "
            "(அ/ஆ, இ/ஈ, உ/ஊ, க/ங), மாத்திரை தவறுதல் அல்லது கூட்டுதல், உச்சரிப்பு குழப்பம்.\n"
            "JSON மட்டும் தரவும்: {\"சரியான_சொல்\": \"தவறான_வடிவம்\", ...}\n"
            "பொதுவான தவறு சாத்தியமில்லாத சொல்லை விட்டுவிடவும்.\n\n"
            "சொற்கள்:\n"
        ),
        "synonym": (
            "கீழே கொடுக்கப்பட்ட ஒவ்வொரு தமிழ் சொல்லுக்கும் ஒத்த பொருள் சொற்களின் பட்டியல் தரவும்:\n"
            "1. ஒரே அல்லது கிட்டத்தட்ட ஒரே பொருள் இருக்க வேண்டும்\n"
            "2. ஒரே இலக்கண வகை இருக்க வேண்டும் (பெயர்ச்சொல் எனில் பெயர்ச்சொல், வினைச்சொல் எனில் வினைச்சொல்)\n"
            "3. வாக்கியத்தில் இயல்பாக பயன்படுத்தலாம்\n"
            "4. மூல வடிவத்தில் இருக்க வேண்டும் (வேற்றுமை உருபுகள் இல்லாமல்)\n"
            "JSON மட்டும் தரவும்: {\"சொல்\": [\"ஒத்தசொல்1\", \"ஒத்தசொல்2\"], ...}\n"
            "நல்ல ஒத்த சொல் இல்லாத சொல்லை காலி பட்டியலுடன் தரவும் [].\n\n"
            "சொற்கள்:\n"
        ),
    }

    result        = {}
    total_batches = (len(words) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nBuilding {dict_type.replace('_', ' ').title()} dictionary ({len(words)} words, {total_batches} batches)...")

    for batch_num, i in enumerate(range(0, len(words), BATCH_SIZE), 1):
        batch = words[i: i + BATCH_SIZE]
        print(f"\n   Batch {batch_num}/{total_batches} - words: {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")
        print(f"   Calling Qwen API...", end="", flush=True)

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

        print(f"\r   Batch {batch_num}/{total_batches} done - {batch_added}/{len(batch)} words got entries | Total so far: {len(result)}")
        time.sleep(0.5)

    return result


def main() -> None:
    print("=" * 60)
    print("  Tamil Dictionary Builder (Qwen API)")
    print("=" * 60)
    print(f"  Input          : {INPUT_FILE}")
    print(f"  Model          : {QWEN_MODEL}")
    print(f"  Coverage target: {COVERAGE_TARGET}%")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    output_dir = make_output_dir()

    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"\nLoaded {len(data)} total entries from file")

    words = select_top_words(data)

    for dict_type, filename in [
        ("alternative_spelling", "tamil_alternative_spelling.json"),
        ("misspelling",          "tamil_misspelling.json"),
        ("synonym",              "tamil_synonym.json"),
    ]:
        d        = build_dict(words, dict_type)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(d)} entries -> {out_path}")

    print("\n" + "=" * 60)
    print("  All Tamil dictionaries built successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()