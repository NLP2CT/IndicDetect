import os
import json
import time
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from transformers import AutoTokenizer

print("RUNNING Creative Writing SCRIPT v16 — DEEPSEEK API TELUGU PARALLEL (KEYWORDS + DIVERSITY)")

INPUT_JSON_PATH = "/Users/user/Desktop/IndicDetect/Benchmark_Data/Human_Written/Telugu/Movie_Reviews/telugu_review_samples.jsonl"
OUTPUT_JSON_PATH = INPUT_JSON_PATH.rsplit(".", 1)[0] + "__movie_reviews_articles_DeepSeek_v3.json"

# ─── DEEPSEEK CONFIGURATION ─────────────────────────────────────────────
MODEL_NAME = "deepseek-chat"                # or "deepseek-reasoner"
BASE_URL = "https://api.deepseek.com/v1"    # DeepSeek OpenAI-compatible endpoint
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"        # environment variable name
# ────────────────────────────────────────────────────────────────────────

TEMPERATURE = 0.85
MAX_MODEL_TOKENS = 2048
MIN_REVIEW_TOKENS = 400
MAX_REVIEW_TOKENS = 530
SLEEP_SECONDS = 0.1

MAX_ATTEMPTS_PER_SAMPLE = 20
NUM_WORKERS = 10

write_lock = threading.Lock()
global_output_hashes = set()
hash_lock = threading.Lock()


# ─── TOKEN UTILS ─────────────────────────────────────────────

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_to_max_tokens(tokenizer, text, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def truncate_to_last_sentence(tokenizer, text, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text

    rough = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    sentence_enders = [".", "।", "!", "?", "॥"]

    last_pos = -1
    for ender in sentence_enders:
        pos = rough.rfind(ender)
        if pos > last_pos:
            last_pos = pos

    if last_pos > len(rough) // 3:
        return rough[: last_pos + 1].strip()

    last_comma = rough.rfind(",")
    if last_comma > len(rough) // 2:
        return rough[: last_comma + 1].strip()

    last_space = rough.rfind(" ")
    if last_space > len(rough) // 2:
        return rough[:last_space].strip()

    return rough


# ─── TEXT CLEANING ───────────────────────────────────────────

def remove_emojis(text):
    return "".join(ch for ch in text if not (0x1F300 <= ord(ch) <= 0x1FAFF))


def clean_text(text):
    text = re.sub(r'(.)\1{3,}', r'\1', text)

    cleaned_parts = []
    for part in re.split(r'(?<=[.।!?॥])\s*', text):
        part = part.strip()
        if not part:
            continue
        telugu_chars = sum(1 for ch in part if '\u0C00' <= ch <= '\u0C7F')
        total_chars = len(part.replace(" ", ""))
        if total_chars > 0 and telugu_chars / total_chars >= 0.4:
            cleaned_parts.append(part)

    text = " ".join(cleaned_parts)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ─── QUALITY CHECKS ─────────────────────────────────────────

def has_excessive_repetition(text):
    words = text.split()
    if len(words) < 10:
        return False

    for n in (2, 3, 4, 5):
        phrase_counts = {}
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        for phrase, count in phrase_counts.items():
            if count > 3:
                return True

    word_counts = {}
    for w in words:
        if len(w) > 3:
            word_counts[w] = word_counts.get(w, 0) + 1
    total = sum(word_counts.values())
    if total > 0:
        for word, count in word_counts.items():
            if count > 5 and count / total > 0.10:
                return True

    return False


def is_quality_output(text):
    if not text or len(text) < 50:
        return False

    telugu_chars = sum(1 for ch in text if '\u0C00' <= ch <= '\u0C7F')
    total_chars = len(text.replace(" ", ""))

    if total_chars == 0:
        return False
    if telugu_chars / total_chars < 0.5:
        return False
    if re.search(r'(.)\1{5,}', text):
        return False

    words = text.split()
    if len(words) < 10:
        return False
    if has_excessive_repetition(text):
        return False

    return True


def get_text_fingerprint(text):
    words = text.split()
    if len(words) < 20:
        return hash(text)
    key = " ".join(words[:10] + words[-10:])
    return hash(key)


def is_duplicate_output(text):
    fp = get_text_fingerprint(text)
    with hash_lock:
        if fp in global_output_hashes:
            return True
        global_output_hashes.add(fp)
    return False


# ─── PROMPTS ────────────────────────────────────────────────

def build_keyword_prompt(sample_text):
    """Step 1: Extract 4 movie-focused keywords from the review sample to understand its critical perspective."""
    return (
        f"క్రింది సినిమా సమీక్ష భాగం నుండి ఖచ్చితంగా 4 కీలక పదాలను తెలుగులో ఇవ్వండి. "
        f"ఈ కీలక పదాలు కథాంశాలు, నటన, సాంకేతిక అంశాలు (దర్శకత్వం, సంగీతం), "
        f"మరియు భావోద్వేగ ప్రభావాలకు సంబంధించినవిగా ఉండాలి. "
        f"వివరణలు లేదా ఇతర వాక్యాలు ఇవ్వకండి:\n{sample_text}"
    )


def build_article_prompt(keywords):
    """Step 2: Generate a full formal Telugu movie review from those 4 keywords."""
    return (
        f"ఈ 4 కీలక పదాల ఆధారంగా పూర్తిగా తెలుగు భాషలో ఒక సంపూర్ణ సినిమా సమీక్ష రాయండి. "
        f"సమీక్ష అధికారిక విమర్శక శైలిలో ఉండాలి. "
        f"కనీసం 30 వాక్యాలు ఉండాలి. "
        f"సమీక్షలో క్రింది అంశాలు స్పష్టంగా ఉండాలి:\n"
        f"- కథాంశం మరియు స్క్రీన్‌ప్లే విశ్లేషణ\n"
        f"- నటీనటుల అభినయ మూల్యాంకనం\n"
        f"- దర్శకత్వం, సినిమాటోగ్రఫీ మరియు సంగీత సమీక్ష\n"
        f"- చలనచిత్రం యొక్క మొత్తం భావోద్వేగ ప్రభావం\n"
        f"అదే పదాలను మళ్ళీ మళ్ళీ ఉపయోగించకండి. వైవిధ్యమైన వాక్య నిర్మాణం ఉపయోగించండి. "
        f"సూచనలు, వివరణలు, లేదా ఉపోద్ఘాతం ఇవ్వకండి. సమీక్ష మాత్రమే రాయండి.\n\n"
        f"కీలక పదాలు:\n{keywords}\n\nసినిమా సమీక్ష:"
    )


def build_extend_prompt(keywords, existing_text):
    """Step 3 (if needed): Extend an existing movie review with NEW critical content."""
    return (
        f"ఈ కీలక పదాల ఆధారంగా క్రింది తెలుగు సినిమా సమీక్షను కొనసాగించండి. "
        f"ఇప్పటికే చర్చించిన అంశాలను మళ్ళీ రాయకండి. "
        f"కొత్తగా క్రింది అంశాలను విశ్లేషించండి:\n"
        f"- ఇంకా చర్చించని నటన లేదా సాంకేతిక అంశాలు\n"
        f"- సన్నివేశాల లోతైన విమర్శనాత్మక విశ్లేషణ\n"
        f"- చలనచిత్రం యొక్క బలాలు మరియు బలహీనతలు\n"
        f"- మొత్తం సమీక్ష నిర్ణయం మరియు రేటింగ్ సూచన\n"
        f"అధికారిక విమర్శక శైలి కొనసాగాలి. "
        f"కనీసం 15 కొత్త వాక్యాలు రాయండి.\n\n"
        f"కీలక పదాలు: {keywords}\n\n"
        f"ఇప్పటివరకు:\n{existing_text}\n\nసమీక్ష కొనసాగింపు:"
    )

# ─── API CALL ────────────────────────────────────────────────

def generate_text(client, prompt, max_tokens, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            # Check for content moderation triggers (DashScope + DeepSeek patterns)
            if any(pat in error_msg.lower() for pat in [
                "data_inspection_failed", "inappropriate", "content_filter",
                "sensitive", "moderation", "safety"
            ]):
                return None
            if "429" in error_msg or "rate" in error_msg.lower() or "500" in error_msg:
                wait = (attempt + 1) * 3 + random.uniform(0, 2)
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return None


# ─── MAIN GENERATION ────────────────────────────────────────

def generate_article(client, tokenizer, sample_text):
    """
    Pipeline:
      1. Read input sample → extract 4 keywords
      2. Use keywords → generate full news article
      3. If too short → extend with keywords context (max 3 times)
      4. Trim to token window at sentence boundary
      5. Quality + duplicate check

    Returns: (article, token_count, status)
      status: "ok", "short", "filtered", "bad_quality", "duplicate"
    """

    # Truncate input to leave room
    sample_text = truncate_to_max_tokens(
        tokenizer, sample_text, MAX_MODEL_TOKENS - MAX_REVIEW_TOKENS - 100
    )

    # ── Step 1: Extract 4 keywords from the input sample ──
    kw_prompt = build_keyword_prompt(sample_text)
    keywords = generate_text(client, kw_prompt, max_tokens=64)
    if keywords is None:
        return None, 0, "filtered"

    keywords = truncate_to_max_tokens(tokenizer, keywords, 128)

    # ── Step 2: Generate article from those keywords ──
    article_prompt = build_article_prompt(keywords)
    article = generate_text(client, article_prompt, max_tokens=1500)
    if article is None:
        return None, 0, "filtered"

    article = remove_emojis(article)
    article = clean_text(article)

    token_count = count_tokens(tokenizer, article)

    # ── Step 3: If too short, extend using keywords + existing text ──
    extend_attempts = 0
    while token_count < MIN_REVIEW_TOKENS and extend_attempts < 3:
        extend_attempts += 1

        context = truncate_to_max_tokens(tokenizer, article, 600)
        ext_prompt = build_extend_prompt(keywords, context)
        continuation = generate_text(client, ext_prompt, max_tokens=1000)

        if continuation is None:
            break

        continuation = remove_emojis(continuation)
        continuation = clean_text(continuation)

        if not continuation or len(continuation.strip()) < 30:
            break

        candidate = article + " " + continuation
        candidate = " ".join(candidate.split())

        if has_excessive_repetition(candidate):
            break

        article = candidate
        token_count = count_tokens(tokenizer, article)
        time.sleep(SLEEP_SECONDS)

    # ── Step 4: Trim if over max — at sentence boundary ──
    if token_count > MAX_REVIEW_TOKENS:
        random_target = random.randint(MIN_REVIEW_TOKENS + 20, MAX_REVIEW_TOKENS)
        article = truncate_to_last_sentence(tokenizer, article, random_target)
        token_count = count_tokens(tokenizer, article)

    # ── Step 5: Ensure ends at a complete sentence ──
    sentence_enders = [".", "।", "!", "?", "॥"]
    if article and not any(article.rstrip().endswith(e) for e in sentence_enders):
        last_pos = -1
        for ender in sentence_enders:
            pos = article.rfind(ender)
            if pos > last_pos:
                last_pos = pos
        if last_pos > len(article) // 3:
            article = article[: last_pos + 1].strip()
            token_count = count_tokens(tokenizer, article)

    # ── Step 6: Token count check ──
    if token_count < MIN_REVIEW_TOKENS:
        return article, token_count, "short"

    # ── Step 7: Quality gate ──
    if not is_quality_output(article):
        return article, token_count, "bad_quality"

    # ── Step 8: Cross-sample duplicate gate ──
    if is_duplicate_output(article):
        return article, token_count, "duplicate"

    return article, token_count, "ok"


def process_single_sample(client, tokenizer, idx, sample_text):
    """Process one sample — never skips. Uses best output as fallback."""
    best_article = None
    best_token_count = 0
    filter_count = 0

    for attempt in range(1, MAX_ATTEMPTS_PER_SAMPLE + 1):
        article, token_count, status = generate_article(client, tokenizer, sample_text)

        # Perfect output
        if status == "ok":
            return idx, article, token_count

        # Track best partial output (short or bad_quality still has text)
        if article is not None and token_count > best_token_count:
            best_article = article
            best_token_count = token_count

        # Content filter — don't waste attempts, go to fallback after 3 filter hits
        if status == "filtered":
            filter_count += 1
            if filter_count >= 3:
                print(f"  Sample #{idx}: content filter hit 3 times, using fallback...")
                break

        if attempt % 5 == 0:
            print(f"  Sample #{idx}: attempt {attempt}/{MAX_ATTEMPTS_PER_SAMPLE} (best: {best_token_count} tok, status: {status})")

        time.sleep(0.3 + random.uniform(0, 0.2))

    # Fallback 1: Use best partial output if it has reasonable length
    if best_article and best_token_count >= MIN_REVIEW_TOKENS:
        print(f"  Sample #{idx}: using best output ({best_token_count} tok)")
        return idx, best_article, best_token_count

    # Fallback 2: Generate a generic article using only the keywords (no input text to trigger filter)
    print(f"  Sample #{idx}: trying keyword-only fallback...")
    safe_keywords = None
    try:
        # Extract keywords from just the first 200 chars to reduce filter triggers
        short_text = sample_text[:200]
        kw_prompt = build_keyword_prompt(short_text)
        safe_keywords = generate_text(client, kw_prompt, max_tokens=64)
    except Exception:
        pass

    if safe_keywords:
        for _ in range(3):
            fb_prompt = build_article_prompt(safe_keywords)
            fallback = generate_text(client, fb_prompt, max_tokens=1500)
            if fallback:
                fallback = remove_emojis(fallback)
                fallback = clean_text(fallback)
                tc = count_tokens(tokenizer, fallback)
                if tc >= MIN_REVIEW_TOKENS and not has_excessive_repetition(fallback):
                    print(f"  Sample #{idx}: keyword fallback succeeded ({tc} tok)")
                    return idx, fallback, tc
                elif tc > best_token_count:
                    best_article = fallback
                    best_token_count = tc
            time.sleep(0.3)

    # Fallback 3: Completely generic Telugu news article
    print(f"  Sample #{idx}: trying generic fallback...")
    generic_prompt = (
    "పూర్తిగా తెలుగు భాషలో ఒక సంపూర్ణ సినిమా సమీక్షను మాత్రమే రాయండి. "
    "కనీసం 30 వాక్యాలు ఉండాలి. సినిమా సమీక్ష మాత్రమే రాయండి."
    )
    for _ in range(3):
        fallback = generate_text(client, generic_prompt, max_tokens=1500)
        if fallback:
            fallback = remove_emojis(fallback)
            fallback = clean_text(fallback)
            tc = count_tokens(tokenizer, fallback)
            if tc >= MIN_REVIEW_TOKENS:
                print(f"  Sample #{idx}: generic fallback succeeded ({tc} tok)")
                return idx, fallback, tc
            elif tc > best_token_count:
                best_article = fallback
                best_token_count = tc
        time.sleep(0.3)

    # Return whatever we have
    if best_article:
        print(f"  Sample #{idx}: returning best available ({best_token_count} tok)")
        return idx, best_article, best_token_count

    print(f"  Sample #{idx}: WARNING - using placeholder")
    return idx, "వార్తా కథనం అందుబాటులో లేదు.", 5


# ─── FILE IO ────────────────────────────────────────────────

def load_input_data(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  Warning: Skipping invalid JSON at line {line_num}")
        print(f"Loaded {len(data)} records from JSONL file")
        return data

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "records", "texts", "samples", "reviews"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                if not isinstance(data, list):
                    data = [data]
        print(f"Loaded {len(data)} records from JSON file")
        return data

    else:
        raise ValueError(f"Unsupported file format: {ext} (use .json or .jsonl)")


def atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ─── MAIN ────────────────────────────────────────────────────

def main():
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise SystemExit(f"Missing {API_KEY_ENV_VAR} environment variable")

    print(f"Model: {MODEL_NAME} | Workers: {NUM_WORKERS} | Temp: {TEMPERATURE}")
    print(f"Token window: {MIN_REVIEW_TOKENS}–{MAX_REVIEW_TOKENS}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Resume support
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        completed_ids = {r["id"] for r in output_data.get("reviews", [])}
        print(f"Resuming: {len(completed_ids)} already done")

        for r in output_data.get("reviews", []):
            global_output_hashes.add(get_text_fingerprint(r.get("review", "")))
    else:
        output_data = {"reviews": []}
        completed_ids = set()
        atomic_write(OUTPUT_JSON_PATH, output_data)

    print(f"Output: {OUTPUT_JSON_PATH}")

    input_json = load_input_data(INPUT_JSON_PATH)

    # Build ordered list of pending samples
    pending = []
    for idx, obj in enumerate(input_json, start=1):
        sample_id = f"{idx:04d}"
        if sample_id in completed_ids:
            continue
        sample_text = obj.get("text", "").strip()
        if sample_text:
            pending.append((idx, sample_text))

    total = len(pending)
    print(f"Pending: {total} samples\n")

    if total == 0:
        print("All done!")
        return

    start_time = time.time()
    completed = 0

    # Process in batches of NUM_WORKERS to maintain order
    for batch_start in range(0, total, NUM_WORKERS):
        batch = pending[batch_start: batch_start + NUM_WORKERS]
        batch_results = {}

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_single_sample, client, tokenizer, idx, text): idx
                for idx, text in batch
            }

            for future in as_completed(futures):
                idx, article, token_count = future.result()
                batch_results[idx] = (article, token_count)

        # Save batch results in order
        for idx, text in batch:
            article, token_count = batch_results[idx]

            completed += 1

            output_data["reviews"].append({
                "id": f"{idx:04d}",
                "review": article,
                "review_token_count": token_count,
            })

            elapsed = time.time() - start_time
            rate = completed / elapsed * 3600 if elapsed > 0 else 0
            eta = (total - completed) / (completed / elapsed) / 60 if completed > 0 else 0

            print(
                f"[{completed}/{total}] OK #{idx} | "
                f"{token_count} tok | "
                f"{rate:.0f}/hr | "
                f"ETA {eta:.1f}m"
            )

        # Save after each batch
        atomic_write(OUTPUT_JSON_PATH, output_data)

    elapsed = time.time() - start_time
    print(f"\nFinished: {completed} saved in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()