import json, time, random, re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator

FILE_TO_PROCESS = "/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Hindi/Paraphase_Attacks/Test.json"

SRC_LANG = "hi"
PIVOT_LANG = "zh-CN"
MAX_CHARS_PER_CHUNK = 4500
RETRIES = 7
MAX_WORKERS = 3
SAVE_EVERY = 50


def translate_once(txt, source, target):
    if not txt.strip():
        return txt
    for attempt in range(RETRIES):
        try:
            result = GoogleTranslator(source=source, target=target).translate(txt)
            if result and result.strip():
                return result
            raise ValueError("Empty translation returned")
        except Exception as e:
            if attempt == RETRIES - 1:
                return txt
            wait = 3 * (attempt + 1)
            time.sleep(wait)


def split_sentences(txt):
    if not txt.strip():
        return []
    txt = re.sub(r"\s+", " ", txt)
    parts = re.split(r"(?<=[\.!?।])\s+", txt)
    return [p.strip() for p in parts if p.strip()] or [txt.strip()]


def chunk_text(txt, limit=MAX_CHARS_PER_CHUNK):
    if len(txt) <= limit:
        return [txt]
    out, start = [], 0
    while start < len(txt):
        end = min(len(txt), start + limit)
        if end == len(txt):
            out.append(txt[start:end])
            break
        split = txt.rfind(" ", start, end)
        if split <= start:
            split = end
        out.append(txt[start:split])
        start = split if split == end else split + 1
    return out


def pipeline(sentence):
    try:
        zh = " ".join(translate_once(ch, SRC_LANG, PIVOT_LANG) for ch in chunk_text(sentence))
        hi = " ".join(translate_once(ch, PIVOT_LANG, SRC_LANG) for ch in chunk_text(zh))
        return hi
    except Exception:
        return sentence


def is_llm_sample(obj):
    return str(obj.get("label", "")).strip().upper() == "LLM"


def process_single(idx_obj):
    idx, obj = idx_obj
    if is_llm_sample(obj) and "text" in obj:
        sents = split_sentences(obj["text"])
        obj["text"] = " ".join(pipeline(s) for s in sents)
        return idx, obj, "processed"
    return idx, obj, "skipped"


def process(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"❌ File not found: {p}")
        return

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ Expected a JSON array")
        return

    processed, skipped, errors = 0, 0, 0
    total = len(data)

    batch_start = 0
    while batch_start < total:
        batch_end = min(batch_start + SAVE_EVERY, total)
        batch = [(i, data[i]) for i in range(batch_start, batch_end)]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single, item): item[0] for item in batch}
            with tqdm(total=len(futures), desc=f"{p.name} [{batch_start+1}-{batch_end}]",
                      initial=0, leave=False) as pbar:
                for future in as_completed(futures):
                    try:
                        idx, obj, status = future.result()
                        data[idx] = obj
                        if status == "processed":
                            processed += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        print(f"  [Error] {e}")
                        errors += 1
                    pbar.update(1)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  💾 Saved {batch_end}/{total} | {processed} translated | {skipped} skipped | {errors} errors")

        batch_start = batch_end

    print(f"\n  ✅ Done {p.name} | {processed} LLM back-translated | "
          f"{skipped} Human skipped | {errors} errors\n")


if __name__ == "__main__":
    print("🚀 Hindi ↔ Chinese Back-Translation (deep-translator)\n")
    print("=" * 70)
    print("Processing:", FILE_TO_PROCESS)
    print(f"Workers: {MAX_WORKERS}")
    print("=" * 70)
    process(FILE_TO_PROCESS)
    print("🎉 Done.")