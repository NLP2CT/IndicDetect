import json
import os
import re
import random
from typing import Union

DELETE_RATE: float = 0.50
SEED: Union[int, None] = None
TEXT_KEY: str = "text"
LABEL_KEY: str = "label"
TARGET_LABEL: str = "LLM"

TELUGU_FUNCTION_WORDS: list[str] = [
    "మరియు", "కానీ", "లేదా", "అలాగే",
    "కూడా", "మాత్రమే", "అయినా", "సైతం", "ఇంకా",
    "గురించి", "కోసం", "వల్ల", "వంటి", "వరకు",
    "తర్వాత", "మధ్య",
    "అంటే", "ఎందుకంటే", "కాబట్టి", "అయితే",
    "అందువల్ల", "కనుక", "తద్వారా", "అందుకే",
    "కావున", "అందుకు",
    "అంతేకాక", "అంతేకాదు", "అయినప్పటికీ", "ఏదేమైనా",
    "అయినప్పటికిని", "పైగా", "అంతట",
    "నుండి", "వైపు", "లోపల", "బయట",
    "పైన", "కింద", "దగ్గర", "మీద",
    "లోనే", "వెనుక", "ముందు",
]

_SEP_BEFORE: str = r'(?<![^\s।॥,.\-:;!?(\"\'—–])'
_SEP_AFTER: str  = r'(?![^\s।॥,.\-:;!?)\"\'—–])'

_COMBINED_PATTERN: re.Pattern = re.compile(
    _SEP_BEFORE
    + "("
    + "|".join(re.escape(w) for w in TELUGU_FUNCTION_WORDS)
    + ")"
    + _SEP_AFTER
)

_WORD_PATTERN: re.Pattern = re.compile(r'[\u0C00-\u0C7F]+')


def find_function_word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _COMBINED_PATTERN.finditer(text)]


def delete_function_words(text: str, delete_rate: float, rng: random.Random) -> str:
    if not isinstance(text, str) or not text:
        return text

    spans = find_function_word_spans(text)

    if spans:
        n_delete = max(1, int(round(len(spans) * delete_rate))) if delete_rate > 0 else 0
        n_delete = min(n_delete, len(spans))
        chosen: set[int] = set(rng.sample(range(len(spans)), n_delete))

        parts: list[str] = []
        prev_end: int = 0

        for idx, (start, end) in enumerate(spans):
            parts.append(text[prev_end:start])
            if idx in chosen:
                skip_end = end
                if skip_end < len(text) and text[skip_end] == " ":
                    skip_end += 1
                prev_end = skip_end
            else:
                parts.append(text[start:end])
                prev_end = end

        parts.append(text[prev_end:])
        return re.sub(r"  +", " ", "".join(parts)).strip()

    else:
        telugu_words = [m for m in _WORD_PATTERN.finditer(text)]
        if not telugu_words:
            return text

        n_delete = max(1, int(round(len(telugu_words) * delete_rate)))
        n_delete = min(n_delete, len(telugu_words))
        chosen_words = set(rng.sample(range(len(telugu_words)), n_delete))

        parts: list[str] = []
        prev_end: int = 0

        for idx, m in enumerate(telugu_words):
            parts.append(text[prev_end:m.start()])
            if idx in chosen_words:
                skip_end = m.end()
                if skip_end < len(text) and text[skip_end] == " ":
                    skip_end += 1
                prev_end = skip_end
            else:
                parts.append(m.group())
                prev_end = m.end()

        parts.append(text[prev_end:])
        return re.sub(r"  +", " ", "".join(parts)).strip()


def process_json_file(
    file_path: str,
    text_key: str = TEXT_KEY,
    label_key: str = LABEL_KEY,
    target_label: str = TARGET_LABEL,
    delete_rate: float = DELETE_RATE,
    seed: Union[int, None] = SEED,
) -> None:
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"❌  Failed to load JSON: {file_path}\n    {exc}")
        return

    if not isinstance(data, list):
        print(f"❌  JSON root must be a list. Skipping: {file_path}")
        return

    rng = random.Random(seed) if seed is not None else random.Random()

    n_skipped: int = 0
    n_modified_func: int = 0
    n_modified_fallback: int = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue

        if entry.get(label_key) != target_label:
            n_skipped += 1
            continue

        if text_key not in entry or not isinstance(entry[text_key], str):
            continue

        original_text = entry[text_key]
        has_func_words = bool(find_function_word_spans(original_text))

        new_text = delete_function_words(original_text, delete_rate, rng)
        entry[text_key] = new_text

        if has_func_words:
            n_modified_func += 1
        else:
            n_modified_fallback += 1

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(
            f"✅  {file_path}\n"
            f"    Modified via function words: {n_modified_func}\n"
            f"    Modified via fallback:       {n_modified_fallback}\n"
            f"    Non-LLM (skipped):           {n_skipped}"
        )
    except Exception as exc:
        print(f"❌  Failed to write updated JSON: {file_path}\n    {exc}")


if __name__ == "__main__":
    SEED = None

    json_file = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Telugu/Petrubation_Attacks/Test.json"

    process_json_file(json_file, text_key=TEXT_KEY, label_key=LABEL_KEY, target_label=TARGET_LABEL, delete_rate=DELETE_RATE, seed=SEED)