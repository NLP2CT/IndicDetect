import json
import os
import re
import random
from typing import Union

INSERT_RATE: float = 0.50
SEED: Union[int, None] = None
TEXT_KEY: str = "text"
LABEL_KEY: str = "label"
TARGET_LABEL: str = "LLM"

_SENT_END: re.Pattern = re.compile(r'([।॥.?!]+)\s*')


def get_inter_sentence_positions(text: str) -> list[int]:
    positions = []
    for m in _SENT_END.finditer(text):
        end = m.end()
        if end < len(text):
            positions.append(end)
    return positions


def insert_paragraphs(text: str, insert_rate: float, rng: random.Random) -> str:
    if not isinstance(text, str) or not text:
        return text

    positions = get_inter_sentence_positions(text)
    if not positions:
        return text

    n_insert = max(1, int(round(len(positions) * insert_rate))) if insert_rate > 0 else 0
    n_insert = min(n_insert, len(positions))

    if n_insert == 0:
        return text

    chosen = set(rng.sample(range(len(positions)), n_insert))

    result = list(text)
    for idx in sorted(chosen, reverse=True):
        result.insert(positions[idx], '\n\n')

    return ''.join(result)


def process_json_file(
    file_path: str,
    text_key: str = TEXT_KEY,
    label_key: str = LABEL_KEY,
    target_label: str = TARGET_LABEL,
    insert_rate: float = INSERT_RATE,
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
    n_modified: int = 0
    n_single_sentence: int = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue

        if entry.get(label_key) != target_label:
            n_skipped += 1
            continue

        if text_key not in entry or not isinstance(entry[text_key], str):
            continue

        original_text = entry[text_key]
        positions = get_inter_sentence_positions(original_text)

        if not positions:
            n_single_sentence += 1
            continue

        new_text = insert_paragraphs(original_text, insert_rate, rng)
        entry[text_key] = new_text
        n_modified += 1

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(
            f"✅  {file_path}\n"
            f"    Modified (LLM):       {n_modified}\n"
            f"    Non-LLM (skipped):    {n_skipped}\n"
            f"    Single sentence:      {n_single_sentence}"
        )
    except Exception as exc:
        print(f"❌  Failed to write updated JSON: {file_path}\n    {exc}")


if __name__ == "__main__":
    SEED = None

    json_file = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Telugu/Insert_Paragraph_Attacks/Test.json"

    process_json_file(json_file, text_key=TEXT_KEY, label_key=LABEL_KEY, target_label=TARGET_LABEL, insert_rate=INSERT_RATE, seed=SEED)