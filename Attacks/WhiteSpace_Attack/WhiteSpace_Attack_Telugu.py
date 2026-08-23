import json
import os
import random
from collections import Counter
from typing import Union

ADD_RATE: float = 0.20
SEED: Union[int, None] = None
TEXT_KEY: str = "text"
LABEL_KEY: str = "label"
TARGET_LABEL: str = "LLM"


def whitespace_addition(text: str, add_rate: float, rng: random.Random) -> str:
    if not isinstance(text, str) or not text:
        return text

    space_positions = [i for i, ch in enumerate(text) if ch == ' ']

    if not space_positions:
        return text

    n_to_add = max(1, int(round(len(space_positions) * add_rate))) if add_rate > 0 else 0

    if n_to_add == 0:
        return text

    chosen = [rng.choice(space_positions) for _ in range(n_to_add)]
    extra = Counter(chosen)

    result = []
    for i, ch in enumerate(text):
        result.append(ch)
        if i in extra:
            result.append(' ' * extra[i])

    return ''.join(result)


def process_json_file(
    file_path: str,
    text_key: str = TEXT_KEY,
    label_key: str = LABEL_KEY,
    target_label: str = TARGET_LABEL,
    add_rate: float = ADD_RATE,
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
    n_no_spaces: int = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue

        if entry.get(label_key) != target_label:
            n_skipped += 1
            continue

        if text_key not in entry or not isinstance(entry[text_key], str):
            continue

        original_text = entry[text_key]

        if ' ' not in original_text:
            n_no_spaces += 1
            continue

        new_text = whitespace_addition(original_text, add_rate, rng)
        entry[text_key] = new_text
        n_modified += 1

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(
            f"✅  {file_path}\n"
            f"    Modified (LLM):    {n_modified}\n"
            f"    Non-LLM (skipped): {n_skipped}\n"
            f"    No spaces found:   {n_no_spaces}"
        )
    except Exception as exc:
        print(f"❌  Failed to write updated JSON: {file_path}\n    {exc}")


if __name__ == "__main__":
    SEED = None

    json_file = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Telugu/White_Space_Attacks/Test.json"

    process_json_file(json_file, text_key=TEXT_KEY, label_key=LABEL_KEY, target_label=TARGET_LABEL, add_rate=ADD_RATE, seed=SEED)