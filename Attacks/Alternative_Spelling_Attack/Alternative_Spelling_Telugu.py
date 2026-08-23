import json
import os
import random
import re

INPUT_FILE : str = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Telugu/Alternative_Spelling_Attacks/Test.json"
DICT_FILE  : str = r"/Users/user/Desktop/IndicDetect/Codes/Attacks/Alternative_Spelling_Attack/telugu_alternative_spelling.json"

LABEL_KEY    : str = "label"
TEXT_KEY     : str = "text"
TARGET_LABEL : str = "LLM"

SEED: int = 42
random.seed(SEED)

_TELUGU_RE: re.Pattern = re.compile(r"[\u0C00-\u0C7F]+")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_alternative_spelling(text: str, alt_dict: dict) -> tuple[str, int]:
    replacements = 0

    def replace_word(match: re.Match) -> str:
        nonlocal replacements
        word = match.group(0)
        alternatives = alt_dict.get(word)
        if alternatives:
            replacements += 1
            return random.choice(alternatives)
        return word

    return _TELUGU_RE.sub(replace_word, text), replacements


def main() -> None:
    data     : list = load_json(INPUT_FILE)
    alt_dict : dict = load_json(DICT_FILE)

    modified, skipped, total_swaps = 0, 0, 0

    for idx, entry in enumerate(data, 1):
        print(f"\r   Processing {idx}/{len(data)} ...", end="", flush=True)
        if not isinstance(entry, dict):
            continue
        if entry.get(LABEL_KEY) == TARGET_LABEL:
            text = entry.get(TEXT_KEY, "")
            if isinstance(text, str):
                new_text, swaps = apply_alternative_spelling(text, alt_dict)
                entry[TEXT_KEY] = new_text
                total_swaps += swaps
                modified += 1 if swaps > 0 else 0
                skipped  += 1 if swaps == 0 else 0

    with open(INPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"\r   Done.{' ' * 30}")
    print(f"Saved -> {INPUT_FILE}")
    print(f"LLM modified  : {modified}")
    print(f"LLM untouched : {skipped}")
    print(f"Total swaps   : {total_swaps}")


if __name__ == "__main__":
    main()