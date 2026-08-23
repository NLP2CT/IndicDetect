import json
import re
import numpy as np

INPUT_FILE   : str   = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Tamil/White_Space_Attack/Test.json"
ADD_RATE     : float = 0.20
LABEL_KEY    : str   = "label"
TEXT_KEY     : str   = "text"
TARGET_LABEL : str   = "LLM"
SEED         : int   = 42

rng = np.random.default_rng(SEED)

_TOKEN_RE: re.Pattern = re.compile(r"[\u0B80-\u0BFF]+|\S+")


def apply_whitespace_addition(text: str) -> tuple[str, int]:
    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < 2:
        return text, 0

    inter_token_positions = list(range(len(tokens) - 1))
    n_add  = max(1, int(len(inter_token_positions) * ADD_RATE))
    chosen = set(rng.choice(inter_token_positions, size=n_add, replace=True))

    gaps   = re.split(r"[\u0B80-\u0BFF]+|\S+", text)
    result = gaps[0] if gaps else ""
    for i, tok in enumerate(tokens):
        result += tok
        if i < len(tokens) - 1:
            space = gaps[i + 1] if (i + 1) < len(gaps) else " "
            if i in chosen:
                result += " " + space
            else:
                result += space
    if len(gaps) > len(tokens):
        result += gaps[-1]

    return result, len(chosen)


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    modified, skipped, total_added = 0, 0, 0

    for idx, entry in enumerate(data, 1):
        print(f"\r   Processing {idx}/{len(data)} ...", end="", flush=True)
        if not isinstance(entry, dict):
            continue
        if entry.get(LABEL_KEY) == TARGET_LABEL:
            text = entry.get(TEXT_KEY, "")
            if isinstance(text, str):
                new_text, added = apply_whitespace_addition(text)
                entry[TEXT_KEY] = new_text
                total_added += added
                if added > 0:
                    modified += 1
                else:
                    skipped += 1

    with open(INPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"\r   Done.{' ' * 30}")
    print(f"Saved -> {INPUT_FILE}")
    print(f"LLM modified    : {modified}")
    print(f"LLM untouched   : {skipped}")
    print(f"Total spaces added: {total_added}")


if __name__ == "__main__":
    main()