import json
import re
import numpy as np

INPUT_FILE    : str   = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Tamil/Insert_paragraph_Attack/Test.json"
INSERT_RATE   : float = 0.50
LABEL_KEY     : str   = "label"
TEXT_KEY      : str   = "text"
TARGET_LABEL  : str   = "LLM"
SEED          : int   = 42

rng = np.random.default_rng(SEED)

_SENT_RE: re.Pattern = re.compile(r"([.!?।]+)\s*")


def apply_insert_paragraph(text: str) -> tuple[str, int]:
    parts    = _SENT_RE.split(text)
    boundary_indices = []

    i = 1
    while i < len(parts) - 1:
        if _SENT_RE.fullmatch(parts[i]):
            boundary_indices.append(i)
        i += 1

    if not boundary_indices:
        return text, 0

    n_insert = max(1, int(len(boundary_indices) * INSERT_RATE))
    chosen   = set(rng.choice(boundary_indices, size=n_insert, replace=False))

    for pos in sorted(chosen, reverse=True):
        parts[pos] = parts[pos] + "\n\n"

    return "".join(parts), len(chosen)


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    modified, skipped, total_inserts = 0, 0, 0

    for idx, entry in enumerate(data, 1):
        print(f"\r   Processing {idx}/{len(data)} ...", end="", flush=True)
        if not isinstance(entry, dict):
            continue
        if entry.get(LABEL_KEY) == TARGET_LABEL:
            text = entry.get(TEXT_KEY, "")
            if isinstance(text, str):
                new_text, inserts = apply_insert_paragraph(text)
                entry[TEXT_KEY] = new_text
                total_inserts += inserts
                if inserts > 0:
                    modified += 1
                else:
                    skipped += 1

    with open(INPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"\r   Done.{' ' * 30}")
    print(f"Saved -> {INPUT_FILE}")
    print(f"LLM modified      : {modified}")
    print(f"LLM untouched     : {skipped}")
    print(f"Total inserts     : {total_inserts}")


if __name__ == "__main__":
    main()