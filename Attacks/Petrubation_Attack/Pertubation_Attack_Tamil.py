import json
import re
import numpy as np

INPUT_FILE   : str   = r"/Users/user/Desktop/IndicDetect/Benchmark_Data/Task_Data/Adverisal_Attacks/Tamil/Petrubation_Attack/Test.json"
DELETE_RATE  : float = 0.50
LABEL_KEY    : str   = "label"
TEXT_KEY     : str   = "text"
TARGET_LABEL : str   = "LLM"
SEED         : int   = 42

rng = np.random.default_rng(SEED)

TAMIL_FUNCTION_WORDS = {
    "மற்றும்", "அல்லது", "ஆனால்", "எனவே", "இருந்தாலும்", "ஏனென்றால்",
    "அதனால்", "அதாவது", "மேலும்", "கூட", "தான்", "இல்", "இல்லை",
    "உள்ள", "என்று", "என", "ஆக", "இந்த", "அந்த", "அவர்",
    "அவள்", "அவன்", "அது", "இது", "அவை", "இவை", "நாம்",
    "நான்", "நீ", "நீங்கள்", "அவர்கள்", "இவர்கள்", "எல்லாம்",
    "சில", "பல", "ஒரு", "ஒரு", "எந்த", "எந்தவொரு",
    "முன்பு", "பின்பு", "இப்போது", "இனி", "வரை", "தவிர",
    "போல", "மட்டும்", "அல்ல", "இல்லாமல்", "என்னை", "உங்கள்",
    "அதை", "இதை", "அதில்", "இதில்", "அங்கு", "இங்கு",
}

_WORD_RE: re.Pattern = re.compile(r"[\u0B80-\u0BFF]+|\S+")


def apply_function_word_deletion(text: str) -> tuple[str, int]:
    tokens = _WORD_RE.findall(text)
    func_indices = [i for i, t in enumerate(tokens) if t in TAMIL_FUNCTION_WORDS]

    if not func_indices:
        return text, 0

    n_delete = max(1, int(len(func_indices) * DELETE_RATE))
    chosen   = rng.choice(func_indices, size=n_delete, replace=False)
    delete_set = set(chosen)

    result = [t for i, t in enumerate(tokens) if i not in delete_set]

    original_spaces = re.split(r"[\u0B80-\u0BFF]+|\S+", text)
    rebuilt = ""
    si = 0
    for i, tok in enumerate(tokens):
        if si < len(original_spaces):
            rebuilt += original_spaces[si]
            si += 1
        if i not in delete_set:
            rebuilt += tok
    if si < len(original_spaces):
        rebuilt += original_spaces[si]

    return rebuilt, n_delete


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    modified, skipped, no_func, total_deleted = 0, 0, 0, 0

    for idx, entry in enumerate(data, 1):
        print(f"\r   Processing {idx}/{len(data)} ...", end="", flush=True)
        if not isinstance(entry, dict):
            continue
        if entry.get(LABEL_KEY) == TARGET_LABEL:
            text = entry.get(TEXT_KEY, "")
            if isinstance(text, str):
                new_text, deleted = apply_function_word_deletion(text)
                entry[TEXT_KEY] = new_text
                total_deleted += deleted
                if deleted > 0:
                    modified += 1
                else:
                    no_func += 1
        else:
            skipped += 1

    with open(INPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"\r   Done.{' ' * 30}")
    print(f"Saved -> {INPUT_FILE}")
    print(f"LLM modified       : {modified}")
    print(f"LLM no func words  : {no_func}")
    print(f"Non-LLM skipped    : {skipped}")
    print(f"Total words deleted: {total_deleted}")


if __name__ == "__main__":
    main()