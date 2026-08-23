import logging
import random
import numpy as np
import torch
import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
import contextlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VALID_LABELS = {"Human", "LLM"}
MIN_THRESHOLD_SAMPLES = 20   # minimum per class for a meaningful threshold
MAX_TOKENS = 512             # guard against context window overflow


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_and_rename_sentence(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "sentence" in item:
            item["text"] = item.pop("sentence")
        if "text" not in item:
            raise ValueError(f"Missing 'text' or 'sentence' key in entry: {item}")
        if item.get("label") not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{item.get('label')}' in entry. "
                f"Expected one of {VALID_LABELS}"
            )
    return data


def check_data_overlap(threshold_data, test_data, test_name):
    train_texts = {item["text"] for item in threshold_data}
    test_texts  = {item["text"] for item in test_data}
    overlap = train_texts & test_texts
    if overlap:
        logging.warning(
            f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping texts between "
            f"threshold data and test file '{test_name}'. "
            f"Removing overlapping entries from test set."
        )
        test_data = [item for item in test_data if item["text"] not in overlap]
        logging.info(f"Test set size after removing overlaps: {len(test_data)}")
    return test_data


# ─────────────────────────────────────────────
# BF16 autocast context (inference only)
# ─────────────────────────────────────────────

@contextlib.contextmanager
def bf16_autocast(enable=True):
    if enable and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


# ─────────────────────────────────────────────
# Core discrepancy function
# FIX #3: normalize by sqrt(n_tokens) so score is length-independent
# FIX #5: caller must pass FP32 logits — enforced via assertion
# ─────────────────────────────────────────────

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    """
    Compute the FastDetectGPT sampling discrepancy score.

    Parameters
    ----------
    logits_ref   : (1, T, V) float32 tensor — reference model logits
    logits_score : (1, T, V) float32 tensor — scoring model logits
    labels       : (1, T)   int64  tensor  — token ids (shifted targets)

    Returns
    -------
    float — scalar discrepancy score (length-normalised)
    """
    # FIX #5: assert FP32 so BF16 cancellation can't corrupt variance
    assert logits_ref.dtype == torch.float32,   "logits_ref must be float32"
    assert logits_score.dtype == torch.float32, "logits_score must be float32"

    if logits_ref.shape[0] != 1 or logits_score.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Batch size must be 1 for all inputs.")

    # Align vocabulary sizes if models differ slightly
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size  = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref  = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels

    lprobs_score   = torch.log_softmax(logits_score, dim=-1)
    probs_ref      = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref       = (probs_ref * lprobs_score).sum(dim=-1)
    # Variance: E[X²] − (E[X])²  — computed in FP32 to avoid cancellation
    var_ref        = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    # Clamp variance to avoid negative values from floating point noise
    var_ref = var_ref.clamp(min=0.0)

    denom = var_ref.sum(dim=-1).sqrt()
    denom = torch.where(
        denom == 0,
        torch.tensor(1e-12, device=denom.device, dtype=denom.dtype),
        denom,
    )

    numerator = log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)

    # FIX #3: length normalisation — divide by sqrt(n_tokens)
    n_tokens    = labels.shape[-1]
    sqrt_tokens = max(n_tokens ** 0.5, 1.0)

    discrepancy = (numerator / denom) / sqrt_tokens
    return discrepancy.mean().item()


# ─────────────────────────────────────────────
# Per-text scoring
# FIX #4: truncation guard
# FIX #5: upcast logits to FP32 before math
# FIX #7: GPU memory cleanup
# ─────────────────────────────────────────────

def get_text_crit(text, args, model_config):
    """Score a single text and return the FastDetectGPT discrepancy value."""

    # FIX #4: enforce max_length to avoid silent context-window overflow
    tokenize_kwargs = dict(
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=MAX_TOKENS,
        truncation=True,
    )

    tokenized = model_config["scoring_tokenizer"](text, **tokenize_kwargs)
    tokenized = {
        k: v.to(args.DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in tokenized.items()
    }
    labels = tokenized["input_ids"][:, 1:]   # shifted targets

    try:
        with torch.no_grad():
            # Inference in BF16 for speed/memory …
            with bf16_autocast(args.use_bf16):
                logits_score = model_config["scoring_model"](**tokenized).logits[:, :-1]

            # FIX #5: … but upcast to FP32 IMMEDIATELY before any numerical ops
            logits_score = logits_score.float()

            if args.reference_model == args.scoring_model:
                logits_ref = logits_score
            else:
                tokenized_ref = model_config["reference_tokenizer"](text, **tokenize_kwargs)
                tokenized_ref = {
                    k: v.to(args.DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_ref.items()
                }

                if not torch.all(tokenized_ref["input_ids"][:, 1:] == labels):
                    raise RuntimeError(
                        "Tokenizer mismatch between scoring and reference models. "
                        "Ensure both models share the same vocabulary."
                    )

                with bf16_autocast(args.use_bf16):
                    logits_ref = model_config["reference_model"](**tokenized_ref).logits[:, :-1]

                # FIX #5: upcast reference logits too
                logits_ref = logits_ref.float()

            return get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

    finally:
        # FIX #7: release GPU tensors between samples to prevent OOM on long runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────

def compute_confusion(y_true, y_pred):
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if   t == 0 and p == 0: tn += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        elif t == 1 and p == 1: tp += 1
    return tn, fp, fn, tp


# ─────────────────────────────────────────────
# Threshold calibration
# FIX #1: detect + correct score direction inversion
# FIX #6: log mean scores for convention sanity-check
# FIX #9: minimum sample guard
# ─────────────────────────────────────────────

def calibrate_threshold(human_scores, llm_scores, label=""):
    """
    Derive the optimal decision threshold from labelled scores.

    Returns
    -------
    optimal_threshold : float
    score_direction   : +1 or -1  (+1 → higher score = LLM)
    roc_auc           : float     (after direction correction)
    """
    tag = f"[{label}] " if label else ""

    # FIX #9: minimum sample guard
    if len(human_scores) < MIN_THRESHOLD_SAMPLES:
        logging.warning(
            f"{tag}Only {len(human_scores)} Human samples for threshold "
            f"calibration (minimum recommended: {MIN_THRESHOLD_SAMPLES}). "
            f"Threshold may be unreliable."
        )
    if len(llm_scores) < MIN_THRESHOLD_SAMPLES:
        logging.warning(
            f"{tag}Only {len(llm_scores)} LLM samples for threshold "
            f"calibration (minimum recommended: {MIN_THRESHOLD_SAMPLES}). "
            f"Threshold may be unreliable."
        )

    # FIX #6: log mean scores so convention issues surface immediately
    mean_human = float(np.mean(human_scores))
    mean_llm   = float(np.mean(llm_scores))
    logging.info(
        f"{tag}Mean score — Human: {mean_human:.4f}  LLM: {mean_llm:.4f}"
    )

    roc_auc, optimal_threshold, *_ = get_roc_metrics(human_scores, llm_scores)

    # FIX #1: detect score direction inversion
    # FastDetectGPT canonical direction: higher discrepancy → more likely LLM.
    # If ROC-AUC < 0.5 the direction is inverted; flip both to correct.
    score_direction = 1
    if roc_auc < 0.5:
        logging.warning(
            f"{tag}ROC-AUC = {roc_auc:.4f} < 0.5 — score direction is INVERTED. "
            f"Flipping decision direction automatically."
        )
        score_direction  = -1
        roc_auc          = 1.0 - roc_auc
        optimal_threshold = -optimal_threshold   # mirror the threshold

    logging.info(
        f"{tag}Optimal threshold: {optimal_threshold:.4f}  "
        f"Direction: {'higher→LLM' if score_direction == 1 else 'lower→LLM'}  "
        f"Train AUC: {roc_auc:.4f}"
    )
    return optimal_threshold, score_direction, roc_auc


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def experiment(args):
    set_all_seeds(args.seed)

    # FIX #2: warn when reference == scoring (degenerate mode)
    if args.reference_model == args.scoring_model:
        logging.warning(
            "Reference model and scoring model are IDENTICAL. "
            "FastDetectGPT requires distinct models for a meaningful discrepancy signal. "
            "In this mode the numerator collapses to log-likelihood minus its own "
            "expectation — results will be substantially degraded. "
            "Consider passing different --reference_model and --scoring_model."
        )

    logging.info(f"Loading reference model : {args.reference_model}")
    logging.info(f"Loading scoring model   : {args.scoring_model}")

    load_kwargs = {"trust_remote_code": True}
    if args.use_bf16:
        load_kwargs["torch_dtype"]      = torch.bfloat16
        load_kwargs["device_map"]       = "auto"
        load_kwargs["low_cpu_mem_usage"] = True

    reference_tokenizer = AutoTokenizer.from_pretrained(
        args.reference_model, trust_remote_code=True
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.reference_model, **load_kwargs
    )
    scoring_tokenizer = AutoTokenizer.from_pretrained(
        args.scoring_model, trust_remote_code=True
    )
    scoring_model = AutoModelForCausalLM.from_pretrained(
        args.scoring_model, **load_kwargs
    )

    if not args.use_bf16:
        reference_model.to(args.DEVICE)
        scoring_model.to(args.DEVICE)

    reference_model.eval()
    scoring_model.eval()

    model_config = {
        "reference_tokenizer": reference_tokenizer,
        "reference_model":     reference_model,
        "scoring_tokenizer":   scoring_tokenizer,
        "scoring_model":       scoring_model,
    }

    output_dir = os.path.dirname(args.threshold_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    # ── Threshold calibration ────────────────────────────────────────────
    logging.info(f"Loading threshold data from {args.threshold_file}")
    threshold_data = load_and_rename_sentence(args.threshold_file)

    train_predictions = {"Human": [], "LLM": []}
    skipped_train = 0
    for item in tqdm(threshold_data, desc="Calibration pass"):
        score = get_text_crit(item["text"], args, model_config)
        if np.isfinite(score):
            train_predictions[item["label"]].append(score)
        else:
            skipped_train += 1

    if skipped_train:
        logging.warning(
            f"Skipped {skipped_train} non-finite calibration scores "
            f"({skipped_train / len(threshold_data):.1%} of total)."
        )

    if not train_predictions["Human"] or not train_predictions["LLM"]:
        raise RuntimeError(
            "Threshold calibration produced no valid scores for one or both classes. "
            "Check your threshold file and model compatibility."
        )

    # FIX #1 + #6 + #9 all happen inside calibrate_threshold
    global_threshold, global_direction, global_train_auc = calibrate_threshold(
        train_predictions["Human"],
        train_predictions["LLM"],
        label="global",
    )

    # ── Test files ───────────────────────────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_files = [f.strip() for f in args.test_data_path.split(",") if f.strip()]

    for idx, filename in enumerate(test_files, start=1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing on: {filename}")
        logging.info(f"{'='*60}")

        test_data = load_and_rename_sentence(filename)
        test_data = check_data_overlap(threshold_data, test_data, filename)

        if not test_data:
            logging.warning(
                f"No test data remaining after overlap removal for '{filename}'. Skipping."
            )
            continue

        preds        = {"Human": [], "LLM": []}
        y_true       = []
        all_scores   = []
        skipped_test = 0

        for item in tqdm(test_data, desc=os.path.basename(filename)):
            score = get_text_crit(item["text"], args, model_config)
            label = item["label"]
            item["text_crit"] = score

            if np.isfinite(score):
                preds[label].append(score)
                all_scores.append(score)
                y_true.append(0 if label == "Human" else 1)
            else:
                skipped_test += 1

        # FIX #8: log non-finite count
        if skipped_test:
            logging.warning(
                f"Skipped {skipped_test} non-finite test scores "
                f"({skipped_test / len(test_data):.1%}) in '{filename}'."
            )

        if not all_scores:
            logging.warning(f"No valid scores for '{filename}'. Skipping.")
            continue

        if not preds["Human"] or not preds["LLM"]:
            logging.warning(
                f"Only one class present in valid scores for '{filename}'. "
                f"AUC will be undefined."
            )

        # ── Decide threshold to use ──────────────────────────────────────
        # FIX #10: optional per-file recalibration using test labels
        # (only enabled with --per_file_threshold; uses global otherwise)
        if args.per_file_threshold and preds["Human"] and preds["LLM"]:
            logging.info("Recalibrating threshold on this test file (--per_file_threshold).")
            threshold, direction, _ = calibrate_threshold(
                preds["Human"], preds["LLM"], label=os.path.basename(filename)
            )
        else:
            threshold = global_threshold
            direction = global_direction

        # FIX #1: apply direction-corrected threshold
        y_pred = [
            1 if direction * s > direction * threshold else 0
            for s in all_scores
        ]

        # ── Metrics ──────────────────────────────────────────────────────
        tn, fp, fn, tp = compute_confusion(y_true, y_pred)
        precision = tp / (tp + fp)             if (tp + fp)             else 0.0
        recall    = tp / (tp + fn)             if (tp + fn)             else 0.0
        f1        = (2 * precision * recall /
                     (precision + recall))     if (precision + recall)  else 0.0
        accuracy  = (tp + tn) / len(y_true)    if y_true                else 0.0

        roc_auc = float("nan")
        if preds["Human"] and preds["LLM"]:
            raw_auc, *_ = get_roc_metrics(preds["Human"], preds["LLM"])
            # Report the corrected AUC (direction-normalised)
            roc_auc = raw_auc if direction == 1 else 1.0 - raw_auc

        logging.info(
            f"Results for '{os.path.basename(filename)}':\n"
            f"  ROC-AUC   : {roc_auc:.4f}\n"
            f"  Accuracy  : {accuracy:.4f}\n"
            f"  Precision : {precision:.4f}\n"
            f"  Recall    : {recall:.4f}\n"
            f"  F1        : {f1:.4f}\n"
            f"  Threshold : {threshold:.4f}  (direction={'higher→LLM' if direction==1 else 'lower→LLM'})\n"
            f"  Conf mat  : TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
            f"  Samples   : {len(all_scores)} valid / {len(test_data)} total"
        )

        result = {
            "roc_auc":           roc_auc,
            "optimal_threshold": threshold,
            "score_direction":   direction,
            "conf_matrix":       [[tn, fp], [fn, tp]],
            "precision":         precision,
            "recall":            recall,
            "f1":                f1,
            "accuracy":          accuracy,
            "num_examples":      len(all_scores),
            "num_skipped":       skipped_test,
        }

        base_name   = os.path.splitext(os.path.basename(filename))[0]
        data_file   = os.path.join(output_dir, f"{base_name}_data_{ts}_{idx}.json")
        result_file = os.path.join(output_dir, f"{base_name}_result_{ts}_{idx}.json")

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        logging.info(f"Saved data   → {data_file}")
        logging.info(f"Saved result → {result_file}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastDetectGPT evaluation — fixed version"
    )
    parser.add_argument(
        "--threshold_file",
        type=str, required=True,
        help="JSON file with labelled Human/LLM samples for threshold calibration."
    )
    parser.add_argument(
        "--test_data_path",
        type=str, required=True,
        help="Comma-separated list of test JSON file paths."
    )
    parser.add_argument(
        "--reference_model",
        type=str, default="Qwen/Qwen3-4B",
        help="HuggingFace model name for the reference model."
    )
    parser.add_argument(
        "--scoring_model",
        type=str, default="Qwen/Qwen3-4B",
        help="HuggingFace model name for the scoring model."
    )
    parser.add_argument(
        "--DEVICE", type=str, default="cuda",
        help="Torch device string (cuda / cpu)."
    )
    parser.add_argument(
        "--seed", type=int, default=2023,
        help="Global random seed."
    )
    parser.add_argument(
        "--use_bf16", action="store_true",
        help="Use BF16 for model inference (requires CUDA with BF16 support)."
    )
    parser.add_argument(
        "--per_file_threshold", action="store_true",
        help=(
            "Recalibrate the decision threshold on each test file separately "
            "(useful when domain shifts exist between threshold and test data)."
        ),
    )
    args = parser.parse_args()

    if args.use_bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; cannot use --use_bf16.")
        if not torch.cuda.is_bf16_supported():
            logging.warning("BF16 not supported on this device — falling back to FP32.")
            args.use_bf16 = False

    experiment(args)