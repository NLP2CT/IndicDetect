import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Monkey-patch: bitsandbytes calls nn.Module.set_submodule() but
# Qwen2ForSequenceClassification's custom __getattr__ hides it.
# This patch restores the method on all nn.Module instances before any
# bitsandbytes code runs.
# ---------------------------------------------------------------------------
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module):
        atoms = target.split(".")
        mod = self
        for atom in atoms[:-1]:
            mod = getattr(mod, atom)
        setattr(mod, atoms[-1], module)
    nn.Module.set_submodule = _set_submodule

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset

from metrics import get_roc_metrics, get_metrics

_DEFAULT_HF_MODEL_ID   = "Qwen/Qwen2.5-7B"
_DEFAULT_MODELS_DIR    = os.path.expanduser("~/IndicDetect/models")
_DEFAULT_MAX_LENGTH    = 512
_DEFAULT_BATCH_SIZE    = 1
_DEFAULT_GRAD_ACCUM    = 16
_DEFAULT_EPOCHS        = 1
_DEFAULT_LR            = 5e-5
_DEFAULT_WARMUP        = 50
_DEFAULT_LOG_STEPS     = 10
_DEFAULT_SAVE_STEPS    = 200
_DEFAULT_EVAL_STEPS    = 200
_DEFAULT_WEIGHT_DECAY  = 0.05
_DEFAULT_LORA_R        = 8
_DEFAULT_LORA_ALPHA    = 16
_DEFAULT_LORA_DROPOUT  = 0.1
_DEFAULT_LORA_MODULES  = ["q_proj", "v_proj"]

LABEL_MAP   = {"human": 0, "llm": 1}
ID_TO_LABEL = {0: "human", 1: "llm"}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def section(title: str):
    line = "=" * 60
    print(f"\n{line}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{line}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold_file",      required=True)
    p.add_argument("--test_data_path",      required=True)
    p.add_argument("--hf_model_id",         default=_DEFAULT_HF_MODEL_ID)
    p.add_argument("--model_cache_dir",     default=None)
    p.add_argument("--models_dir",          default=_DEFAULT_MODELS_DIR)
    p.add_argument("--max_length",          type=int,   default=_DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size",          type=int,   default=_DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum_steps",    type=int,   default=_DEFAULT_GRAD_ACCUM)
    p.add_argument("--num_epochs",          type=int,   default=_DEFAULT_EPOCHS)
    p.add_argument("--learning_rate",       type=float, default=_DEFAULT_LR)
    p.add_argument("--warmup_steps",        type=int,   default=_DEFAULT_WARMUP)
    p.add_argument("--logging_steps",       type=int,   default=_DEFAULT_LOG_STEPS)
    p.add_argument("--save_steps",          type=int,   default=_DEFAULT_SAVE_STEPS)
    p.add_argument("--eval_steps",          type=int,   default=_DEFAULT_EVAL_STEPS)
    p.add_argument("--weight_decay",        type=float, default=_DEFAULT_WEIGHT_DECAY)
    p.add_argument("--use_bf16",            action="store_true")
    p.add_argument("--lora_r",              type=int,   default=_DEFAULT_LORA_R)
    p.add_argument("--lora_alpha",          type=int,   default=_DEFAULT_LORA_ALPHA)
    p.add_argument("--lora_dropout",        type=float, default=_DEFAULT_LORA_DROPOUT)
    p.add_argument("--lora_target_modules", default=",".join(_DEFAULT_LORA_MODULES))

    args = p.parse_args()

    model_slug = args.hf_model_id.replace("/", "__")
    if args.model_cache_dir is None:
        args.model_cache_dir = os.path.join(args.models_dir, model_slug)

    args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    args.test_data_paths     = [p.strip() for p in args.test_data_path.split(",") if p.strip()]
    args.results_dir         = os.path.dirname(os.path.abspath(args.threshold_file))
    args.domain_name         = os.path.basename(args.results_dir)

    return args


def get_run_output_dir(args: argparse.Namespace) -> str:
    os.makedirs(args.models_dir, exist_ok=True)
    if not os.access(args.models_dir, os.W_OK):
        raise PermissionError(f"No write permission on: {args.models_dir}")
    train_stem = os.path.splitext(os.path.basename(args.threshold_file))[0]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = os.path.join(args.models_dir, f"{train_stem}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_unique_output_path(directory: str, base_name: str, extension: str) -> str:
    os.makedirs(directory, exist_ok=True)
    candidate = os.path.join(directory, f"{base_name}{extension}")
    if not os.path.exists(candidate):
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(directory, f"{base_name}_{timestamp}{extension}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base_name}_{timestamp}_{counter}{extension}")
        counter += 1
    return candidate


def compute_detection_score(human_prob: float, ai_prob: float, pred_label: str) -> float:
    raw_score  = ai_prob - human_prob
    confidence = abs(raw_score)
    score = confidence if pred_label == "llm" else -confidence
    return round(float(score), 10)


def _split_preds_by_label(ai_probs: np.ndarray, labels: np.ndarray):
    real_preds   = ai_probs[labels == 0].tolist()
    sample_preds = ai_probs[labels == 1].tolist()
    return real_preds, sample_preds


class VerboseCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step  = state.global_step
        total = state.max_steps
        pct   = 100 * step / total if total else 0
        parts = [f"step {step}/{total} ({pct:.1f}%)"]
        for key in ("loss", "learning_rate", "epoch"):
            if key in logs:
                val = logs[key]
                parts.append(f"{key}={val:.5f}" if isinstance(val, float) else f"{key}={val}")
        log("  " + "  |  ".join(parts))

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        section("Evaluation results")
        for k, v in metrics.items():
            print(f"    {k:<35} {v}", flush=True)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log(f"Checkpoint saved at step {state.global_step}")

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        section("Training started")
        log(f"Total steps : {state.max_steps}")
        log(f"Epochs      : {args.num_train_epochs}")
        log(f"Batch size  : {args.per_device_train_batch_size}  (grad accum × {args.gradient_accumulation_steps})")
        log(f"LR          : {args.learning_rate}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        section("Training complete")
        log(f"Best eval loss : {state.best_metric}")
        log(f"Checkpoint     : {state.best_model_checkpoint}")


def load_model_and_tokenizer(args: argparse.Namespace):
    os.makedirs(args.model_cache_dir, exist_ok=True)
    config_file = os.path.join(args.model_cache_dir, "config.json")
    cached      = os.path.exists(config_file)
    source      = args.model_cache_dir if cached else args.hf_model_id

    section("Loading model & tokenizer")
    if cached:
        log(f"Cache hit — loading from {args.model_cache_dir}")
    else:
        log(f"No cache found — downloading {args.hf_model_id}")
        log(f"Destination : {args.model_cache_dir}")

    log("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        padding_side="right",
        cache_dir=args.model_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer ready  |  vocab size = {tokenizer.vocab_size}")

    log("Configuring 4-bit quantisation (NF4, double quant, bfloat16 compute) …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    log("Loading model weights …  (this may take a few minutes on first run)")
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.model_cache_dir,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    log("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(model)

    log(f"Applying LoRA  (r={args.lora_r}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, modules={args.lora_target_modules}) …")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not cached:
        log("Saving tokenizer to cache …")
        tokenizer.save_pretrained(args.model_cache_dir)
        log(f"Tokenizer saved → {args.model_cache_dir}")

    log("Model ready.")
    return model, tokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int, desc: str = ""):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = self._load(data_path, desc)

    def _load(self, path: str, desc: str):
        log(f"Loading dataset{' (' + desc + ')' if desc else ''} : {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = []
        skipped = 0
        for item in tqdm(data, desc="  Parsing records", unit="rec", dynamic_ncols=True):
            label_str = item["label"].strip().lower()
            label = LABEL_MAP.get(label_str, -1)
            if label == -1:
                skipped += 1
                continue
            samples.append({
                "sample_number": item.get("sample_number", len(samples) + 1),
                "text":          item["text"],
                "Domain":        item.get("Domain", item.get("domain", "")),
                "total_tokens":  item.get("total_tokens", 0),
                "label":         label,
                "label_str":     label_str,
            })
        human_count = sum(1 for s in samples if s["label"] == 0)
        llm_count   = sum(1 for s in samples if s["label"] == 1)
        log(f"  Loaded {len(samples)} samples  |  human={human_count}  llm={llm_count}"
            + (f"  |  skipped={skipped}" if skipped else ""))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item     = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoding["labels"] = item["label"]
        return encoding


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs    = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    ai_probs = probs[:, 1]
    labels   = np.array(labels)

    real_preds, sample_preds = _split_preds_by_label(ai_probs, labels)
    roc_auc, optimal_threshold, _, _, _, _, _ = get_roc_metrics(real_preds, sample_preds)
    _, conf_matrix, precision, recall, f1, accuracy = get_metrics(
        real_preds, sample_preds, optimal_threshold
    )

    return {
        "roc_auc":           round(float(roc_auc), 4),
        "optimal_threshold": round(float(optimal_threshold), 4),
        "precision":         round(float(precision), 4),
        "recall":            round(float(recall), 4),
        "f1":                round(float(f1), 4),
        "accuracy":          round(float(accuracy), 4),
    }


def _run_inference(model, tokenizer, dataset, batch_size: int, desc: str):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader   = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    model.eval()
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc=f"  {desc}", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        labels = batch.pop("labels")
        batch  = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        all_logits.append(outputs.logits.float().cpu().numpy())
        all_labels.extend(labels.numpy().tolist())
        pbar.set_postfix({"batches_done": len(all_logits)})

    logits = np.concatenate(all_logits, axis=0)
    labels = np.array(all_labels)
    return logits, labels


def compute_optimal_threshold(model, tokenizer, threshold_file: str, args) -> float:
    section("Computing optimal threshold")
    log(f"Source : {threshold_file}")

    dataset        = TextClassificationDataset(threshold_file, tokenizer, args.max_length, "threshold file")
    logits, labels = _run_inference(model, tokenizer, dataset, args.batch_size, "Threshold inference")

    probs    = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    ai_probs = probs[:, 1]
    labels   = np.array(labels)

    real_preds, sample_preds = _split_preds_by_label(ai_probs, labels)

    log("Running get_roc_metrics to find optimal threshold …")
    roc_auc, optimal_threshold, _, _, _, _, _ = get_roc_metrics(real_preds, sample_preds)

    log(f"  ROC-AUC  = {roc_auc:.4f}")
    log(f"  τ*       = {optimal_threshold:.4f}  (Youden's J: max TPR - FPR)")
    return float(optimal_threshold)


def evaluate_single_file(model, tokenizer, test_path: str, threshold: float, args) -> dict:
    test_domain = os.path.basename(os.path.dirname(os.path.abspath(test_path)))

    section(f"Evaluating : {test_domain}")
    log(f"File      : {test_path}")
    log(f"Threshold : {threshold:.4f}")

    test_dataset   = TextClassificationDataset(test_path, tokenizer, args.max_length, test_domain)
    logits, labels = _run_inference(model, tokenizer, test_dataset, args.batch_size, f"Inference [{test_domain}]")

    probs       = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    human_probs = probs[:, 0]
    ai_probs    = probs[:, 1]
    labels      = np.array(labels)

    real_preds, sample_preds = _split_preds_by_label(ai_probs, labels)

    log("Computing ROC-AUC via get_roc_metrics …")
    roc_auc, _, _, _, _, _, _ = get_roc_metrics(real_preds, sample_preds)

    log("Computing classification metrics via get_metrics …")
    _, conf_matrix, precision, recall, f1, accuracy = get_metrics(
        real_preds, sample_preds, threshold
    )

    all_scores = real_preds + sample_preds
    preds_flat = np.array([1 if s >= threshold else 0 for s in all_scores])

    human_indices   = np.where(labels == 0)[0]
    llm_indices     = np.where(labels == 1)[0]
    ordered_indices = np.concatenate([human_indices, llm_indices])
    preds           = np.empty(len(labels), dtype=int)
    preds[ordered_indices] = preds_flat

    metrics_result = {
        "threshold_used": round(float(threshold), 4),
        "roc_auc":        round(float(roc_auc), 4),
        "conf_matrix":    conf_matrix,
        "precision":      round(float(precision), 4),
        "recall":         round(float(recall), 4),
        "f1":             round(float(f1), 4),
        "accuracy":       round(float(accuracy), 4),
        "num_examples":   int(len(labels)),
    }

    log("Results:")
    for k, v in metrics_result.items():
        print(f"    {k:<20} {v}", flush=True)

    log("Building per-sample prediction records …")
    data_records = []
    for i, sample in enumerate(tqdm(test_dataset.samples, desc="  Writing records", unit="rec", dynamic_ncols=True)):
        pred_label_str  = ID_TO_LABEL[int(preds[i])]
        detection_score = compute_detection_score(float(human_probs[i]), float(ai_probs[i]), pred_label_str)
        data_records.append({
            "sample_number":   sample["sample_number"],
            "text":            sample["text"],
            "Domain":          sample["Domain"],
            "total_tokens":    sample["total_tokens"],
            "label":           pred_label_str,
            "detection_score": detection_score,
        })

    data_out = get_unique_output_path(args.results_dir, f"{test_domain}_Test_Data", ".json")
    with open(data_out, "w", encoding="utf-8") as f:
        json.dump(data_records, f, ensure_ascii=False, indent=4)
    log(f"Data saved    → {data_out}")

    metrics_out = get_unique_output_path(args.results_dir, f"{test_domain}_Test_Result", ".json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=4)
    log(f"Metrics saved → {metrics_out}")

    return metrics_result


def evaluate_all_test_files(model, tokenizer, threshold: float, args):
    section(f"Running evaluation on {len(args.test_data_paths)} test file(s)")
    for i, test_path in enumerate(args.test_data_paths, 1):
        log(f"[{i}/{len(args.test_data_paths)}] {test_path}")
        evaluate_single_file(model, tokenizer, test_path, threshold, args)
    section("All evaluations complete")


def main():
    args = parse_args()

    section("Run configuration")
    print(f"  threshold_file : {args.threshold_file}", flush=True)
    print(f"  domain         : {args.domain_name}", flush=True)
    print(f"  results dir    : {args.results_dir}", flush=True)
    print(f"  test files     : {len(args.test_data_paths)}", flush=True)
    for tp in args.test_data_paths:
        print(f"    • {tp}", flush=True)
    print(f"  model          : {args.hf_model_id}", flush=True)
    print(f"  model cache    : {args.model_cache_dir}", flush=True)
    print(f"  max_length     : {args.max_length}", flush=True)
    print(f"  batch_size     : {args.batch_size}", flush=True)
    print(f"  grad_accum     : {args.grad_accum_steps}", flush=True)
    print(f"  epochs         : {args.num_epochs}", flush=True)
    print(f"  lr             : {args.learning_rate}", flush=True)
    print(f"  bf16           : {args.use_bf16}", flush=True)

    run_output_dir = get_run_output_dir(args)
    log(f"Run output dir : {run_output_dir}")

    model, tokenizer = load_model_and_tokenizer(args)

    section("Building datasets")
    train_dataset = TextClassificationDataset(args.threshold_file,     tokenizer, args.max_length, "train")
    eval_dataset  = TextClassificationDataset(args.test_data_paths[0], tokenizer, args.max_length, "eval")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    log(f"Train size : {len(train_dataset)}  |  Eval size : {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir                  = run_output_dir,
        num_train_epochs            = args.num_epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum_steps,
        gradient_checkpointing      = True,
        learning_rate               = args.learning_rate,
        weight_decay                = args.weight_decay,
        warmup_steps                = args.warmup_steps,
        lr_scheduler_type           = "cosine",
        logging_steps               = args.logging_steps,
        save_steps                  = args.save_steps,
        eval_steps                  = args.eval_steps,
        eval_strategy               = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        bf16                        = args.use_bf16,
        optim                       = "paged_adamw_8bit",
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,
        report_to                   = "none",
        save_total_limit            = 2,
        disable_tqdm                = False,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        processing_class = tokenizer,
        data_collator    = data_collator,
        compute_metrics  = compute_metrics,
        callbacks        = [
            EarlyStoppingCallback(early_stopping_patience=3),
            VerboseCallback(),
        ],
    )

    trainer.train()

    section("Saving fine-tuned model")
    log(f"Saving to {run_output_dir} …")
    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)
    log("Model and tokenizer saved.")

    optimal_threshold = compute_optimal_threshold(model, tokenizer, args.threshold_file, args)
    evaluate_all_test_files(model, tokenizer, optimal_threshold, args)


if __name__ == "__main__":
    main()