#!/usr/bin/env python3
import argparse
import logging
import os
import random
import json
from typing import Dict, Any, List

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from metrics import get_roc_metrics, get_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label):
    lbl = str(label).strip().lower()
    if lbl in ("human",):
        return "Human"
    if lbl in ("llm", "ai", "machine"):
        return "LLM"
    return None


def load_and_rename_sentence(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "sentence" in item and "text" not in item:
            item["text"] = item.pop("sentence")
        if "text" not in item:
            raise ValueError(f"Missing 'text' or 'sentence' key in entry: {item}")
        if item.get("label") is None:
            raise ValueError(f"Missing 'label' key in entry: {item}")
        lbl = normalize_label(item["label"])
        if lbl is None:
            raise ValueError(f"Invalid label '{item['label']}'. Expected 'Human' or 'LLM'.")
        item["label"] = lbl
    return data


def check_data_overlap(train_data, test_data, test_name):
    train_texts = {item["text"] for item in train_data}
    test_texts = {item["text"] for item in test_data}
    overlap = train_texts & test_texts
    if overlap:
        logging.warning(
            f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping texts between "
            f"train data and test file '{test_name}'. Removing from test set."
        )
        test_data = [item for item in test_data if item["text"] not in overlap]
        logging.info(f"Test set size after removing overlaps: {len(test_data)}")
    return test_data


def build_output_name(filename, parent_levels=2):
    parts = []
    path = os.path.abspath(filename)
    base = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.dirname(path)
    for _ in range(parent_levels):
        parts.append(os.path.basename(parent))
        parent = os.path.dirname(parent)
    parts.reverse()
    parts.append(base)
    return "_".join(parts)


def _unpack_roc(ret):
    if isinstance(ret, dict):
        return (
            ret.get("roc_auc"),
            ret.get("optimal_threshold"),
            ret.get("conf_matrix"),
            ret.get("precision"),
            ret.get("recall"),
            ret.get("f1"),
            ret.get("accuracy"),
            ret.get("tpr_at_fpr_0_01"),
        )
    seq = list(ret)
    if len(seq) < 8:
        seq += [None] * (8 - len(seq))
    return tuple(seq[:8])


def _unpack_metrics(ret):
    if isinstance(ret, dict):
        return (
            ret.get("optimal_threshold"),
            ret.get("conf_matrix"),
            ret.get("precision"),
            ret.get("recall"),
            ret.get("f1"),
            ret.get("accuracy"),
        )
    seq = list(ret)
    if len(seq) < 6:
        seq += [None] * (6 - len(seq))
    return tuple(seq[:6])


def _split_paths(csv: str) -> List[str]:
    if csv is None:
        return []
    cleaned = []
    for x in csv.split(","):
        y = x.strip().strip("'").strip('"')
        if y:
            cleaned.append(y)
    return cleaned


def _has_any_path(csv: str) -> bool:
    return len(_split_paths(csv)) > 0


def _score_texts(detector, tokenizer, texts: List[str], device: str, batch_size: int = 64) -> List[float]:
    out: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokenized = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = detector(**tokenized).logits
            # keep your existing convention: column 0 is "Human side"
            probs = torch.softmax(logits, dim=-1)[:, 0].detach().cpu().numpy().tolist()
        out.extend(probs)
    return out


def compute_threshold_from_validation(detector, tokenizer, valid_data, device):
    texts = [item["text"] for item in valid_data]
    labels = [item["label"] for item in valid_data]
    scores = _score_texts(detector, tokenizer, texts, device, batch_size=64)

    preds = {"Human": [], "LLM": []}
    for lbl, sc in zip(labels, scores):
        if np.isfinite(sc):
            preds[lbl].append(-float(sc))

    if not preds["Human"] or not preds["LLM"]:
        raise RuntimeError("Validation data has no valid scores for one or both classes.")

    roc_auc, optimal_threshold, *_ = _unpack_roc(get_roc_metrics(preds["Human"], preds["LLM"]))
    logging.info(f"Validation threshold: {optimal_threshold:.6f} (AUC: {roc_auc:.4f})")
    return optimal_threshold


def eval_experiment(args, model_path: str, test_data_path: str,
                    optimal_threshold: float, train_data: List[Dict],
                    output_dir: str) -> None:
    filenames = _split_paths(test_data_path)
    if not filenames:
        return

    logging.info(f"Loading model from {model_path} ...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(args.DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    if "xlm-roberta-base" in args.model_name:
        model_name_tag = "xlm-roberta-base"
    elif "xlm-roberta-large" in args.model_name:
        model_name_tag = "xlm-roberta-large"
    else:
        model_name_tag = args.model_name

    for filename in filenames:
        logging.info(f"Testing on: {filename}")
        test_data = load_and_rename_sentence(filename)
        test_data = check_data_overlap(train_data, test_data, filename)

        if not test_data:
            logging.warning(f"No data remaining for {filename} after overlap removal. Skipping.")
            continue

        set_all_seeds(args.seed)

        texts = [item["text"] for item in test_data]
        labels = [item["label"] for item in test_data]
        scores_human_side = _score_texts(detector, tokenizer, texts, args.DEVICE, batch_size=64)

        for item, sc in zip(test_data, scores_human_side):
            item["prediction"] = float(sc)

        preds = {"Human": [], "LLM": []}
        for lbl, sc in zip(labels, scores_human_side):
            if np.isfinite(sc):
                preds[lbl].append(-float(sc))

        if not preds["Human"] or not preds["LLM"]:
            logging.warning(f"Missing predictions for one class in {filename}. Skipping.")
            continue

        roc_auc, _, _, _, _, _, _, tpr_at_fpr_0_01 = _unpack_roc(
            get_roc_metrics(preds["Human"], preds["LLM"])
        )

        opt_thr, conf_matrix, precision, recall, f1, accuracy = _unpack_metrics(
            get_metrics(preds["Human"], preds["LLM"], optimal_threshold)
        )

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": float(optimal_threshold),
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01,
            "num_examples": len(texts),
        }

        logging.info(f"Metrics: {result}")

        unique_name = build_output_name(filename)
        data_file = os.path.join(output_dir, f"{unique_name}.{model_name_tag}_data.json")
        result_file = os.path.join(output_dir, f"{unique_name}.{model_name_tag}_result.json")

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        logging.info(f"Saved -> {data_file}")
        logging.info(f"Saved -> {result_file}")


class JSONDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data[idx]
        text = row["text"]
        label = 0 if row["label"] == "Human" else 1
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = label
        return inputs


def compute_metrics(eval_pred) -> Dict[str, float]:
    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": float(acc), "f1": float(f1), "precision": float(precision), "recall": float(recall)}


class EvalAccuracyCallback(TrainerCallback):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else -1
        eval_accuracy = metrics.get("eval_accuracy")
        eval_f1 = metrics.get("eval_f1")
        eval_precision = metrics.get("eval_precision")
        eval_recall = metrics.get("eval_recall")
        print(
            f"Epoch: {epoch} - Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, "
            f"Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}"
        )
        os.makedirs(self.model_path, exist_ok=True)
        with open(os.path.join(self.model_path, "eval_result.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"Epoch: {epoch} - Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, "
                f"Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}\n"
            )


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 10, metric_key: str = "eval_loss", mode: str = "min"):
        self.patience = patience
        self.metric_key = metric_key
        self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.metric_key not in metrics:
            return
        current = metrics[self.metric_key]
        better = (current >= self.best_metric) if self.mode == "max" else (current <= self.best_metric)
        if better:
            self.best_metric = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


def stratified_split_min_per_class(data: List[Dict[str, Any]], valid_per_class: int):
    """Always keep at least 1 training example per class when possible."""
    human = [x for x in data if x["label"] == "Human"]
    llm = [x for x in data if x["label"] == "LLM"]

    if not human or not llm:
        raise ValueError(f"Need both classes. Got Human={len(human)} LLM={len(llm)}")

    # If dataset is small, shrink validation size so train is not empty
    v_h = min(valid_per_class, max(len(human) - 1, 0))
    v_l = min(valid_per_class, max(len(llm) - 1, 0))

    if v_h == 0 or v_l == 0:
        raise ValueError(
            f"Not enough examples to create validation split with at least 1 train sample per class. "
            f"Human={len(human)} LLM={len(llm)}"
        )

    valid = human[-v_h:] + llm[-v_l:]
    train = human[:-v_h] + llm[:-v_l]
    return train, valid


def run(args):
    output_dir = os.path.dirname(os.path.abspath(args.train_data_path))
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, args.save_model_path)
    result_path = os.path.join(output_dir, f"{args.save_model_path}_results")

    if args.mode == "train":
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "eval_result.txt"), "w", encoding="utf-8"):
            pass

        train_data_all = load_and_rename_sentence(args.train_data_path)

        set_all_seeds(args.seed)
        random.shuffle(train_data_all)

        # For a ~200-example train file, do NOT hardcode "-200" slicing.
        # Use a small per-class validation split (default 20 per class).
        train_data, valid_data = stratified_split_min_per_class(
            train_data_all, valid_per_class=args.valid_per_class
        )

        set_all_seeds(args.seed)
        random.shuffle(train_data)
        random.shuffle(valid_data)

        logging.info(f"Training data size: {len(train_data)}")
        logging.info(f"Validation data size: {len(valid_data)}")

        # Use Auto* so roberta/xlm-roberta matches correctly
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

        # If you only have ~200 total, train on all remaining (no [:2000] cap)
        train_dataset = JSONDataset(train_data, tokenizer)
        valid_dataset = JSONDataset(valid_data, tokenizer)

        if len(train_dataset) == 0:
            raise ValueError(
                "Train dataset is empty. Check your split or set --valid_per_class smaller."
            )

        training_args = TrainingArguments(
            output_dir=result_path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",
            seed=args.seed,
            save_total_limit=1,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EvalAccuracyCallback(model_path),
                EarlyStoppingCallback(patience=10, metric_key="eval_loss", mode="min"),
            ],
        )

        trainer.train()
        eval_result = trainer.evaluate()
        for key in sorted(eval_result.keys()):
            logging.info(f"{key}: {eval_result[key]}")

        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        model.config.save_pretrained(model_path)

        with open(os.path.join(model_path, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=4, ensure_ascii=False)

        detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(args.DEVICE)
        eval_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        optimal_threshold = compute_threshold_from_validation(
            detector, eval_tokenizer, valid_data, args.DEVICE
        )
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Optimal threshold from validation: {optimal_threshold:.6f}")

        eval_experiment(args, model_path, args.test_data_path,
                        optimal_threshold, train_data_all, output_dir)

        if _has_any_path(args.transfer_test_data_path):
            eval_experiment(args, model_path, args.transfer_test_data_path,
                            optimal_threshold, train_data_all, output_dir)

    elif args.mode == "eval":
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

        train_data_all = load_and_rename_sentence(args.train_data_path)

        # For eval thresholding, make a validation split the same way
        train_data, valid_data = stratified_split_min_per_class(
            train_data_all, valid_per_class=args.valid_per_class
        )

        detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(args.DEVICE)
        eval_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        optimal_threshold = compute_threshold_from_validation(
            detector, eval_tokenizer, valid_data, args.DEVICE
        )
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Optimal threshold from validation: {optimal_threshold:.6f}")

        eval_experiment(args, model_path, args.test_data_path,
                        optimal_threshold, train_data_all, output_dir)

        if _has_any_path(args.transfer_test_data_path):
            eval_experiment(args, model_path, args.transfer_test_data_path,
                            optimal_threshold, train_data_all, output_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--save_model_path", default="roberta_base_classifier", type=str)
    parser.add_argument("--train_data_path", default="", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--transfer_test_data_path", type=str, default="")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", default=2023, type=int)

    # NEW: control validation size so small datasets work
    parser.add_argument("--valid_per_class", default=20, type=int,
                        help="How many validation examples per class (Human/LLM).")

    parser.add_argument("--mode", default="train", type=str, choices=["train", "eval"])
    parser.add_argument("--DEVICE", default="cuda", type=str)
    args = parser.parse_args()

    if args.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA not available; falling back to CPU.")
        args.DEVICE = "cpu"

    run(args)