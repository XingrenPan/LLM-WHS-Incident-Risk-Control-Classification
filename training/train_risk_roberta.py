#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_train_evaluate_roberta_risk_23class.py

Train and evaluate a traditional RoBERTa Risk Category classifier on the curated 23-class risk dataset.

Purpose:
- The Llama Risk Category model has already been completed.
- This script trains a traditional NLP baseline/model for the web application:
    RoBERTa-base -> 23-class Risk Category classifier

Default expected dataset:
- data/risk_description_only_main_classes_curated_23class

The script is intentionally robust:
- It tries to find train/test files automatically inside --dataset-dir.
- It supports csv/xlsx/json/jsonl/parquet.
- It tries to auto-detect text and label columns.
- It saves model, tokenizer, label mapping, evaluation metrics, predictions, classification report,
  and confusion pairs.

Batch-size policy:
- Default train batch size is 32 and eval batch size is 64.
- Batch sizes above 64 are blocked for this project.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MISSING_LABEL_VALUES = {
    "", "-", "--", "0", "0.0", "nan", "none", "null", "na", "n/a",
    "missing", "unknown", "not applicable",
}

TEXT_COL_CANDIDATES = [
    "WHAT_HAPPENED_ENGLISH", "what_happened_english", "description", "Description",
    "text", "Text", "incident_description", "Incident Description", "input_text",
]

LABEL_COL_CANDIDATES = [
    "risk_category", "Risk Category", "SAFETY_RISK_CATEGORY", "safety_risk_category",
    "risk_for_control_curated", "curated_risk_category", "final_risk_category",
    "label", "Label", "target", "category", "Category",
]


def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip()


def is_missing_label(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return norm_space(x).lower() in MISSING_LABEL_VALUES


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_file_by_keywords(dataset_dir: Path, keywords: List[str]) -> Optional[Path]:
    exts = {".csv", ".xlsx", ".xls", ".jsonl", ".json", ".parquet"}
    files = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    lowered = [(p, p.name.lower()) for p in files]
    for kw in keywords:
        for p, name in lowered:
            if kw in name:
                return p
    return None


def auto_find_train_test_files(dataset_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    train = find_file_by_keywords(dataset_dir, ["train"])
    test = find_file_by_keywords(dataset_dir, ["test", "eval", "validation", "valid", "val"])
    return train, test


def auto_detect_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if kind == "text":
        object_cols = [c for c in cols if df[c].dtype == "object"]
        if object_cols:
            scores = []
            for c in object_cols:
                lens = df[c].dropna().astype(str).str.len()
                scores.append((float(lens.mean()) if len(lens) else 0.0, c))
            scores.sort(reverse=True)
            return scores[0][1]

    if kind == "label":
        scores = []
        for c in cols:
            nunique = df[c].dropna().nunique()
            if 2 <= nunique <= 100:
                name = str(c).lower()
                bonus = 100 if any(x in name for x in ["risk", "category", "label", "target"]) else 0
                scores.append((bonus - abs(nunique - 23), c))
        if scores:
            scores.sort(reverse=True)
            return scores[0][1]

    raise KeyError(f"Could not auto-detect {kind} column. Available columns: {cols}")


def clean_dataset(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = df.copy()
    out[text_col] = out[text_col].apply(norm_space)
    out[label_col] = out[label_col].apply(norm_space)
    out = out[out[text_col].astype(str).str.len() > 0].copy()
    out = out[~out[label_col].apply(is_missing_label)].copy()
    return out.reset_index(drop=True)


def load_train_test(args) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    dataset_dir = Path(args.dataset_dir)
    train_file = Path(args.train_file) if args.train_file else None
    test_file = Path(args.test_file) if args.test_file else None

    if train_file is None or test_file is None:
        auto_train, auto_test = auto_find_train_test_files(dataset_dir)
        train_file = train_file or auto_train
        test_file = test_file or auto_test

    if train_file is not None and test_file is not None and train_file.exists() and test_file.exists():
        print(f"Using train file: {train_file}")
        print(f"Using test/eval file: {test_file}")
        train_df = read_table(train_file)
        test_df = read_table(test_file)
        text_col = args.text_col or auto_detect_column(train_df, TEXT_COL_CANDIDATES, "text")
        label_col = args.label_col or auto_detect_column(train_df, LABEL_COL_CANDIDATES, "label")

        if text_col not in test_df.columns:
            text_col_test = auto_detect_column(test_df, TEXT_COL_CANDIDATES, "text")
            test_df = test_df.rename(columns={text_col_test: text_col})
        if label_col not in test_df.columns:
            label_col_test = auto_detect_column(test_df, LABEL_COL_CANDIDATES, "label")
            test_df = test_df.rename(columns={label_col_test: label_col})

        return clean_dataset(train_df, text_col, label_col), clean_dataset(test_df, text_col, label_col), text_col, label_col

    # Fallback: split one full curated file.
    full_file = Path(args.full_file) if args.full_file else find_file_by_keywords(dataset_dir, ["curated", "dataset", "data", "full"])
    if full_file is None or not full_file.exists():
        raise FileNotFoundError(
            f"Could not find train/test files in {dataset_dir}. Pass --train-file and --test-file explicitly."
        )
    print(f"Train/test files not found. Splitting full file: {full_file}")
    full_df = read_table(full_file)
    text_col = args.text_col or auto_detect_column(full_df, TEXT_COL_CANDIDATES, "text")
    label_col = args.label_col or auto_detect_column(full_df, LABEL_COL_CANDIDATES, "label")
    full_df = clean_dataset(full_df, text_col, label_col)
    train_df, test_df = train_test_split(
        full_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=full_df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), text_col, label_col


class RiskTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_training_args(args, output_dir: Path) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir / "trainer_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "seed": args.seed,
        "report_to": "none",
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"
    if "bf16" in params:
        kwargs["bf16"] = bool(args.bf16 and torch.cuda.is_available())
    if "fp16" in params:
        kwargs["fp16"] = bool(args.fp16 and torch.cuda.is_available() and not args.bf16)
    if "dataloader_num_workers" in params:
        kwargs["dataloader_num_workers"] = args.num_workers
    return TrainingArguments(**kwargs)


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def save_json(obj: Any, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/risk_description_only_main_classes_curated_23class")
    parser.add_argument("--train-file", default="")
    parser.add_argument("--test-file", default="")
    parser.add_argument("--full-file", default="")
    parser.add_argument("--text-col", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--test-size", type=float, default=0.30)

    parser.add_argument("--base-model", default="roberta-base")
    parser.add_argument("--model-dir", default="models/roberta_risk_category_23class_curated")
    parser.add_argument("--output-dir", default="outputs/risk_category_prediction_roberta_23class/evaluation")

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted-loss", action="store_true", default=True)
    parser.add_argument("--no-weighted-loss", dest="weighted_loss", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    if args.train_batch_size > 64 or args.eval_batch_size > 64:
        raise ValueError("For this project, --train-batch-size and --eval-batch-size should be <= 64.")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading curated risk dataset...")
    train_df, test_df, text_col, label_col = load_train_test(args)
    print(f"Detected text column: {text_col}")
    print(f"Detected label column: {label_col}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    labels_all = sorted(pd.concat([train_df[label_col], test_df[label_col]], ignore_index=True).dropna().astype(str).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_all)}
    id2label = {i: label for label, i in label2id.items()}

    train_df["label_id"] = train_df[label_col].astype(str).map(label2id)
    test_df["label_id"] = test_df[label_col].astype(str).map(label2id)
    if train_df["label_id"].isna().any() or test_df["label_id"].isna().any():
        raise RuntimeError("Some labels could not be mapped to label ids.")

    label_distribution = {
        "train": train_df[label_col].value_counts().to_dict(),
        "test": test_df[label_col].value_counts().to_dict(),
        "all": pd.concat([train_df[label_col], test_df[label_col]]).value_counts().to_dict(),
    }
    save_json(label2id, model_dir / "label2id.json")
    save_json({str(k): v for k, v in id2label.items()}, model_dir / "id2label.json")
    save_json(labels_all, model_dir / "label_order.json")
    save_json({
        "dataset_dir": args.dataset_dir,
        "text_col": text_col,
        "label_col": label_col,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_labels": int(len(labels_all)),
        "labels": labels_all,
        "label_distribution": label_distribution,
    }, output_dir / "dataset_info.json")

    print("Labels:")
    print(json.dumps(labels_all, indent=2, ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=len(labels_all),
        label2id=label2id,
        id2label={str(i): label for i, label in id2label.items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, config=config)

    train_dataset = RiskTextDataset(train_df[text_col].astype(str).tolist(), train_df["label_id"].astype(int).tolist(), tokenizer, args.max_length)
    test_dataset = RiskTextDataset(test_df[text_col].astype(str).tolist(), test_df["label_id"].astype(int).tolist(), tokenizer, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if args.weighted_loss:
        y_train = train_df["label_id"].astype(int).values
        classes = np.array(sorted(np.unique(y_train)))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_full = np.ones(len(labels_all), dtype=np.float32)
        for cls_id, w in zip(classes, weights):
            weight_full[int(cls_id)] = float(w)
        class_weights = torch.tensor(weight_full, dtype=torch.float32)
        save_json({id2label[i]: float(weight_full[i]) for i in range(len(weight_full))}, output_dir / "class_weights.json")
        print("Using weighted cross-entropy loss.")

    training_args = build_training_args(args, output_dir)
    callbacks = []
    if args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
        class_weights=class_weights,
    )
    try:
        trainer = WeightedTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = WeightedTrainer(**trainer_kwargs, tokenizer=tokenizer)

    print("Starting RoBERTa Risk Category training...")
    train_result = trainer.train()
    print("Evaluating best model...")
    eval_metrics = trainer.evaluate(test_dataset)

    print("Saving model and tokenizer...")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    y_true = pred_output.label_ids
    y_pred = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_conf = probs.max(axis=-1)
    pred_labels = [id2label[int(i)] for i in y_pred]
    true_labels = [id2label[int(i)] for i in y_true]

    pred_df = test_df.copy()
    pred_df["true_label"] = true_labels
    pred_df["pred_label"] = pred_labels
    pred_df["pred_confidence"] = pred_conf
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
    top3 = np.argsort(-probs, axis=1)[:, :3]
    for k in range(3):
        pred_df[f"top{k+1}_label"] = [id2label[int(i)] for i in top3[:, k]]
        pred_df[f"top{k+1}_prob"] = probs[np.arange(len(probs)), top3[:, k]]

    predictions_path = output_dir / "roberta_risk_test_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    report = classification_report(true_labels, pred_labels, labels=labels_all, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    report_path = output_dir / "roberta_risk_classification_report.csv"
    report_df.to_csv(report_path, index=False)

    confusion = (
        pd.DataFrame({"true_label": true_labels, "pred_label": pred_labels})
        .groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion_path = output_dir / "roberta_risk_confusion_pairs.csv"
    confusion.to_csv(confusion_path, index=False)

    mismatch_path = output_dir / "roberta_risk_mismatches.csv"
    pred_df[~pred_df["correct"]].to_csv(mismatch_path, index=False)

    summary = {
        "task": "RoBERTa Risk Category 23-class classification",
        "base_model": args.base_model,
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "text_col": text_col,
        "label_col": label_col,
        "num_labels": int(len(labels_all)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "weighted_loss": bool(args.weighted_loss),
        "max_length": int(args.max_length),
        "epochs_requested": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "gradient_accumulation_steps": int(args.grad_accum_steps),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "train_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in train_result.metrics.items()},
        "eval_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in eval_metrics.items()},
        "test_accuracy_manual": float(accuracy_score(true_labels, pred_labels)),
        "test_macro_f1_manual": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "test_weighted_f1_manual": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "correct_count": int(pred_df["correct"].sum()),
        "wrong_count": int((~pred_df["correct"]).sum()),
        "label_distribution": label_distribution,
        "prediction_distribution": dict(Counter(pred_labels)),
        "output_files": {
            "model_dir": str(model_dir),
            "predictions_csv": str(predictions_path),
            "classification_report_csv": str(report_path),
            "confusion_pairs_csv": str(confusion_path),
            "mismatches_csv": str(mismatch_path),
            "dataset_info_json": str(output_dir / "dataset_info.json"),
            "label2id_json": str(model_dir / "label2id.json"),
            "id2label_json": str(model_dir / "id2label.json"),
        },
    }
    summary_path = output_dir / "roberta_risk_evaluation_summary.json"
    save_json(summary, summary_path)

    print("\nDone. RoBERTa Risk Category model saved to:")
    print(model_dir)
    print("\nEvaluation outputs saved to:")
    print(output_dir)
    print("\nKey summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
