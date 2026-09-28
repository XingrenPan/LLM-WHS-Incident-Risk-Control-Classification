#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply the curated 23-class Llama-3.1-8B-Instruct QLoRA Risk Category model
at full-table scale.

Default purpose:
  - Read the original Events Inc. table.
  - Select rows where SAFETY_RISK_CATEGORY is missing.
  - Predict one of the curated 23 Risk Categories.
  - Write auditable candidate-label columns without modifying the original label column.

Default inputs:
  data/Events_Inc.xlsx
  data/risk_description_only_main_classes_curated_23class/risk_label_list.json
  models/llama31_8b_risk_description_only_main_classes_curated_23class

Default outputs:
  outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated.csv
  outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated.partial.csv
  outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated_summary.json
  outputs/risk_category_prediction_curated_23class/full_prediction/predicted_risk_category_distribution_curated.csv
  outputs/risk_category_prediction_curated_23class/full_prediction/need_further_review_curated.csv

Notes:
  - This script treats generated labels as candidate labels, not final ground truth.
  - It intentionally uses neutral wording such as curated / curation and contains no person-specific wording.
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel


DEFAULT_INPUT_FILE = "data/Events_Inc.xlsx"
DEFAULT_LABEL_LIST_JSON = "data/risk_description_only_main_classes_curated_23class/risk_label_list.json"
DEFAULT_MODEL_DIR = "models/llama31_8b_risk_description_only_main_classes_curated_23class"
DEFAULT_OUTPUT_DIR = "outputs/risk_category_prediction_curated_23class/full_prediction"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DEFAULT_TEXT_COL = "WHAT_HAPPENED_ENGLISH"
DEFAULT_RISK_COL = "SAFETY_RISK_CATEGORY"
DEFAULT_LABEL_SOURCE = "llama31_8b_risk_description_only_main_classes_curated_23class"

# Output columns are kept close to the Report 5 format, with curated-specific source/version information.
PRED_COL = "predicted_risk_category_23class"
RAW_COL = "raw_model_output"
PARSE_COL = "risk_parse_status"
STATUS_COL = "risk_review_status"
PRIORITY_COL = "risk_review_priority"
REASON_COL = "risk_review_reason"
SOURCE_COL = "risk_label_source"
RISK_FOR_CONTROL_COL = "risk_for_control_curated"
RISK_FOR_CONTROL_SOURCE_COL = "risk_for_control_source"


# Categories that were weaker or more boundary-sensitive in earlier checks.
# These are not rejected automatically, but can be flagged for moderate review priority if desired.
BOUNDARY_SENSITIVE_CATEGORIES = {
    "Occupational Safety",
    "Acute Chemical Exposure",
    "Energy release (excl. Electrical)",
    "Asset Integrity",
    "Entanglement / crushing",
    "Process Safety",
    "Fall from height",
}


ALIASES = {
    "falling from height": "Fall from height",
    "fall from heights": "Fall from height",
    "dropped objects": "Dropped / Falling Object",
    "dropped object": "Dropped / Falling Object",
    "falling object": "Dropped / Falling Object",
    "falling objects": "Dropped / Falling Object",
    "uncontrolled release of energy": "Energy release (excl. Electrical)",
    "energy release": "Energy release (excl. Electrical)",
    "electrical": "Electrical (incl. Arc Flash/Blast)",
    "vehicle & mobile equipment": "Vehicles & Mobile Equipment",
    "vehicles and mobile equipment": "Vehicles & Mobile Equipment",
    "mobile equipment": "Vehicles & Mobile Equipment",
    "non process fire and explosion": "Non Process Fire & Explosion",
    "non-process fire & explosion": "Non Process Fire & Explosion",
    "non process fire & explosion": "Non Process Fire & Explosion",
    "non-process fire and explosion obs": "Non-process Fire and Explosion (obs)",
    "non process fire and explosion obs": "Non-process Fire and Explosion (obs)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict missing Risk Categories in the full Events Inc. table using the curated 23-class Llama adapter."
    )

    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--label-list-json", default=DEFAULT_LABEL_LIST_JSON)
    parser.add_argument("--adapter-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)

    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL)
    parser.add_argument("--risk-col", default=DEFAULT_RISK_COL)

    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-description-chars", type=int, default=8)
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)

    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap for smoke testing. 0 means no cap.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help="Optional inclusive original DataFrame row index lower bound.",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="Optional exclusive original DataFrame row index upper bound.",
    )
    parser.add_argument(
        "--predict-all",
        action="store_true",
        help="Predict all rows with non-empty descriptions, not only rows with missing risk labels. Use with caution.",
    )
    parser.add_argument(
        "--overwrite-existing-prediction-columns",
        action="store_true",
        help="Allow overwriting existing prediction output columns in the loaded input file.",
    )
    parser.add_argument(
        "--flag-boundary-sensitive-categories",
        action="store_true",
        help="Mark boundary-sensitive predicted categories as medium review priority instead of low.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16. By default bf16 is used on CUDA devices.",
    )

    return parser.parse_args()


def normalise_text(s: object) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def normalise_label_text(s: object) -> str:
    return normalise_text(s).lower()


def load_labels(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        if "labels" in labels and isinstance(labels["labels"], list):
            labels = labels["labels"]
        else:
            labels = [v for _, v in sorted(labels.items(), key=lambda kv: str(kv[0]))]

    labels = [normalise_text(x) for x in labels if normalise_text(x)]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate labels found in risk label list.")
    if len(labels) != 23:
        print(f"[WARN] Loaded {len(labels)} labels, not 23. Continuing because the label list is the source of truth.")
    return labels


def read_table(input_file: Path) -> pd.DataFrame:
    suffix = input_file.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(input_file, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(input_file)
    raise ValueError(f"Unsupported input file type: {input_file}")


def ensure_output_columns(df: pd.DataFrame, overwrite: bool) -> pd.DataFrame:
    df = df.copy()
    output_cols = [
        PRED_COL,
        RAW_COL,
        PARSE_COL,
        STATUS_COL,
        PRIORITY_COL,
        REASON_COL,
        SOURCE_COL,
        RISK_FOR_CONTROL_COL,
        RISK_FOR_CONTROL_SOURCE_COL,
    ]

    existing = [c for c in output_cols if c in df.columns]
    if existing and not overwrite:
        raise ValueError(
            "The input table already contains prediction output columns:\n"
            + "\n".join(existing)
            + "\nUse --overwrite-existing-prediction-columns if you intentionally want to refresh them."
        )

    for col in output_cols:
        df[col] = ""

    return df


def build_user_prompt(text: str, labels: List[str]) -> str:
    label_block = "\n".join(f"- {x}" for x in labels)
    return (
        "Classify the following incident description into exactly one Risk Category.\n\n"
        "Allowed Risk Categories:\n"
        f"{label_block}\n\n"
        "Incident description:\n"
        f"{text}\n\n"
        "Return only the exact Risk Category name."
    )


def build_prompt(tokenizer, text: str, labels: List[str]) -> str:
    system_msg = (
        "You are a safety incident classification assistant. "
        "You must choose one exact Risk Category from the allowed list. "
        "Do not explain your answer."
    )
    user_msg = build_user_prompt(text, labels)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def parse_prediction(raw_text: object, labels: List[str]) -> Tuple[Optional[str], str]:
    if raw_text is None:
        return None, "empty"

    text = normalise_text(raw_text)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text).strip()
    first_line = text.splitlines()[0].strip() if text else ""

    norm_to_label = {normalise_label_text(x): x for x in labels}

    # Exact match on first line or full output.
    for candidate in [first_line, text]:
        key = normalise_label_text(candidate)
        if key in norm_to_label:
            return norm_to_label[key], "exact_match"
        if key in ALIASES and ALIASES[key] in labels:
            return ALIASES[key], "alias_match"

    # Remove common prefixes and retry.
    cleaned = re.sub(
        r"^(risk category|category|answer|prediction|predicted risk category)\s*[:\-]\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip()
    key = normalise_label_text(cleaned)
    if key in norm_to_label:
        return norm_to_label[key], "exact_match_after_prefix_removal"
    if key in ALIASES and ALIASES[key] in labels:
        return ALIASES[key], "alias_match_after_prefix_removal"

    # Containment: valid only if exactly one allowed category is present.
    text_norm = normalise_label_text(text)
    found = []
    for label in labels:
        pattern = re.escape(normalise_label_text(label))
        if re.search(pattern, text_norm):
            found.append(label)

    found = sorted(set(found))
    if len(found) == 1:
        return found[0], "contained_single_category"
    if len(found) > 1:
        return None, "multiple_categories_found"

    return None, "parse_failed"


def load_tokenizer(adapter_dir: Path, base_model: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(args: argparse.Namespace, adapter_dir: Path, use_bf16: bool):
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_rows_for_prediction(df: pd.DataFrame, args: argparse.Namespace) -> List[int]:
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")

    text_nonempty = df[args.text_col].fillna("").astype(str).str.strip() != ""

    if args.predict_all:
        mask = text_nonempty
    else:
        if args.risk_col not in df.columns:
            raise KeyError(
                f"Risk column not found: {args.risk_col}. Use --predict-all if the table has no existing label column."
            )
        risk_missing = df[args.risk_col].isna() | (df[args.risk_col].astype(str).str.strip() == "")
        mask = text_nonempty & risk_missing

    if args.start_row is not None:
        mask &= df.index >= args.start_row
    if args.end_row is not None:
        mask &= df.index < args.end_row

    indices = df.index[mask].tolist()
    if args.max_records and args.max_records > 0:
        indices = indices[: args.max_records]
    return [int(i) for i in indices]


def set_original_label_context(df: pd.DataFrame, args: argparse.Namespace):
    """Fill risk_for_control columns for rows with existing labels."""
    if args.risk_col not in df.columns:
        return

    original = df[args.risk_col].fillna("").astype(str).str.strip()
    has_original = original != ""
    df.loc[has_original, RISK_FOR_CONTROL_COL] = original.loc[has_original]
    df.loc[has_original, RISK_FOR_CONTROL_SOURCE_COL] = "existing_label"


def mark_need_review_for_empty_descriptions(df: pd.DataFrame, candidate_indices: Iterable[int], args: argparse.Namespace) -> Tuple[List[int], int]:
    valid_indices = []
    insufficient_count = 0
    for idx in candidate_indices:
        text = normalise_text(df.at[idx, args.text_col])
        if len(text) < args.min_description_chars:
            insufficient_count += 1
            df.at[idx, PRED_COL] = ""
            df.at[idx, RAW_COL] = ""
            df.at[idx, PARSE_COL] = "not_predicted_insufficient_description"
            df.at[idx, STATUS_COL] = "need_further_review"
            df.at[idx, PRIORITY_COL] = "high"
            df.at[idx, REASON_COL] = "insufficient_description"
            df.at[idx, SOURCE_COL] = DEFAULT_LABEL_SOURCE
            df.at[idx, RISK_FOR_CONTROL_COL] = ""
            df.at[idx, RISK_FOR_CONTROL_SOURCE_COL] = "need_further_review"
        else:
            valid_indices.append(idx)
    return valid_indices, insufficient_count


def apply_prediction_to_row(
    df: pd.DataFrame,
    idx: int,
    pred: Optional[str],
    raw: str,
    parse_status: str,
    args: argparse.Namespace,
):
    df.at[idx, PRED_COL] = pred or ""
    df.at[idx, RAW_COL] = normalise_text(raw)
    df.at[idx, PARSE_COL] = parse_status
    df.at[idx, SOURCE_COL] = DEFAULT_LABEL_SOURCE

    if pred is None:
        df.at[idx, STATUS_COL] = "need_further_review"
        df.at[idx, PRIORITY_COL] = "high"
        df.at[idx, REASON_COL] = parse_status
        df.at[idx, RISK_FOR_CONTROL_COL] = ""
        df.at[idx, RISK_FOR_CONTROL_SOURCE_COL] = "need_further_review"
    else:
        df.at[idx, STATUS_COL] = "auto_label_candidate"
        if args.flag_boundary_sensitive_categories and pred in BOUNDARY_SENSITIVE_CATEGORIES:
            df.at[idx, PRIORITY_COL] = "medium"
            df.at[idx, REASON_COL] = "boundary_sensitive_category"
        else:
            df.at[idx, PRIORITY_COL] = "low"
            df.at[idx, REASON_COL] = "valid_23class_prediction"
        df.at[idx, RISK_FOR_CONTROL_COL] = pred
        df.at[idx, RISK_FOR_CONTROL_SOURCE_COL] = "curated_model_candidate"


@torch.no_grad()
def predict_rows(
    df: pd.DataFrame,
    row_indices: List[int],
    args: argparse.Namespace,
    labels: List[str],
    model,
    tokenizer,
    output_dir: Path,
    partial_path: Path,
):
    device = get_model_device(model)
    total = len(row_indices)
    if total == 0:
        print("No rows selected for model prediction after pre-checks.")
        return

    num_batches = (total + args.eval_batch_size - 1) // args.eval_batch_size

    for batch_id, start in enumerate(tqdm(range(0, total, args.eval_batch_size), total=num_batches), start=1):
        end = min(start + args.eval_batch_size, total)
        batch_indices = row_indices[start:end]
        batch_texts = [normalise_text(df.at[idx, args.text_col]) for idx in batch_indices]
        batch_prompts = [build_prompt(tokenizer, text, labels) for text in batch_texts]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        input_len = encoded["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        raw_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for idx, raw in zip(batch_indices, raw_outputs):
            pred, status = parse_prediction(raw, labels)
            apply_prediction_to_row(df, idx, pred, raw, status, args)

        if args.checkpoint_every_batches > 0 and batch_id % args.checkpoint_every_batches == 0:
            df.to_csv(partial_path, index=False)
            print(f"[checkpoint] Saved partial predictions to {partial_path}", flush=True)


def summarise_and_save(df: pd.DataFrame, args: argparse.Namespace, labels: List[str], output_dir: Path, runtime_sec: float) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "risk_category_predictions_87k_curated.csv"
    partial_path = output_dir / "risk_category_predictions_87k_curated.partial.csv"
    distribution_path = output_dir / "predicted_risk_category_distribution_curated.csv"
    review_path = output_dir / "need_further_review_curated.csv"
    summary_path = output_dir / "risk_category_predictions_87k_curated_summary.json"

    df.to_csv(final_path, index=False)

    predicted_nonempty = df[PRED_COL].fillna("").astype(str).str.strip() != ""
    prediction_distribution = (
        df.loc[predicted_nonempty, PRED_COL]
        .value_counts(dropna=False)
        .rename_axis("predicted_risk_category")
        .reset_index(name="count")
    )
    prediction_distribution.to_csv(distribution_path, index=False)

    need_review = df[df[STATUS_COL] == "need_further_review"].copy()
    need_review.to_csv(review_path, index=False)

    parse_counts = df.loc[df[SOURCE_COL] == DEFAULT_LABEL_SOURCE, PARSE_COL].value_counts(dropna=False).to_dict()
    status_counts = df.loc[df[SOURCE_COL] == DEFAULT_LABEL_SOURCE, STATUS_COL].value_counts(dropna=False).to_dict()
    priority_counts = df.loc[df[SOURCE_COL] == DEFAULT_LABEL_SOURCE, PRIORITY_COL].value_counts(dropna=False).to_dict()
    reason_counts = df.loc[df[SOURCE_COL] == DEFAULT_LABEL_SOURCE, REASON_COL].value_counts(dropna=False).to_dict()

    risk_missing_total = None
    if args.risk_col in df.columns:
        risk_missing_total = int((df[args.risk_col].isna() | (df[args.risk_col].astype(str).str.strip() == "")).sum())

    summary = {
        "input_file": args.input_file,
        "output_dir": str(output_dir),
        "output_file": str(final_path),
        "partial_file": str(partial_path),
        "distribution_file": str(distribution_path),
        "need_further_review_file": str(review_path),
        "total_rows": int(len(df)),
        "risk_missing_rows_total": risk_missing_total,
        "rows_with_curated_model_source": int((df[SOURCE_COL] == DEFAULT_LABEL_SOURCE).sum()),
        "predicted_rows_nonempty": int(predicted_nonempty.sum()),
        "num_allowed_labels": int(len(labels)),
        "allowed_labels": labels,
        "parse_status_counts": {str(k): int(v) for k, v in parse_counts.items()},
        "review_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "review_priority_counts": {str(k): int(v) for k, v in priority_counts.items()},
        "review_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "prediction_distribution": {
            str(row["predicted_risk_category"]): int(row["count"])
            for _, row in prediction_distribution.iterrows()
        },
        "runtime_seconds": runtime_sec,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nPrediction summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    input_file = Path(args.input_file)
    label_list_json = Path(args.label_list_json)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not label_list_json.exists():
        raise FileNotFoundError(f"Label list JSON not found: {label_list_json}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    labels = load_labels(label_list_json)
    df = read_table(input_file)
    df = ensure_output_columns(df, overwrite=args.overwrite_existing_prediction_columns)
    set_original_label_context(df, args)

    candidate_indices = select_rows_for_prediction(df, args)
    valid_indices, insufficient_count = mark_need_review_for_empty_descriptions(df, candidate_indices, args)

    use_bf16 = bool(torch.cuda.is_available() and not args.no_bf16)

    setup = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "input_file": str(input_file),
        "label_list_json": str(label_list_json),
        "output_dir": str(output_dir),
        "text_col": args.text_col,
        "risk_col": args.risk_col,
        "predict_all": bool(args.predict_all),
        "selected_rows_before_description_filter": int(len(candidate_indices)),
        "insufficient_description_rows": int(insufficient_count),
        "rows_for_model_prediction": int(len(valid_indices)),
        "num_labels": int(len(labels)),
        "labels": labels,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "bf16": bool(use_bf16),
    }
    print("Curated full-table prediction setup:")
    print(json.dumps(setup, ensure_ascii=False, indent=2))

    start_time = time.time()
    partial_path = output_dir / "risk_category_predictions_87k_curated.partial.csv"

    model = None
    tokenizer = None
    try:
        tokenizer = load_tokenizer(adapter_dir, args.base_model)
        model = load_model(args, adapter_dir, use_bf16)
        predict_rows(df, valid_indices, args, labels, model, tokenizer, output_dir, partial_path)
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving partial predictions before exiting...")
        df.to_csv(partial_path, index=False)
        raise
    except Exception:
        print("\n[Error] Saving partial predictions before raising the exception...")
        df.to_csv(partial_path, index=False)
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runtime_sec = time.time() - start_time
    summarise_and_save(df, args, labels, output_dir, runtime_sec)


if __name__ == "__main__":
    main()
