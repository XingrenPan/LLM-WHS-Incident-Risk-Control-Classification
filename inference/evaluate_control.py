#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_evaluate_definition_guided_control_scoring_existing_labels_v2.py

Fixed constrained likelihood-ranking evaluation for definition-guided Control Category labelling.

Fix compared with previous scoring script:
- The previous version could split one row's candidate options across multiple flushes.
- That caused duplicated evaluated rows, e.g. evaluated_rows > existing labelled rows.
- This v2 scores one source row at a time, while internally batching candidate controls.
- Therefore evaluated_rows should equal evaluation_rows_with_definitions.

Batch policy:
- Default candidate batch size is 64.
- Do not set candidate batch size above 64 unless explicitly needed.

Outputs:
- scoring_existing_label_predictions.csv
- scoring_existing_label_summary.json
- scoring_existing_label_classification_report.csv
- scoring_existing_label_confusion_pairs.csv
- scoring_existing_label_per_risk_performance.csv
- scoring_existing_label_mismatches.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    from sklearn.metrics import classification_report, f1_score
except Exception:
    classification_report = None
    f1_score = None


MISSING_PLACEHOLDERS = {
    "",
    "-",
    "--",
    "0",
    "0.0",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "missing",
    "unknown",
    "not applicable",
}


def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip()


def norm_key(s: Any) -> str:
    s = norm_space(s).lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return norm_space(s)


def is_missing_value(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return norm_space(x).lower() in MISSING_PLACEHOLDERS


def truncate_text(text: str, max_chars: int) -> str:
    text = norm_space(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def keyword_tokens(text: str) -> set:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", str(text).lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "are", "was", "were", "from",
        "control", "controls", "risk", "critical", "scope", "objective", "applies",
        "performance", "criteria", "operating", "activities", "normal", "conditions",
        "personnel", "work", "working", "used", "use", "ensure", "required",
        "requirements", "completed", "includes", "include", "including", "where",
        "relevant", "operations", "olympic", "dam", "bhp",
    }
    return {t for t in toks if t not in stop and len(t) > 2}


def shortlist_definitions(row_text: str, definitions: List[dict], top_n: int) -> List[dict]:
    if len(definitions) <= top_n:
        return definitions

    row_toks = keyword_tokens(row_text)
    scored = []
    for d in definitions:
        hay = f"{d.get('control_name','')} {d.get('definition','')}"
        toks = keyword_tokens(hay)
        overlap = len(row_toks & toks)
        name_toks = keyword_tokens(str(d.get("control_name", "")))
        overlap += 2 * len(row_toks & name_toks)
        scored.append((overlap, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_n]]


def canonicalize_risk(risk: Any, available_risks: List[str]) -> Optional[str]:
    if is_missing_value(risk):
        return None

    risk_norm = norm_key(risk)
    available_by_norm = {norm_key(r): r for r in available_risks}

    if risk_norm in available_by_norm:
        return available_by_norm[risk_norm]

    aliases = {
        norm_key("Non-process Fire and Explosion (obs)"): "Non Process Fire & Explosion",
        norm_key("Non-process Fire and Explosion"): "Non Process Fire & Explosion",
        norm_key("Non Process Fire and Explosion"): "Non Process Fire & Explosion",
        norm_key("Vehicles and Mobile Equipment"): "Vehicles & Mobile Equipment",
        norm_key("Fall from Height"): "Fall from height",
        norm_key("Explosives and blasting"): None,
        norm_key("Occupational Safety"): None,
        norm_key("Mental Health"): None,
        norm_key("Other unspecified"): None,
    }

    if risk_norm in aliases:
        mapped = aliases[risk_norm]
        if mapped and norm_key(mapped) in available_by_norm:
            return available_by_norm[norm_key(mapped)]
        return None

    for r in available_risks:
        rk = norm_key(r)
        if risk_norm == rk or risk_norm in rk or rk in risk_norm:
            return r
    return None


CONTROL_CANONICAL_ALIASES = {
    norm_key("Earth Bunds, Signage and Truck loader stops"): "Earth Bunds, Signage and Truck loader stops",
    norm_key("Earth Bunds, Signage & Tipples"): "Earth Bunds, Signage and Truck loader stops",
    norm_key("EARTH BUNDS, SIGNAGE & TIPPLES"): "Earth Bunds, Signage and Truck loader stops",

    norm_key("Scaffolding and Portable Ladders"): "Scaffolding and Portable Ladders",
    norm_key("Scaff and Portable Ladders"): "Scaffolding and Portable Ladders",
    norm_key("SCAFF AND PORTABLE LADDERS"): "Scaffolding and Portable Ladders",

    norm_key("Structural Integrity Inspect"): "Structural Integrity Inspect",
    norm_key("STRUCTURAL INTEGRITY INSPECT"): "Structural Integrity Inspect",

    norm_key("Fall Prevention Systems and Rope Access"): "Fall Prevention Systems and Rope Access",
    norm_key("WAH & Rope Access Mine"): "Fall Prevention Systems and Rope Access",
    norm_key("WAH & Rope Access Surface"): "Fall Prevention Systems and Rope Access",
    norm_key("WAH and Rope Access Mine"): "Fall Prevention Systems and Rope Access",
    norm_key("WAH and Rope Access Surface"): "Fall Prevention Systems and Rope Access",

    norm_key("Mobile Working at Heights Equipment"): "Mobile Working at Heights Equipment",
    norm_key("Operations and Maintenance of Mobile WAH Equip"): "Mobile Working at Heights Equipment",
    norm_key("Operations and Maintenance of Mobile WAH Equipment"): "Mobile Working at Heights Equipment",
    norm_key("OP & MAINT MOBILE WAH EQUIP"): "Mobile Working at Heights Equipment",
    norm_key("OP & MAINT OF MOBILE WAH EQUIP"): "Mobile Working at Heights Equipment",
    norm_key("OP AND MAINT MOBILE WAH EQUIP"): "Mobile Working at Heights Equipment",
    norm_key("OP AND MAINT OF MOBILE WAH EQUIP"): "Mobile Working at Heights Equipment",

    norm_key("Fitness for Work"): "Fitness for Work",
    norm_key("FITNESS FOR WORK"): "Fitness for Work",
    norm_key("Crisis and Emergency Management"): "Crisis and Emergency Management",
    norm_key("CRISIS AND EMERGENCY MANAGEMENT"): "Crisis and Emergency Management",
}


def title_case_control(s: Any) -> str:
    s = norm_space(s)
    if not s:
        return s
    if re.search(r"[a-z]", s):
        return s
    return s.title()


def canonical_control_label(label: Any) -> Optional[str]:
    if is_missing_value(label):
        return None
    k = norm_key(label)
    if k in CONTROL_CANONICAL_ALIASES:
        return CONTROL_CANONICAL_ALIASES[k]
    return title_case_control(label)


def load_definitions(definitions_json: Path, risk_options_json: Path) -> Tuple[Dict[str, List[dict]], Dict[str, List[str]]]:
    with definitions_json.open("r", encoding="utf-8") as f:
        defs = json.load(f)
    with risk_options_json.open("r", encoding="utf-8") as f:
        opts = json.load(f)
    return {str(k): v for k, v in defs.items()}, {str(k): v for k, v in opts.items()}


def make_scoring_prompt(
    description: str,
    location: str,
    risk_title: str,
    control_options: List[str],
    control_definitions: List[dict],
    definition_top_n: int,
    max_definition_chars: int,
    max_description_chars: int,
) -> str:
    description = truncate_text(description, max_description_chars)
    location = truncate_text(location, 220) if not is_missing_value(location) else "Unknown"

    row_text = f"{description} {location}"
    shortlisted = shortlist_definitions(row_text, control_definitions, top_n=definition_top_n)

    options_block = "\n".join([f"- {name}" for name in control_options])
    def_lines = []
    for d in shortlisted:
        name = norm_space(d.get("control_name", ""))
        definition = truncate_text(norm_space(d.get("definition", "")), max_definition_chars)
        if name:
            def_lines.append(f"- {name}: {definition}")
    definitions_block = "\n".join(def_lines)

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict occupational safety control-category classifier.
Choose only one exact Control Category from the allowed list.
Use the incident description, location, and control definitions.
Do not invent categories.<|eot_id|><|start_header_id|>user<|end_header_id|>
Risk Category:
{risk_title}

Location:
{location}

Incident Description:
{description}

Allowed Control Categories:
{options_block}

Relevant Control Definitions:
{definitions_block}

Question:
Which single Control Category best matches this incident?

Answer with only the exact Control Category name.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

    if args.adapter_dir:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but --adapter-dir was provided.")
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    return model, tokenizer


def build_scoring_sequence(tokenizer, prompt: str, candidate: str, max_input_tokens: int):
    answer = candidate.strip()
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cand_ids = tokenizer(answer, add_special_tokens=False).input_ids

    max_prompt_len = max_input_tokens - len(cand_ids) - 1
    if max_prompt_len < 32:
        raise ValueError("Candidate plus prompt exceeds max_input_tokens. Increase max_input_tokens or reduce definitions.")
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + cand_ids
    labels = [-100] * len(prompt_ids) + cand_ids
    return input_ids, labels


@torch.inference_mode()
def score_candidate_batch(model, tokenizer, items: List[Tuple[str, str]], args) -> List[float]:
    seqs = []
    max_len = 0
    for prompt, cand in items:
        ids, labels = build_scoring_sequence(tokenizer, prompt, cand, args.max_input_tokens)
        seqs.append((ids, labels))
        max_len = max(max_len, len(ids))

    pad_id = tokenizer.pad_token_id
    input_batch = []
    label_batch = []
    attn_batch = []

    for ids, labels in seqs:
        pad_len = max_len - len(ids)
        input_batch.append(ids + [pad_id] * pad_len)
        label_batch.append(labels + [-100] * pad_len)
        attn_batch.append([1] * len(ids) + [0] * pad_len)

    device = model.device
    input_ids = torch.tensor(input_batch, dtype=torch.long, device=device)
    labels = torch.tensor(label_batch, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_batch, dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.shape[-1]
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss = loss_flat.view(shift_labels.shape)

    mask = (shift_labels != -100).float()
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    avg_nll = (loss * mask).sum(dim=1) / token_counts

    return avg_nll.detach().cpu().float().tolist()


def relative_scores_from_nll(nlls: List[float]) -> List[float]:
    vals = torch.tensor([-x for x in nlls], dtype=torch.float32)
    probs = torch.softmax(vals, dim=0)
    return probs.tolist()


def score_one_row(prompt: str, options: List[str], model, tokenizer, args) -> List[dict]:
    all_scores = []

    for start in range(0, len(options), args.candidate_batch_size):
        batch_options = options[start:start + args.candidate_batch_size]
        items = [(prompt, cand) for cand in batch_options]
        nlls = score_candidate_batch(model, tokenizer, items, args)
        for cand, nll in zip(batch_options, nlls):
            all_scores.append({
                "candidate": cand,
                "candidate_canonical": canonical_control_label(cand),
                "avg_nll": float(nll),
            })

    all_scores.sort(key=lambda x: x["avg_nll"])
    rel_scores = relative_scores_from_nll([x["avg_nll"] for x in all_scores])
    for rank, (x, rel) in enumerate(zip(all_scores, rel_scores), start=1):
        x["rank"] = rank
        x["relative_score"] = float(rel)

    return all_scores


def safe_value_counts(series: pd.Series) -> List[dict]:
    vc = series.value_counts(dropna=False)
    out = []
    for k, v in vc.items():
        if isinstance(k, float) and math.isnan(k):
            key = None
        else:
            key = str(k)
        out.append({"value": key, "count": int(v)})
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--main-file", default="outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated.csv")
    parser.add_argument("--definitions-json", default="outputs/control_category_audit/control_definitions_extracted_v2/control_definitions_for_prompt.json")
    parser.add_argument("--risk-options-json", default="outputs/control_category_audit/control_definitions_extracted_v2/risk_to_control_options.json")
    parser.add_argument("--output-dir", default="outputs/control_category_prediction/definition_guided_control_candidates/evaluation_existing_labels_scoring_v2")

    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter-dir", default="", help="Optional LoRA adapter. Default: base instruct model only.")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")

    parser.add_argument("--text-col", default="WHAT_HAPPENED_ENGLISH")
    parser.add_argument("--location-col", default="Location")
    parser.add_argument("--risk-col", default="risk_for_control_curated")
    parser.add_argument("--control-col", default="What Control Failed")

    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=64)

    parser.add_argument("--definition-top-n", type=int, default=12)
    parser.add_argument("--max-definition-chars", type=int, default=420)
    parser.add_argument("--max-description-chars", type=int, default=900)
    parser.add_argument("--max-input-tokens", type=int, default=4096)

    args = parser.parse_args()

    if args.candidate_batch_size > 64:
        raise ValueError("For this project, --candidate-batch-size should be <= 64.")

    main_file = Path(args.main_file)
    definitions_json = Path(args.definitions_json)
    risk_options_json = Path(args.risk_options_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(main_file, low_memory=False)
    definitions_by_risk, options_by_risk = load_definitions(definitions_json, risk_options_json)
    available_risks = list(definitions_by_risk.keys())

    for col in [args.text_col, args.risk_col, args.control_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    if args.location_col not in df.columns:
        df[args.location_col] = ""

    existing_valid = ~df[args.control_col].apply(is_missing_value)
    eval_df = df.loc[existing_valid].copy()
    eval_df["_canonical_definition_risk"] = [
        canonicalize_risk(x, available_risks) for x in eval_df[args.risk_col].tolist()
    ]
    eval_df = eval_df[eval_df["_canonical_definition_risk"].notna()].copy()
    eval_df["true_control_canonical"] = eval_df[args.control_col].apply(canonical_control_label)
    eval_df = eval_df[eval_df["true_control_canonical"].notna()].copy()
    eval_df = eval_df[
        eval_df["_canonical_definition_risk"].apply(lambda r: r in definitions_by_risk and len(options_by_risk.get(r, [])) > 0)
    ].copy()

    if args.max_records is not None:
        eval_df = eval_df.head(args.max_records).copy()

    setup = {
        "main_file": str(main_file),
        "definitions_json": str(definitions_json),
        "risk_options_json": str(risk_options_json),
        "output_dir": str(output_dir),
        "total_rows": int(len(df)),
        "existing_valid_control_rows": int(existing_valid.sum()),
        "evaluation_rows_with_definitions": int(len(eval_df)),
        "max_records": args.max_records,
        "candidate_batch_size": args.candidate_batch_size,
        "definition_top_n": args.definition_top_n,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir or None,
    }

    print("\nFixed likelihood-ranking evaluation setup:")
    print(json.dumps(setup, indent=2, ensure_ascii=False))

    if len(eval_df) == 0:
        raise RuntimeError("No existing-labelled rows with matching definitions were found.")

    model, tokenizer = load_model(args)
    print("Model loaded.")

    rows = []
    start_time = time.time()

    for source_index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Scoring existing labels"):
        risk = row["_canonical_definition_risk"]
        options = options_by_risk.get(risk, [])
        defs = definitions_by_risk.get(risk, [])

        desc = norm_space(row.get(args.text_col, ""))
        loc = norm_space(row.get(args.location_col, ""))
        true_raw = norm_space(row.get(args.control_col, ""))
        true_canon = canonical_control_label(true_raw)

        if not desc or is_missing_value(desc) or not options:
            ranked = []
        else:
            prompt = make_scoring_prompt(
                description=desc,
                location=loc,
                risk_title=risk,
                control_options=options,
                control_definitions=defs,
                definition_top_n=args.definition_top_n,
                max_definition_chars=args.max_definition_chars,
                max_description_chars=args.max_description_chars,
            )
            ranked = score_one_row(prompt, options, model, tokenizer, args)

        top = ranked[:3]
        while len(top) < 3:
            top.append({
                "candidate": None,
                "candidate_canonical": None,
                "avg_nll": None,
                "relative_score": None,
            })

        pred1_canon = top[0]["candidate_canonical"]
        pred2_canon = top[1]["candidate_canonical"]
        pred3_canon = top[2]["candidate_canonical"]

        rows.append({
            "source_index": source_index,
            "event_number": row.get("Event Number", ""),
            "risk_used": risk,
            "description": desc,
            "location": loc,
            "true_control_raw": true_raw,
            "true_control_canonical": true_canon,
            "pred_top1_raw": top[0]["candidate"],
            "pred_top2_raw": top[1]["candidate"],
            "pred_top3_raw": top[2]["candidate"],
            "pred_top1_canonical": pred1_canon,
            "pred_top2_canonical": pred2_canon,
            "pred_top3_canonical": pred3_canon,
            "top1_avg_nll": top[0]["avg_nll"],
            "top2_avg_nll": top[1]["avg_nll"],
            "top3_avg_nll": top[2]["avg_nll"],
            "top1_relative_score": top[0]["relative_score"],
            "top2_relative_score": top[1]["relative_score"],
            "top3_relative_score": top[2]["relative_score"],
            "top1_raw_exact_match": norm_key(top[0]["candidate"]) == norm_key(true_raw) if top[0]["candidate"] else False,
            "top1_compatible_match": pred1_canon == true_canon if pred1_canon and true_canon else False,
            "top3_compatible_match": true_canon in [pred1_canon, pred2_canon, pred3_canon],
            "options_count": len(options),
            "all_ranked_candidates_json": json.dumps(ranked, ensure_ascii=False),
        })

    runtime = time.time() - start_time
    pred_df = pd.DataFrame(rows)

    predictions_path = output_dir / "scoring_existing_label_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    mismatches = pred_df[~pred_df["top1_compatible_match"].fillna(False)].copy()
    mismatches_path = output_dir / "scoring_existing_label_mismatches.csv"
    mismatches.to_csv(mismatches_path, index=False)

    top1_raw_exact = float(pred_df["top1_raw_exact_match"].mean()) if len(pred_df) else 0.0
    top1_compatible = float(pred_df["top1_compatible_match"].mean()) if len(pred_df) else 0.0
    top3_compatible = float(pred_df["top3_compatible_match"].mean()) if len(pred_df) else 0.0

    valid = pred_df[pred_df["pred_top1_canonical"].notna()].copy()
    report_path = output_dir / "scoring_existing_label_classification_report.csv"
    if classification_report is not None and len(valid):
        labels = sorted(set(valid["true_control_canonical"].dropna()) | set(valid["pred_top1_canonical"].dropna()))
        report = classification_report(
            valid["true_control_canonical"],
            valid["pred_top1_canonical"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
        report_df.to_csv(report_path, index=False)

        macro_f1 = float(f1_score(
            valid["true_control_canonical"],
            valid["pred_top1_canonical"],
            average="macro",
            zero_division=0,
        ))
        weighted_f1 = float(f1_score(
            valid["true_control_canonical"],
            valid["pred_top1_canonical"],
            average="weighted",
            zero_division=0,
        ))
    else:
        pd.DataFrame().to_csv(report_path, index=False)
        macro_f1 = None
        weighted_f1 = None

    confusion = (
        pred_df[pred_df["true_control_canonical"].notna()]
        .groupby(["true_control_canonical", "pred_top1_canonical"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion_path = output_dir / "scoring_existing_label_confusion_pairs.csv"
    confusion.to_csv(confusion_path, index=False)

    per_risk_rows = []
    for risk, sub in pred_df.groupby("risk_used", dropna=False):
        per_risk_rows.append({
            "risk_used": risk,
            "rows": int(len(sub)),
            "top1_raw_exact_accuracy": float(sub["top1_raw_exact_match"].mean()) if len(sub) else 0.0,
            "top1_compatible_accuracy": float(sub["top1_compatible_match"].mean()) if len(sub) else 0.0,
            "top3_compatible_accuracy": float(sub["top3_compatible_match"].mean()) if len(sub) else 0.0,
        })
    per_risk_df = pd.DataFrame(per_risk_rows).sort_values("rows", ascending=False)
    per_risk_path = output_dir / "scoring_existing_label_per_risk_performance.csv"
    per_risk_df.to_csv(per_risk_path, index=False)

    summary = {
        **setup,
        "evaluation_runtime_seconds": runtime,
        "evaluated_rows": int(len(pred_df)),
        "row_count_check_passed": int(len(pred_df)) == int(len(eval_df)),
        "top1_raw_exact_accuracy": top1_raw_exact,
        "top1_compatible_accuracy": top1_compatible,
        "top3_compatible_accuracy": top3_compatible,
        "macro_f1_compatible_labels": macro_f1,
        "weighted_f1_compatible_labels": weighted_f1,
        "mismatch_rows_top1_compatible": int((~pred_df["top1_compatible_match"].fillna(False)).sum()),
        "top3_miss_rows_compatible": int((~pred_df["top3_compatible_match"].fillna(False)).sum()),
        "true_label_distribution": pred_df["true_control_canonical"].value_counts(dropna=False).to_dict(),
        "pred_top1_distribution": pred_df["pred_top1_canonical"].value_counts(dropna=False).to_dict(),
        "output_files": {
            "predictions_csv": str(predictions_path),
            "mismatches_csv": str(mismatches_path),
            "classification_report_csv": str(report_path),
            "confusion_pairs_csv": str(confusion_path),
            "per_risk_performance_csv": str(per_risk_path),
        },
    }

    summary_path = output_dir / "scoring_existing_label_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Fixed likelihood-ranking evaluation outputs saved to:")
    print(output_dir)
    print("\nKey summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
