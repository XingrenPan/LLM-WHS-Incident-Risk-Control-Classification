#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_generate_control_candidates_scoring_v2.py

Full definition-guided Control Category candidate generation using constrained likelihood scoring.

Why v2:
- The previous 17 script used free generation and was biased.
- The 18_v2 sanity evaluation showed constrained likelihood scoring was much more reliable.
- This script applies the same scoring method to the full table:
    existing valid Control labels -> keep unchanged
    missing Control labels + available risk definitions -> score all allowed controls and rank Top1/Top2/Top3
    missing Control labels + no risk definitions -> need_further_review

Important:
- Candidate labels are not ground truth.
- Keep source/status/reason fields for downstream review.
- Batch size is capped at 64 for this project.

Default inputs:
- Main file:
  outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated.csv
- Definitions:
  outputs/control_category_audit/control_definitions_extracted_v2/control_definitions_for_prompt.json
  outputs/control_category_audit/control_definitions_extracted_v2/risk_to_control_options.json

Main outputs:
- control_candidates_scoring_v2.csv
- control_candidates_scoring_v2.partial.csv
- need_further_review_scoring_v2.csv
- control_candidate_distribution_scoring_v2.csv
- control_candidate_summary_scoring_v2.json
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


def write_partial(df: pd.DataFrame, output_dir: Path) -> None:
    partial_path = output_dir / "control_candidates_scoring_v2.partial.csv"
    df.drop(columns=["_canonical_definition_risk", "_has_control_definitions_for_risk"], errors="ignore").to_csv(partial_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--main-file", default="outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated.csv")
    parser.add_argument("--definitions-json", default="outputs/control_category_audit/control_definitions_extracted_v2/control_definitions_for_prompt.json")
    parser.add_argument("--risk-options-json", default="outputs/control_category_audit/control_definitions_extracted_v2/risk_to_control_options.json")
    parser.add_argument("--output-dir", default="outputs/control_category_prediction/definition_guided_control_candidates_scoring_v2")

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
    parser.add_argument("--save-every", type=int, default=500)

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

    if not main_file.exists():
        raise FileNotFoundError(f"Main file not found: {main_file}")
    if not definitions_json.exists():
        raise FileNotFoundError(f"Definitions JSON not found: {definitions_json}")
    if not risk_options_json.exists():
        raise FileNotFoundError(f"Risk options JSON not found: {risk_options_json}")

    print("Loading data...")
    df = pd.read_csv(main_file, low_memory=False)
    definitions_by_risk, options_by_risk = load_definitions(definitions_json, risk_options_json)
    available_risks = list(definitions_by_risk.keys())

    for col in [args.text_col, args.risk_col, args.control_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    if args.location_col not in df.columns:
        df[args.location_col] = ""

    output_cols = [
        "control_scoring_v2_top1",
        "control_scoring_v2_top2",
        "control_scoring_v2_top3",
        "control_scoring_v2_top1_avg_nll",
        "control_scoring_v2_top2_avg_nll",
        "control_scoring_v2_top3_avg_nll",
        "control_scoring_v2_top1_relative_score",
        "control_scoring_v2_top2_relative_score",
        "control_scoring_v2_top3_relative_score",
        "control_scoring_v2_all_ranked_candidates_json",
        "control_scoring_v2_risk_used",
        "control_scoring_v2_options_count",
        "control_scoring_v2_review_status",
        "control_scoring_v2_review_priority",
        "control_scoring_v2_review_reason",
        "control_scoring_v2_label_source",
        "control_for_downstream_scoring_v2",
        "control_for_downstream_scoring_v2_source",
    ]
    for c in output_cols:
        if c not in df.columns:
            df[c] = pd.NA

    existing_valid = ~df[args.control_col].apply(is_missing_value)
    df.loc[existing_valid, "control_for_downstream_scoring_v2"] = df.loc[existing_valid, args.control_col].astype(str)
    df.loc[existing_valid, "control_for_downstream_scoring_v2_source"] = "existing_control_label"
    df.loc[existing_valid, "control_scoring_v2_label_source"] = "existing_control_label"
    df.loc[existing_valid, "control_scoring_v2_review_status"] = "existing_label"
    df.loc[existing_valid, "control_scoring_v2_review_priority"] = "none"
    df.loc[existing_valid, "control_scoring_v2_review_reason"] = "existing_valid_control_label"

    canonical_risks = []
    has_defs = []
    for x in df[args.risk_col].tolist():
        cr = canonicalize_risk(x, available_risks)
        canonical_risks.append(cr)
        has_defs.append(cr is not None and cr in definitions_by_risk and len(options_by_risk.get(cr, [])) > 0)
    df["_canonical_definition_risk"] = canonical_risks
    df["_has_control_definitions_for_risk"] = has_defs

    missing_control = ~existing_valid
    no_defs_mask = missing_control & (~df["_has_control_definitions_for_risk"])

    df.loc[no_defs_mask, "control_scoring_v2_review_status"] = "need_further_review"
    df.loc[no_defs_mask, "control_scoring_v2_review_priority"] = "high"
    df.loc[no_defs_mask, "control_scoring_v2_review_reason"] = "no_control_definitions_for_risk"
    df.loc[no_defs_mask, "control_scoring_v2_label_source"] = "not_predicted"
    df.loc[no_defs_mask, "control_for_downstream_scoring_v2_source"] = "need_further_review"

    candidate_mask = missing_control & df["_has_control_definitions_for_risk"]
    candidate_indices = df.index[candidate_mask].tolist()

    if args.max_records is not None:
        run_indices = candidate_indices[:args.max_records]
        skipped = candidate_indices[args.max_records:]
        if skipped:
            df.loc[skipped, "control_scoring_v2_review_status"] = "not_run_limited_mode"
            df.loc[skipped, "control_scoring_v2_review_priority"] = "none"
            df.loc[skipped, "control_scoring_v2_review_reason"] = "not_predicted_due_to_max_records"
            df.loc[skipped, "control_scoring_v2_label_source"] = "not_predicted"
            df.loc[skipped, "control_for_downstream_scoring_v2_source"] = "not_run_limited_mode"
    else:
        run_indices = candidate_indices

    setup = {
        "main_file": str(main_file),
        "definitions_json": str(definitions_json),
        "risk_options_json": str(risk_options_json),
        "output_dir": str(output_dir),
        "total_rows": int(len(df)),
        "existing_valid_control_rows": int(existing_valid.sum()),
        "missing_control_rows": int(missing_control.sum()),
        "missing_rows_with_definitions": int(candidate_mask.sum()),
        "missing_rows_without_definitions": int(no_defs_mask.sum()),
        "rows_to_score": int(len(run_indices)),
        "unique_definition_risks": int(len(available_risks)),
        "candidate_batch_size": args.candidate_batch_size,
        "definition_top_n": args.definition_top_n,
        "max_records": args.max_records,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir or None,
    }

    print("\nScoring-v2 Control candidate generation setup:")
    print(json.dumps(setup, indent=2, ensure_ascii=False))

    if len(run_indices) > 0:
        model, tokenizer = load_model(args)
        print("Model loaded.")

        start_time = time.time()
        scored_count = 0

        for idx in tqdm(run_indices, desc="Scoring missing control rows"):
            row = df.loc[idx]
            risk = row["_canonical_definition_risk"]
            options = options_by_risk.get(risk, [])
            defs = definitions_by_risk.get(risk, [])

            desc = norm_space(row.get(args.text_col, ""))
            loc = norm_space(row.get(args.location_col, ""))

            if not desc or is_missing_value(desc):
                df.at[idx, "control_scoring_v2_review_status"] = "need_further_review"
                df.at[idx, "control_scoring_v2_review_priority"] = "high"
                df.at[idx, "control_scoring_v2_review_reason"] = "insufficient_description"
                df.at[idx, "control_scoring_v2_label_source"] = "not_predicted"
                df.at[idx, "control_for_downstream_scoring_v2_source"] = "need_further_review"
                continue

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

            top1 = top[0]["candidate"]

            df.at[idx, "control_scoring_v2_top1"] = top[0]["candidate"]
            df.at[idx, "control_scoring_v2_top2"] = top[1]["candidate"]
            df.at[idx, "control_scoring_v2_top3"] = top[2]["candidate"]
            df.at[idx, "control_scoring_v2_top1_avg_nll"] = top[0]["avg_nll"]
            df.at[idx, "control_scoring_v2_top2_avg_nll"] = top[1]["avg_nll"]
            df.at[idx, "control_scoring_v2_top3_avg_nll"] = top[2]["avg_nll"]
            df.at[idx, "control_scoring_v2_top1_relative_score"] = top[0]["relative_score"]
            df.at[idx, "control_scoring_v2_top2_relative_score"] = top[1]["relative_score"]
            df.at[idx, "control_scoring_v2_top3_relative_score"] = top[2]["relative_score"]
            df.at[idx, "control_scoring_v2_all_ranked_candidates_json"] = json.dumps(ranked, ensure_ascii=False)
            df.at[idx, "control_scoring_v2_risk_used"] = risk
            df.at[idx, "control_scoring_v2_options_count"] = len(options)

            if top1:
                df.at[idx, "control_scoring_v2_review_status"] = "auto_label_candidate"
                df.at[idx, "control_scoring_v2_review_priority"] = "low"
                df.at[idx, "control_scoring_v2_review_reason"] = "valid_definition_guided_likelihood_ranking"
                df.at[idx, "control_scoring_v2_label_source"] = "definition_guided_likelihood_candidate"
                df.at[idx, "control_for_downstream_scoring_v2"] = top1
                df.at[idx, "control_for_downstream_scoring_v2_source"] = "definition_guided_likelihood_candidate"
            else:
                df.at[idx, "control_scoring_v2_review_status"] = "need_further_review"
                df.at[idx, "control_scoring_v2_review_priority"] = "high"
                df.at[idx, "control_scoring_v2_review_reason"] = "scoring_failed"
                df.at[idx, "control_scoring_v2_label_source"] = "not_predicted"
                df.at[idx, "control_for_downstream_scoring_v2_source"] = "need_further_review"

            scored_count += 1
            if args.save_every > 0 and scored_count % args.save_every == 0:
                write_partial(df, output_dir)

        runtime = time.time() - start_time
        print(f"\nScoring runtime seconds: {runtime:.2f}")

    # Write final outputs.
    full_df = df.drop(columns=["_canonical_definition_risk", "_has_control_definitions_for_risk"], errors="ignore").copy()

    full_path = output_dir / "control_candidates_scoring_v2.csv"
    full_df.to_csv(full_path, index=False)

    need_review = full_df[
        full_df["control_scoring_v2_review_status"].astype(str).isin(["need_further_review", "not_run_limited_mode"])
    ].copy()
    need_review_path = output_dir / "need_further_review_scoring_v2.csv"
    need_review.to_csv(need_review_path, index=False)

    dist = full_df["control_for_downstream_scoring_v2"].value_counts(dropna=False).reset_index()
    dist.columns = ["control_category", "count"]
    dist_path = output_dir / "control_candidate_distribution_scoring_v2.csv"
    dist.to_csv(dist_path, index=False)

    candidate_rows = full_df["control_scoring_v2_label_source"].eq("definition_guided_likelihood_candidate")

    summary = {
        **setup,
        "scored_rows_completed": int(candidate_rows.sum()),
        "review_status_counts_all_rows": safe_value_counts(full_df["control_scoring_v2_review_status"]),
        "review_priority_counts_all_rows": safe_value_counts(full_df["control_scoring_v2_review_priority"]),
        "label_source_counts_all_rows": safe_value_counts(full_df["control_for_downstream_scoring_v2_source"]),
        "top1_relative_score_summary": {
            "mean": float(pd.to_numeric(full_df.loc[candidate_rows, "control_scoring_v2_top1_relative_score"], errors="coerce").mean()) if int(candidate_rows.sum()) else None,
            "median": float(pd.to_numeric(full_df.loc[candidate_rows, "control_scoring_v2_top1_relative_score"], errors="coerce").median()) if int(candidate_rows.sum()) else None,
            "p10": float(pd.to_numeric(full_df.loc[candidate_rows, "control_scoring_v2_top1_relative_score"], errors="coerce").quantile(0.10)) if int(candidate_rows.sum()) else None,
            "p90": float(pd.to_numeric(full_df.loc[candidate_rows, "control_scoring_v2_top1_relative_score"], errors="coerce").quantile(0.90)) if int(candidate_rows.sum()) else None,
        },
        "candidate_distribution_top_30": dist.head(30).to_dict(orient="records"),
        "output_files": {
            "full_predictions_csv": str(full_path),
            "need_further_review_csv": str(need_review_path),
            "candidate_distribution_csv": str(dist_path),
            "partial_csv": str(output_dir / "control_candidates_scoring_v2.partial.csv"),
        },
    }

    summary_path = output_dir / "control_candidate_summary_scoring_v2.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Scoring-v2 outputs saved to:")
    print(output_dir)
    print("\nKey summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
