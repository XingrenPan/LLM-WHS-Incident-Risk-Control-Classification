#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-description Risk Category inference with the curated 23-class
Llama-3.1-8B-Instruct QLoRA adapter.

Purpose:
  - Input one incident description.
  - Rank all allowed Risk Categories using label likelihood scoring.
  - Return Top-1 / Top-K candidate categories.
  - Generate one concise evidence-based reason for the Top-1 category.

Default inputs:
  data/risk_description_only_main_classes_curated_23class/risk_label_list.json
  models/llama31_8b_risk_description_only_main_classes_curated_23class

Example:
  python 09_predict_single_description_topk_reason.py \
    --description "Worker slipped while climbing down a ladder and nearly fell from the platform." \
    --top-k 3

Notes:
  - Candidate scores are relative ranking scores, not calibrated probabilities.
  - The original dataset labels are not modified by this script.
  - The script uses neutral curated terminology and contains no person-specific wording.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel


DEFAULT_LABEL_LIST_JSON = "data/risk_description_only_main_classes_curated_23class/risk_label_list.json"
DEFAULT_ADAPTER_DIR = "models/llama31_8b_risk_description_only_main_classes_curated_23class"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_JSON = "outputs/risk_category_prediction_curated_23class/single_description_prediction.json"
DEFAULT_LABEL_SOURCE = "llama31_8b_risk_description_only_main_classes_curated_23class"


# Safety-critical evidence guardrails.
# These rules do not override the model's Top-1 prediction by default.
# They mark the case for manual review when important evidence is present but
# the corresponding safety-critical category is not ranked highly enough.
SAFETY_GUARDRAIL_CATEGORY = "Confined Space"
CONFINED_SPACE_STRONG_PATTERNS = [
    r"\bconfined\s+space\b",
    r"\bconfined\s+space\s+entry\b",
    r"\btank\s+entry\b",
    r"\bentered\s+(?:a\s+|the\s+)?tank\b",
    r"\benter(?:ed|ing)?\s+(?:a\s+|the\s+)?tank\b",
    r"\binside\s+(?:a\s+|the\s+)?tank\b",
    r"\btank\s+(?:for\s+)?inspection\b",
    r"\bvessel\s+entry\b",
    r"\bentered\s+(?:a\s+|the\s+)?vessel\b",
    r"\bmanhole\b",
    r"\bgas\s+test\b",
    r"\batmospheric\s+test\b",
    r"\batmosphere\s+test\b",
    r"\blow\s+oxygen\b",
    r"\boxygen\s+alarm\b",
    r"\boxygen\s+deficien(?:t|cy)\b",
    r"\bentry\s+permit\b",
]


CATEGORY_CUES: Dict[str, List[str]] = {
    "Fall from height": [
        "ladder", "scaffold", "scaffolding", "platform", "height", "heights", "open edge",
        "open hole", "open void", "handrail", "guardrail", "harness", "fall arrest", "ewp",
        "roof", "stairs", "stair", "pit", "grid mesh", "elevated", "working at height",
    ],
    "Dropped / Falling Object": [
        "dropped", "falling object", "fell from", "object fell", "falling material", "dropped object",
        "overhead", "struck by", "tool fell", "rock fall", "rockfall",
    ],
    "Vehicles & Mobile Equipment": [
        "vehicle", "truck", "light vehicle", "lv", "loader", "dozer", "excavator", "haul truck",
        "forklift", "mobile equipment", "collision", "traffic", "reverse", "reversing", "park brake",
    ],
    "Electrical (incl. Arc Flash/Blast)": [
        "electrical", "electric", "arc flash", "arc blast", "energised", "energized", "voltage",
        "cable", "powerline", "switchboard", "isolator", "shock", "breaker",
    ],
    "Energy release (excl. Electrical)": [
        "energy release", "pressure", "pressurised", "pressurized", "hydraulic", "pneumatic",
        "stored energy", "uncontrolled release", "burst", "hose", "line of fire",
    ],
    "Loss of Containment": [
        "spill", "leak", "leaking", "release", "overflow", "loss of containment", "rupture",
        "chemical spill", "fuel spill", "oil spill", "tank", "pipe leak",
    ],
    "Engulfment / inrush": [
        "engulfment", "inrush", "buried", "trapped by material", "mud rush", "water inrush",
        "silo", "stockpile", "material flow", "drawpoint",
    ],
    "Geotechnical Stability": [
        "wall failure", "slope", "bench", "ground failure", "geotechnical", "rockfall",
        "pit wall", "collapse", "subsidence", "highwall", "instability",
    ],
    "Entanglement / crushing": [
        "entangled", "entanglement", "crush", "crushed", "pinch point", "caught", "caught between",
        "conveyor", "unguarded", "nip point", "moving parts",
    ],
    "Lifting": [
        "lift", "lifting", "crane", "sling", "rigging", "dogging", "load", "suspended load",
        "chain block", "hoist", "forklift lift",
    ],
    "Confined Space": [
        "confined space", "tank entry", "vessel", "manhole", "oxygen", "atmosphere", "gas test",
        "entry permit", "enclosed space",
    ],
    "Explosives and blasting": [
        "blast", "blasting", "explosive", "detonator", "misfire", "shotfirer", "charge", "magazine",
    ],
    "Non Process Fire & Explosion": [
        "fire", "smoke", "flame", "explosion", "combustion", "thermal event", "hot work",
        "vehicle fire", "building fire",
    ],
    "Non-process Fire and Explosion (obs)": [
        "observation", "obs", "fire observation", "explosion observation",
    ],
    "Process Safety": [
        "process safety", "plant upset", "process control", "process hazard", "process incident",
        "operating envelope", "critical control",
    ],
    "Asset Integrity": [
        "asset integrity", "corrosion", "crack", "structural", "failure", "inspection", "degradation",
        "defect", "worn", "fatigue", "pipe support",
    ],
    "Occupational Safety": [
        "slip", "trip", "housekeeping", "walkway", "manual handling", "cut", "laceration", "sprain",
        "ergonomic", "pothole", "uneven", "stumble", "floor", "ankle",
    ],
    "Acute Chemical Exposure": [
        "chemical exposure", "splash", "acid", "caustic", "fume", "gas exposure", "inhaled", "skin contact",
        "eye contact", "msds", "sds", "irritation",
    ],
    "Carcinogen Exposure": [
        "carcinogen", "asbestos", "silica", "diesel particulate", "benzene", "radiation", "fibres",
        "respirable dust", "crystalline silica",
    ],
    "Mental Health": [
        "mental health", "stress", "fatigue", "bullying", "harassment", "psychological", "anxiety",
        "wellbeing", "trauma", "threatening behaviour", "threatening behavior",
    ],
    "Physical Health": [
        "illness", "heat stress", "dehydration", "medical", "pain", "strain", "muscle", "neck pain",
        "back pain", "health", "injury", "odv",
    ],
    "Aviation": [
        "aircraft", "helicopter", "aviation", "flight", "pilot", "helipad", "airstrip", "drone", "uav",
    ],
    "Other unspecified": [
        "other", "unspecified", "unclear", "not specified", "unknown",
    ],
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
        description="Predict Top-K Risk Categories and a concise reason for a single incident description."
    )

    parser.add_argument("--description", default="", help="Incident description text.")
    parser.add_argument("--description-file", default="", help="Optional text file containing one description.")
    parser.add_argument("--input-csv", default="", help="Optional CSV file for batch single-description style inference.")
    parser.add_argument("--text-col", default="WHAT_HAPPENED_ENGLISH", help="Text column when --input-csv is used.")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path when --input-csv is used.")

    parser.add_argument("--label-list-json", default=DEFAULT_LABEL_LIST_JSON)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--score-batch-size", type=int, default=23)
    parser.add_argument("--reason-mode", choices=["model", "evidence", "none"], default="model")
    parser.add_argument("--max-reason-words", type=int, default=25)
    parser.add_argument("--reason-max-new-tokens", type=int, default=48)
    parser.add_argument("--min-description-chars", type=int, default=8)
    parser.add_argument("--medium-score-threshold", type=float, default=0.45)
    parser.add_argument("--medium-margin-threshold", type=float, default=0.10)
    parser.add_argument("--guardrail-top-n", type=int, default=3, help="Mark for review if a safety-critical category evidence is detected but the category is not in the top N ranked labels.")
    parser.add_argument("--disable-safety-guardrails", action="store_true", help="Disable evidence-based safety guardrails.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--print-prompt", action="store_true", help="Print the classification prompt for debugging.")

    return parser.parse_args()


def normalise_text(s: object) -> str:
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s.strip())


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


def build_classification_prompt(tokenizer, text: str, labels: List[str]) -> str:
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


def build_reason_prompt(tokenizer, text: str, top1: str, top_candidates: List[Dict], max_words: int) -> str:
    alternatives = ", ".join([x["category"] for x in top_candidates[1:]]) if len(top_candidates) > 1 else "None"
    system_msg = (
        "You are a safety incident classification assistant. "
        "Give concise evidence-based reasons only. "
        "Do not introduce facts that are not present in the incident description."
    )
    user_msg = (
        "Incident description:\n"
        f"{text}\n\n"
        f"Predicted Risk Category: {top1}\n"
        f"Alternative candidate categories: {alternatives}\n\n"
        f"Give one concise reason for the predicted category in no more than {max_words} words. "
        "Return only the reason sentence."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def load_tokenizer(adapter_dir: Path, base_model: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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


def truncate_prompt_ids(prompt_ids: List[int], answer_len: int, max_length: int) -> List[int]:
    max_prompt_len = max(1, max_length - answer_len)
    # Keep the beginning of the prompt to match the training script's truncation behavior.
    return prompt_ids[:max_prompt_len]


@torch.no_grad()
def score_labels(
    model,
    tokenizer,
    description: str,
    labels: List[str],
    max_length: int,
    score_batch_size: int,
) -> List[Dict]:
    device = get_model_device(model)
    prompt = build_classification_prompt(tokenizer, description, labels)
    prompt_ids_full = tokenizer.encode(prompt, add_special_tokens=False)

    examples = []
    for label in labels:
        # A leading space is intentionally not added because the model was trained with exact labels.
        answer_ids = tokenizer.encode(label, add_special_tokens=False)
        prompt_ids = truncate_prompt_ids(prompt_ids_full, len(answer_ids), max_length)
        input_ids = prompt_ids + answer_ids
        loss_labels = [-100] * len(prompt_ids) + answer_ids
        examples.append({
            "category": label,
            "input_ids": input_ids,
            "loss_labels": loss_labels,
            "answer_token_count": len(answer_ids),
        })

    results = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for start in range(0, len(examples), score_batch_size):
        batch = examples[start:start + score_batch_size]
        max_len = max(len(x["input_ids"]) for x in batch)
        input_batch = []
        label_batch = []
        attention_batch = []

        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_batch.append(ex["input_ids"] + [pad_id] * pad_len)
            label_batch.append(ex["loss_labels"] + [-100] * pad_len)
            attention_batch.append([1] * len(ex["input_ids"]) + [0] * pad_len)

        input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
        label_tensor = torch.tensor(label_batch, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attention_batch, dtype=torch.long, device=device)

        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = label_tensor[:, 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        token_losses = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        valid_mask = shift_labels.ne(-100)
        loss_sums = (token_losses * valid_mask).sum(dim=1)
        token_counts = valid_mask.sum(dim=1).clamp(min=1)
        mean_losses = loss_sums / token_counts

        for ex, mean_loss, loss_sum in zip(batch, mean_losses.detach().cpu().tolist(), loss_sums.detach().cpu().tolist()):
            results.append({
                "category": ex["category"],
                "mean_nll": float(mean_loss),
                "sum_nll": float(loss_sum),
                "answer_token_count": int(ex["answer_token_count"]),
            })

    # Convert mean NLL into relative scores. These are not calibrated probabilities.
    scores = torch.tensor([-x["mean_nll"] for x in results], dtype=torch.float32)
    probs = torch.softmax(scores, dim=0).tolist()
    for item, prob in zip(results, probs):
        item["relative_score"] = float(prob)

    results.sort(key=lambda x: x["relative_score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


def find_evidence_cues(description: str, category: str, max_cues: int = 3) -> List[str]:
    text = normalise_text(description).lower()
    cues = []
    for cue in CATEGORY_CUES.get(category, []):
        cue_norm = cue.lower()
        if re.search(r"\b" + re.escape(cue_norm) + r"\b", text) or cue_norm in text:
            cues.append(cue)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for cue in cues:
        if cue not in seen:
            out.append(cue)
            seen.add(cue)
    return out[:max_cues]


def truncate_words(text: str, max_words: int) -> str:
    text = normalise_text(text)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def evidence_reason(description: str, category: str, max_words: int) -> str:
    cues = find_evidence_cues(description, category)
    if cues:
        reason = f"The description mentions {', '.join(cues)}, which aligns with {category}."
    else:
        reason = f"The incident description aligns most closely with {category} among the allowed risk categories."
    return truncate_words(reason, max_words)


def clean_reason(raw: str, labels: List[str], predicted_category: str, description: str, max_words: int) -> str:
    text = normalise_text(raw)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text).strip()
    # Keep the first sentence/line to avoid long explanations.
    text = text.splitlines()[0].strip() if text else ""
    if text:
        # Remove common prefixes.
        text = re.sub(r"^(reason|brief reason|explanation)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()

    # If the model merely repeats a category, gives a generic non-evidence reason,
    # or produces unstable text, use evidence fallback.
    label_norms = {normalise_label_text(x) for x in labels}
    generic_reason = bool(re.search(
        r"aligns most closely|among the allowed risk categories|incident description aligns",
        text,
        flags=re.IGNORECASE,
    ))
    if (
        not text
        or normalise_label_text(text) in label_norms
        or len(text) < 8
        or "((((" in text
        or generic_reason
    ):
        return evidence_reason(description, predicted_category, max_words)

    # Avoid unsupported hedging that sounds too verbose.
    text = truncate_words(text, max_words)
    return text


@torch.no_grad()
def generate_reason(
    model,
    tokenizer,
    description: str,
    labels: List[str],
    top_candidates: List[Dict],
    args: argparse.Namespace,
) -> str:
    if args.reason_mode == "none":
        return ""

    top1 = top_candidates[0]["category"]
    if args.reason_mode == "evidence":
        return evidence_reason(description, top1, args.max_reason_words)

    device = get_model_device(model)
    tokenizer.padding_side = "left"
    prompt = build_reason_prompt(tokenizer, description, top1, top_candidates, args.max_reason_words)
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=args.reason_max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = encoded["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    reason = clean_reason(raw, labels, top1, description, args.max_reason_words)
    tokenizer.padding_side = "right"
    return reason



def find_confined_space_guardrail_cues(description: str) -> List[str]:
    """Return strong confined-space cues found in the description."""
    text = normalise_text(description).lower()
    cues: List[str] = []

    # Strong phrase/regex patterns.
    for pattern in CONFINED_SPACE_STRONG_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            # Convert a readable cue from the pattern.
            cue = pattern
            cue = cue.replace(r"\b", "").replace(r"\s+", " ")
            cue = re.sub(r"\(\?:.*?\)", "", cue)
            cue = cue.replace("?", "").replace("\\", "")
            cue = re.sub(r"[\^$+*{}\[\]|()]", "", cue).strip()
            if cue:
                cues.append(cue)

    # Contextual tank evidence. "tank" alone is too broad because it may also be
    # related to loss of containment, asset integrity, or fire/explosion.
    if "tank" in text and re.search(r"\b(entry|enter|entered|entering|inside|inspection|oxygen|gas\s+test|atmosphere|permit)\b", text):
        cues.append("tank with entry/inspection/atmosphere evidence")

    # Deduplicate and keep the output short.
    out: List[str] = []
    seen = set()
    for cue in cues:
        cue = normalise_text(cue)
        if cue and cue.lower() not in seen:
            out.append(cue)
            seen.add(cue.lower())
    return out[:5]


def get_category_rank(ranked: List[Dict], category: str) -> Optional[int]:
    for item in ranked:
        if item.get("category") == category:
            return int(item.get("rank", 0))
    return None


def evaluate_safety_guardrails(description: str, ranked: List[Dict], args: argparse.Namespace) -> List[Dict]:
    """Detect safety-critical evidence/ranking inconsistencies.

    Current rule:
      - If strong Confined Space evidence is detected but Confined Space is not
        within the top N ranked categories, mark the case for manual review.
    """
    if args.disable_safety_guardrails:
        return []

    guardrails: List[Dict] = []
    top_n = max(1, int(args.guardrail_top_n))

    confined_cues = find_confined_space_guardrail_cues(description)
    if confined_cues:
        rank = get_category_rank(ranked, SAFETY_GUARDRAIL_CATEGORY)
        if rank is None or rank > top_n:
            guardrails.append({
                "guardrail_category": SAFETY_GUARDRAIL_CATEGORY,
                "guardrail_reason": "confined_space_evidence_detected_but_not_top3",
                "guardrail_priority": "high",
                "detected_category_rank": rank,
                "evidence_cues": confined_cues,
                "message": (
                    "Strong confined-space evidence was detected, but Confined Space was not ranked in the top "
                    f"{top_n} candidates. Manual review is required."
                ),
            })
    return guardrails


def guardrail_reason_sentence(guardrails: List[Dict], max_words: int) -> str:
    if not guardrails:
        return ""
    g = guardrails[0]
    cues = g.get("evidence_cues", [])
    cue_text = ", ".join(cues[:3]) if cues else "safety-critical evidence"
    category = g.get("guardrail_category", "a safety-critical category")
    reason = f"The description mentions {cue_text}, suggesting possible {category}; manual review is required."
    return truncate_words(reason, max_words)

def determine_review_status(top_candidates: List[Dict], description: str, args: argparse.Namespace) -> Tuple[str, str, str]:
    if len(normalise_text(description)) < args.min_description_chars:
        return "need_further_review", "high", "insufficient_description"

    if not top_candidates:
        return "need_further_review", "high", "no_candidate_scores"

    top1_score = float(top_candidates[0]["relative_score"])
    top2_score = float(top_candidates[1]["relative_score"]) if len(top_candidates) > 1 else 0.0
    margin = top1_score - top2_score

    if top1_score < args.medium_score_threshold:
        return "auto_label_candidate", "medium", "low_relative_top1_score"
    if margin < args.medium_margin_threshold:
        return "auto_label_candidate", "medium", "small_top1_top2_margin"
    return "auto_label_candidate", "low", "clear_top_candidate"


def predict_one(description: str, args: argparse.Namespace, labels: List[str], model, tokenizer) -> Dict:
    description = normalise_text(description)
    if len(description) < args.min_description_chars:
        return {
            "description": description,
            "top1_category": "",
            "top_candidates": [],
            "brief_reason": "Description is too short to support a reliable Risk Category decision.",
            "review_status": "need_further_review",
            "review_priority": "high",
            "review_reason": "insufficient_description",
            "risk_label_source": DEFAULT_LABEL_SOURCE,
            "score_note": "No scores were computed because the description was insufficient.",
        }

    if args.print_prompt:
        print("\nClassification prompt:\n")
        print(build_classification_prompt(tokenizer, description, labels))

    ranked = score_labels(
        model=model,
        tokenizer=tokenizer,
        description=description,
        labels=labels,
        max_length=args.max_length,
        score_batch_size=args.score_batch_size,
    )

    top_k = max(1, min(args.top_k, len(ranked)))
    top_candidates = []
    for item in ranked[:top_k]:
        top_candidates.append({
            "rank": int(item["rank"]),
            "category": item["category"],
            "relative_score": round(float(item["relative_score"]), 6),
            "mean_nll": round(float(item["mean_nll"]), 6),
            "answer_token_count": int(item["answer_token_count"]),
        })

    review_status, review_priority, review_reason = determine_review_status(top_candidates, description, args)

    safety_guardrails = evaluate_safety_guardrails(description, ranked, args)
    if safety_guardrails:
        review_status = "need_further_review"
        review_priority = safety_guardrails[0].get("guardrail_priority", "high")
        review_reason = safety_guardrails[0].get("guardrail_reason", "safety_guardrail_triggered")
        reason = guardrail_reason_sentence(safety_guardrails, args.max_reason_words)
    else:
        reason = generate_reason(model, tokenizer, description, labels, top_candidates, args)

    output = {
        "description": description,
        "top1_category": top_candidates[0]["category"],
        "top_candidates": top_candidates,
        "brief_reason": reason,
        "review_status": review_status,
        "review_priority": review_priority,
        "review_reason": review_reason,
        "safety_guardrail_triggered": bool(safety_guardrails),
        "safety_guardrails": safety_guardrails,
        "risk_label_source": DEFAULT_LABEL_SOURCE,
        "score_note": "relative_score is a ranking score over the 23 allowed labels, not a calibrated probability.",
    }
    return output


def print_result(result: Dict):
    print("\nPrediction result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("top_candidates"):
        print("\nReadable summary:")
        print(f"Top-1 Risk Category: {result['top1_category']}")
        print("Top candidates:")
        for item in result["top_candidates"]:
            print(f"  {item['rank']}. {item['category']}  score={item['relative_score']:.4f}")
        if result.get("brief_reason"):
            print(f"Brief reason: {result['brief_reason']}")
        print(f"Review: {result['review_status']} / {result['review_priority']} / {result['review_reason']}")
        if result.get("safety_guardrail_triggered"):
            for g in result.get("safety_guardrails", []):
                cues = ", ".join(g.get("evidence_cues", []))
                print(f"Safety guardrail: {g.get('guardrail_category')} | rank={g.get('detected_category_rank')} | cues={cues}")


def load_description_from_args(args: argparse.Namespace) -> str:
    if args.description_file:
        path = Path(args.description_file)
        if not path.exists():
            raise FileNotFoundError(f"Description file not found: {path}")
        return path.read_text(encoding="utf-8")
    return args.description


def run_single(args: argparse.Namespace, labels: List[str], model, tokenizer):
    description = load_description_from_args(args)
    if not normalise_text(description):
        raise ValueError("Please provide --description or --description-file, or use --input-csv for batch mode.")

    result = predict_one(description, args, labels, model, tokenizer)
    print_result(result)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON result to: {out_path}")


def run_batch_csv(args: argparse.Namespace, labels: List[str], model, tokenizer):
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found in CSV: {args.text_col}")

    rows = []
    for i, text in enumerate(df[args.text_col].tolist(), start=1):
        print(f"\nPredicting row {i}/{len(df)}", flush=True)
        result = predict_one(str(text), args, labels, model, tokenizer)
        flat = {
            "row_number": i,
            args.text_col: normalise_text(text),
            "top1_category": result.get("top1_category", ""),
            "brief_reason": result.get("brief_reason", ""),
            "review_status": result.get("review_status", ""),
            "review_priority": result.get("review_priority", ""),
            "review_reason": result.get("review_reason", ""),
            "safety_guardrail_triggered": result.get("safety_guardrail_triggered", False),
            "safety_guardrail_category": "|".join([g.get("guardrail_category", "") for g in result.get("safety_guardrails", [])]),
            "safety_guardrail_detected_rank": "|".join([str(g.get("detected_category_rank", "")) for g in result.get("safety_guardrails", [])]),
            "safety_guardrail_evidence": "|".join([", ".join(g.get("evidence_cues", [])) for g in result.get("safety_guardrails", [])]),
        }
        for item in result.get("top_candidates", []):
            r = item["rank"]
            flat[f"top{r}_category"] = item["category"]
            flat[f"top{r}_relative_score"] = item["relative_score"]
            flat[f"top{r}_mean_nll"] = item["mean_nll"]
        rows.append(flat)

    out_csv = Path(args.output_csv) if args.output_csv else input_path.with_name(input_path.stem + "_topk_reason.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved batch output to: {out_csv}")


def main():
    args = parse_args()
    set_seed(args.seed)

    label_list_json = Path(args.label_list_json)
    adapter_dir = Path(args.adapter_dir)
    if not label_list_json.exists():
        raise FileNotFoundError(f"Label list JSON not found: {label_list_json}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    labels = load_labels(label_list_json)
    args.top_k = max(1, min(args.top_k, len(labels)))
    use_bf16 = bool(torch.cuda.is_available() and not args.no_bf16)

    print("Single-description Risk Category inference setup:")
    print(json.dumps({
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "label_list_json": str(label_list_json),
        "num_labels": len(labels),
        "top_k": args.top_k,
        "reason_mode": args.reason_mode,
        "safety_guardrails_enabled": not args.disable_safety_guardrails,
        "guardrail_top_n": args.guardrail_top_n,
        "bf16": use_bf16,
    }, ensure_ascii=False, indent=2))

    tokenizer = load_tokenizer(adapter_dir, args.base_model)
    model = load_model(args, adapter_dir, use_bf16)

    try:
        if args.input_csv:
            run_batch_csv(args, labels, model, tokenizer)
        else:
            run_single(args, labels, model, tokenizer)
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
