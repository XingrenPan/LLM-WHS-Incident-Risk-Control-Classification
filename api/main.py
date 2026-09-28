#!/usr/bin/env python3
"""
23_api_risk_control_combined_v3_multicontrol.py

Report 8 prototype API:
Risk Top-K + definition-guided Control candidate scoring +
rationale-grounded multi-control recommendation set.

Key difference from Report 7 v2:
- The Control output is not only the highest-likelihood Top-K.
- The API selects a small, functionally diverse set of Control Categories.
- Every returned Control recommendation includes a rationale, evidence snippet(s),
  control function, and primary/supplementary role.

Expected RunPod environment variables:
  API_TOKEN
  MODEL_DIR=models
  RISK_ADAPTER_DIR=$MODEL_DIR/llama31_8b_risk_description_only_main_classes_curated_23class
  CONTROL_DEFINITIONS_JSON=outputs/control_category_audit/control_definitions_extracted_v2/control_definitions_for_prompt.json
  RISK_OPTIONS_JSON=outputs/control_category_audit/control_definitions_extracted_v2/risk_to_control_options.json

Optional environment variables:
  BASE_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
  RISK_LABELS_FILE=/path/to/risk_labels.json
  RISK_CANDIDATE_BATCH_SIZE=64
  CONTROL_CANDIDATE_BATCH_SIZE=4
  RATIONALE_MODE=template      # template | llm
  SUSPEND_CONTROL_ON_HIGH_RISK=false
  GUARDRAIL_RULES_JSON=config/risk_pathway_control_guardrails_mining_v3_manual_handling.json
  RISK_MIN_REL_SCORE=0.25
  RISK_MIN_MARGIN=0.03
  CONTROL_MIN_REL_SCORE=0.10
  CONTROL_MIN_MARGIN=0.03

Start:
  uvicorn 23_api_risk_control_combined_v3_multicontrol:app --host 0.0.0.0 --port 9001
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
RISK_ADAPTER_DIR = os.getenv(
    "RISK_ADAPTER_DIR",
    f"{MODEL_DIR}/llama31_8b_risk_description_only_main_classes_curated_23class",
)
CONTROL_DEFINITIONS_JSON = os.getenv(
    "CONTROL_DEFINITIONS_JSON",
    "outputs/control_category_audit/control_definitions_extracted_v2/control_definitions_for_prompt.json",
)
RISK_OPTIONS_JSON = os.getenv(
    "RISK_OPTIONS_JSON",
    "outputs/control_category_audit/control_definitions_extracted_v2/risk_to_control_options.json",
)
RISK_LABELS_FILE = os.getenv("RISK_LABELS_FILE", "")

API_TOKEN = os.getenv("API_TOKEN", "")
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]

RISK_CANDIDATE_BATCH_SIZE = int(os.getenv("RISK_CANDIDATE_BATCH_SIZE", "64"))
CONTROL_CANDIDATE_BATCH_SIZE = int(os.getenv("CONTROL_CANDIDATE_BATCH_SIZE", "4"))

RISK_MIN_REL_SCORE = float(os.getenv("RISK_MIN_REL_SCORE", "0.25"))
RISK_MIN_MARGIN = float(os.getenv("RISK_MIN_MARGIN", "0.03"))
CONTROL_MIN_REL_SCORE = float(os.getenv("CONTROL_MIN_REL_SCORE", "0.10"))
CONTROL_MIN_MARGIN = float(os.getenv("CONTROL_MIN_MARGIN", "0.03"))

RATIONALE_MODE = os.getenv("RATIONALE_MODE", "template").strip().lower()  # template | llm
MAX_RATIONALE_NEW_TOKENS = int(os.getenv("MAX_RATIONALE_NEW_TOKENS", "220"))
SUSPEND_CONTROL_ON_HIGH_RISK = os.getenv("SUSPEND_CONTROL_ON_HIGH_RISK", "false").lower() in {"1", "true", "yes"}

DEFAULT_RISK_TOP_K = int(os.getenv("DEFAULT_RISK_TOP_K", "3"))
DEFAULT_CONTROL_TOP_K = int(os.getenv("DEFAULT_CONTROL_TOP_K", "5"))

# IMPORTANT: If your original v2 script already contains the exact 23 labels used
# during Risk LoRA training, paste them here or set RISK_LABELS_FILE to a JSON list.
# The script will also try to discover labels from common metadata files.
DEFAULT_RISK_LABELS = [
    "Acute Chemical Exposure",
    "Aviation",
    "Biological",
    "Confined Space",
    "Dropped / Falling Object",
    "Electrical (incl. Arc Flash/Blast)",
    "Energy release (excl. Electrical)",
    "Environmental",
    "Fall from height",
    "Fire / Explosion",
    "Fitness for Work",
    "Geotechnical / Ground Failure",
    "Heat Stress",
    "Lifting",
    "Loss of Containment",
    "Manual Handling / Ergonomics",
    "Mobile Equipment / Vehicles",
    "Non Process Fire & Explosion",
    "Pressure / Stored Energy",
    "Slip / Trip / Fall",
    "Structural Failure",
    "Vehicles & Mobile Equipment",
    "Working at Height",
]

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class ControlCandidate:
    category: str
    definition: str = ""
    risk_key: str = ""


class PredictRiskControlRequest(BaseModel):
    description: str = Field(..., min_length=2)
    location: str = "Unknown"
    risk_top_k: int = Field(DEFAULT_RISK_TOP_K, ge=1, le=10)
    control_top_k: int = Field(DEFAULT_CONTROL_TOP_K, ge=1, le=10)
    confirmed_risk: Optional[str] = None
    include_rationale: bool = True
    rationale_mode: Optional[str] = None  # template | llm


class PredictRiskRequest(BaseModel):
    description: str = Field(..., min_length=2)
    location: str = "Unknown"
    risk_top_k: int = Field(DEFAULT_RISK_TOP_K, ge=1, le=10)


class PredictControlFromRiskRequest(BaseModel):
    description: str = Field(..., min_length=2)
    risk_category: str = Field(..., min_length=1)
    location: str = "Unknown"
    control_top_k: int = Field(DEFAULT_CONTROL_TOP_K, ge=1, le=10)
    risk_review_priority: str = "low"
    include_rationale: bool = True
    rationale_mode: Optional[str] = None


# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Risk-Control Combined API v3 - Multi-Control Rationale",
    version="3.3-report8-statement-guardrails",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> None:
    if not API_TOKEN:
        # Fail closed. This avoids accidentally exposing the model without a token.
        raise HTTPException(status_code=500, detail="API_TOKEN is not set on the server.")
    if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalise_key(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def softmax_from_negative_nll(nlls: List[float]) -> List[float]:
    if not nlls:
        return []
    # score = exp(-nll). Use stable softmax over -nll.
    values = [-float(x) for x in nlls]
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def margin_from_topk(items: List[Dict[str, Any]]) -> float:
    if len(items) < 2:
        return 1.0
    return float(items[0].get("relative_score", 0.0)) - float(items[1].get("relative_score", 0.0))


def review_from_scores(topk: List[Dict[str, Any]], min_score: float, min_margin: float) -> Tuple[str, str, str]:
    if not topk:
        return "review", "high", "No candidate was available."
    top1 = float(topk[0].get("relative_score", 0.0))
    margin = margin_from_topk(topk)
    reasons = []
    if top1 < min_score:
        reasons.append(f"Top-1 relative score {top1:.4f} is below threshold {min_score:.4f}.")
    if margin < min_margin:
        reasons.append(f"Top-1/Top-2 margin {margin:.4f} is below threshold {min_margin:.4f}.")
    if reasons:
        priority = "medium" if top1 >= min_score * 0.5 else "high"
        return "review", priority, " ".join(reasons)
    return "auto", "low", "Top candidate passed provisional score and margin thresholds."


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", str(text or "").strip())
    return [p.strip() for p in parts if p.strip()]


EVIDENCE_KEYWORDS = {
    "hazard_source_control": ["wet", "spill", "spilt", "leak", "slippery", "oil", "water", "acid", "chemical", "dust", "debris", "dropped", "falling", "unsecured", "broken", "damaged", "defect", "fire", "gas"],
    "engineering_environmental_control": ["guard", "edge", "barrier", "ventilation", "drain", "surface", "equipment", "machine", "plant", "lighting", "blind spot", "design"],
    "isolation_access_control": ["access", "entered", "area", "zone", "underneath", "near", "pedestrian", "vehicle", "traffic", "barricade", "exclusion", "restricted", "isolate"],
    "inspection_maintenance": ["inspection", "inspect", "maintenance", "pre-start", "prestart", "check", "test", "defect", "failed", "failure", "service"],
    "administrative_procedure": ["procedure", "permit", "ptw", "training", "competency", "communication", "supervision", "instruction", "plan", "jsa", "swms"],
    "warning_signage": ["sign", "signage", "warning", "label", "visible", "notice", "alert", "wet", "slippery"],
    "ppe_personal_protection": ["ppe", "glove", "goggles", "respirator", "mask", "helmet", "hard hat", "harness", "lanyard", "boots", "footwear", "glasses"],
    "emergency_response": ["alarm", "evacuation", "rescue", "emergency", "first aid", "shower", "eyewash", "fire extinguisher", "response"],
    "fitness_competency": ["fatigue", "drug", "alcohol", "fitness", "competency", "licence", "license", "trained", "authorised", "authorized"],
    "general_control": [],
}



# -----------------------------------------------------------------------------
# Report 8 hazard-pathway layer
# -----------------------------------------------------------------------------

# The v3 prototype should not simply attach a rationale to the likelihood Top-K.
# This layer detects common incident mechanisms and uses them to choose a safer
# risk context for Control recommendation and to diversify the selected controls.

PATHWAY_FUNCTION_PRIORITY = {
    "wet_slip_fall": [
        "hazard_source_control",       # remove/dry/clean the low-friction surface
        "isolation_access_control",    # keep people away until the area is safe
        "warning_signage",             # warn people before exposure
        "engineering_environmental_control",
        "inspection_maintenance",
        "administrative_procedure",
    ],
    "dropped_object": [
        "hazard_source_control",
        "engineering_environmental_control",
        "inspection_maintenance",
        "isolation_access_control",
        "administrative_procedure",
        "ppe_personal_protection",
    ],
    "vehicle_pedestrian": [
        "isolation_access_control",
        "administrative_procedure",
        "engineering_environmental_control",
        "warning_signage",
        "inspection_maintenance",
    ],
    "electrical_isolation": [
        "isolation_access_control",
        "administrative_procedure",
        "inspection_maintenance",
        "ppe_personal_protection",
        "emergency_response",
    ],
    "confined_space": [
        "administrative_procedure",
        "inspection_maintenance",
        "emergency_response",
        "isolation_access_control",
        "ppe_personal_protection",
    ],
    "chemical_release_exposure": [
        "hazard_source_control",
        "isolation_access_control",
        "ppe_personal_protection",
        "emergency_response",
        "administrative_procedure",
        "inspection_maintenance",
    ],
    "lifting": [
        "engineering_environmental_control",
        "inspection_maintenance",
        "administrative_procedure",
        "isolation_access_control",
    ],
    "generic": [
        "hazard_source_control",
        "engineering_environmental_control",
        "isolation_access_control",
        "administrative_procedure",
        "warning_signage",
        "inspection_maintenance",
        "ppe_personal_protection",
        "emergency_response",
    ],
}


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


def detect_hazard_pathway(description: str) -> Dict[str, Any]:
    """Return a lightweight, auditable pathway hypothesis for Report 8 selection.

    This is intentionally rule-based. It is not a replacement for the Risk LoRA.
    It is a guardrail/selection layer used to avoid obviously poor downstream
    Control recommendations when the Risk Top-1 conflicts with strong incident
    evidence, such as a wet-floor slip being routed through Energy release.
    """
    d = str(description or "")
    low = d.lower()

    fall_terms = ["slip", "slipped", "slipping", "trip", "tripped", "fall", "fell", "fallen"]
    wet_terms = ["wet", "water", "bathroom", "shower", "washroom", "floor", "slippery", "low friction", "spill", "spilt", "spilled"]
    if _contains_any(low, fall_terms) and _contains_any(low, wet_terms):
        return {
            "pathway_id": "wet_slip_fall",
            "preferred_risk": "Slip / Trip / Fall",
            "confidence": "high",
            "summary": "Wet or slippery walking surface reduced traction, allowing worker exposure to a slip/fall pathway.",
            "hazard_factors": ["wet/slippery floor or surface", "reduced traction", "worker exposure to the area"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["wet_slip_fall"],
        }

    if _contains_any(low, ["dropped", "falling object", "fell from", "spanner", "tool dropped", "object dropped"]):
        return {
            "pathway_id": "dropped_object",
            "preferred_risk": "Dropped / Falling Object",
            "confidence": "high",
            "summary": "An object could fall or has fallen from height and expose people below.",
            "hazard_factors": ["object at height", "dropped/falling object", "people underneath or nearby"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["dropped_object"],
        }

    if _contains_any(low, ["vehicle", "truck", "loader", "forklift", "revers", "pedestrian", "near miss", "mobile equipment"]):
        return {
            "pathway_id": "vehicle_pedestrian",
            "preferred_risk": "Vehicles & Mobile Equipment",
            "confidence": "medium",
            "summary": "Mobile equipment or vehicle movement created a collision or interaction pathway.",
            "hazard_factors": ["mobile equipment movement", "pedestrian/vehicle interaction", "traffic exposure"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["vehicle_pedestrian"],
        }

    if _contains_any(low, ["electrical", "electric", "arc flash", "isolation", "isolat", "energised", "energized", "live cable"]):
        return {
            "pathway_id": "electrical_isolation",
            "preferred_risk": "Electrical (incl. Arc Flash/Blast)",
            "confidence": "medium",
            "summary": "Electrical energy or isolation failure may expose a worker to shock, arc flash, or release of energy.",
            "hazard_factors": ["electrical energy", "isolation/verification", "worker exposure"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["electrical_isolation"],
        }

    if _contains_any(low, ["confined", "tank", "vessel", "low oxygen", "oxygen alarm", "gas test", "gas testing", "atmosphere", "entry permit"]):
        return {
            "pathway_id": "confined_space",
            "preferred_risk": "Confined Space",
            "confidence": "high",
            "summary": "Confined-space entry or atmospheric hazard evidence is present.",
            "hazard_factors": ["confined-space entry", "atmospheric hazard", "permit/testing/rescue controls"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["confined_space"],
        }

    if _contains_any(low, ["chemical", "acid", "caustic", "solvent", "spill", "leak", "release", "hose", "containment", "exposure", "burn"]):
        return {
            "pathway_id": "chemical_release_exposure",
            "preferred_risk": "Loss of Containment",
            "confidence": "medium",
            "summary": "Chemical release, spill, or exposure evidence is present.",
            "hazard_factors": ["release/spill source", "worker exposure", "containment or response"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["chemical_release_exposure"],
        }

    if _contains_any(low, ["lift", "lifting", "crane", "sling", "rigging", "load", "hoist"]):
        return {
            "pathway_id": "lifting",
            "preferred_risk": "Lifting",
            "confidence": "medium",
            "summary": "A lifting operation, suspended load, or rigging pathway is present.",
            "hazard_factors": ["lifting equipment/load", "rigging or planning", "people near the load"],
            "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["lifting"],
        }

    return {
        "pathway_id": "generic",
        "preferred_risk": None,
        "confidence": "low",
        "summary": "No strong pathway-specific rule was triggered; use model Risk and likelihood-ranked controls with diversity selection.",
        "hazard_factors": [],
        "desired_control_functions": PATHWAY_FUNCTION_PRIORITY["generic"],
    }


def choose_risk_for_control(description: str, risk_prediction: Dict[str, Any], confirmed_risk: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Choose which Risk should provide the Control definition pool.

    The model Risk Top-1 remains visible. When a strong hazard-pathway rule clearly
    conflicts with Top-1 but maps to a known Risk label, Control generation uses the
    pathway-preferred Risk and marks the result provisional. This directly fixes the
    wet-bathroom case where Top-1 was Energy release but Top-2 was Slip / Trip / Fall.
    """
    pathway = detect_hazard_pathway(description)
    model_top1 = risk_prediction.get("predicted_risk_category")
    top_labels = [x.get("label") for x in risk_prediction.get("top_k", []) if x.get("label")]

    if confirmed_risk:
        return confirmed_risk, {
            "source": "confirmed_risk",
            "risk_used_for_control": confirmed_risk,
            "model_top1_risk": model_top1,
            "pathway": pathway,
            "override_applied": False,
            "provisional": False,
            "reason": "A confirmed Risk Category was supplied by the caller and was used for Control recommendation.",
        }

    preferred = pathway.get("preferred_risk")
    if preferred:
        preferred_norm = normalise_key(preferred)
        label_norms = {normalise_key(x): x for x in RISK_LABELS}
        top_norms = {normalise_key(x): x for x in top_labels}
        model_top1_norm = normalise_key(model_top1 or "")
        preferred_known = preferred_norm in label_norms or preferred_norm in top_norms or bool(get_controls_for_risk(preferred))

        if preferred_known and preferred_norm != model_top1_norm and pathway.get("confidence") in {"high", "medium"}:
            return preferred, {
                "source": "hazard_pathway_guardrail",
                "risk_used_for_control": preferred,
                "model_top1_risk": model_top1,
                "pathway": pathway,
                "override_applied": True,
                "provisional": True,
                "reason": (
                    f"The model Top-1 Risk was '{model_top1}', but the incident triggered the "
                    f"'{pathway['pathway_id']}' pathway. Control recommendation therefore uses "
                    f"'{preferred}' as a provisional Risk context."
                ),
            }

    return model_top1, {
        "source": "model_top1",
        "risk_used_for_control": model_top1,
        "model_top1_risk": model_top1,
        "pathway": pathway,
        "override_applied": False,
        "provisional": risk_prediction.get("review_priority") in {"medium", "high"},
        "reason": "The model Top-1 Risk Category was used for Control recommendation.",
    }


def pathway_keyword_boost(label: str, definition: str, control_function: str, pathway: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    """Return (boost, penalty, reasons) for pathway-aware Control selection."""
    pid = pathway.get("pathway_id", "generic")
    text = normalise_key(f"{label} {definition}")
    boost = 0.0
    penalty = 0.0
    reasons: List[str] = []

    desired = pathway.get("desired_control_functions") or PATHWAY_FUNCTION_PRIORITY["generic"]
    if control_function in desired:
        # Earlier functions are preferred because they interrupt more direct parts of the pathway.
        idx = desired.index(control_function)
        b = max(0.04, 0.18 - idx * 0.025)
        boost += b
        reasons.append(f"matches desired function {control_function}")

    if pid == "wet_slip_fall":
        if any(k in text for k in ["housekeeping", "clean", "cleaning", "dry", "spill", "slip", "slippery", "floor", "surface", "walkway", "traction", "drainage"]):
            boost += 0.42
            reasons.append("addresses wet/slippery surface or housekeeping source control")
        if any(k in text for k in ["sign", "signage", "warning", "demarcation", "label", "notice"]):
            boost += 0.34
            reasons.append("supports warning/signage for wet-floor exposure")
        if any(k in text for k in ["barricade", "exclusion", "access", "restrict", "separation", "isolation", "cordon"]):
            boost += 0.30
            reasons.append("supports access restriction or exposure prevention")
        if any(k in text for k in ["emergency", "crisis", "open hole", "breakthrough", "permit to work", "ptw"]):
            penalty += 0.22
            reasons.append("less specific to a wet-floor slip pathway")

    elif pid == "dropped_object":
        if any(k in text for k in ["dropped", "falling", "object", "tool", "tether", "toe board", "net", "inspection", "barricade", "exclusion"]):
            boost += 0.32
            reasons.append("matches dropped-object prevention or exclusion pathway")
    elif pid == "vehicle_pedestrian":
        if any(k in text for k in ["traffic", "vehicle", "pedestrian", "separation", "driver", "revers", "spotter", "segregation"]):
            boost += 0.34
            reasons.append("matches vehicle/pedestrian interaction pathway")
    elif pid == "electrical_isolation":
        if any(k in text for k in ["isolation", "electrical", "arc", "test", "permit", "loto", "lockout", "tagout"]):
            boost += 0.34
            reasons.append("matches electrical isolation or verification pathway")
    elif pid == "confined_space":
        if any(k in text for k in ["confined", "gas", "atmosphere", "oxygen", "entry", "permit", "rescue", "monitoring"]):
            boost += 0.34
            reasons.append("matches confined-space entry, gas testing, or rescue pathway")
    elif pid == "chemical_release_exposure":
        if any(k in text for k in ["containment", "chemical", "spill", "leak", "isolation", "ppe", "glove", "goggles", "eyewash", "shower", "emergency"]):
            boost += 0.30
            reasons.append("matches chemical containment/exposure pathway")
    elif pid == "lifting":
        if any(k in text for k in ["lifting", "crane", "rigging", "sling", "load", "plant", "equipment", "exclusion"]):
            boost += 0.30
            reasons.append("matches lifting operation controls")

    return boost, penalty, reasons

def extract_evidence(description: str, control_function: str, max_items: int = 2) -> List[str]:
    desc = str(description or "")
    sentences = sentence_split(desc)
    if not sentences:
        return [truncate_text(desc, 180)] if desc.strip() else []

    kws = EVIDENCE_KEYWORDS.get(control_function, [])
    matched = []
    for sent in sentences:
        low = sent.lower()
        if any(kw in low for kw in kws):
            matched.append(truncate_text(sent, 180))
        if len(matched) >= max_items:
            break

    if matched:
        return matched
    # Fallback: return the first incident sentence, but mark it as general evidence.
    return [truncate_text(sentences[0], 180)]

def overall_control_logic_statement(description: str, pathway: Optional[Dict[str, Any]] = None) -> str:
    """Return an application-facing event-level explanation for the recommended controls.

    This is deliberately written as a complete scenario statement rather than a
    short evidence token list. The UI can show this field to explain why multiple
    controls may be recommended for one incident.
    """
    pathway = pathway or detect_hazard_pathway(description)
    pid = pathway.get("pathway_id", "generic")

    if pid == "wet_slip_fall":
        return (
            "In this wet-floor fall scenario, one control may remove the immediate hazard by "
            "cleaning or drying the floor, another may prevent exposure by restricting access to "
            "the affected area, and another may provide administrative warning through wet-floor signage."
        )
    if pid == "dropped_object":
        return (
            "In this dropped-object scenario, one control may prevent the object from falling by securing tools "
            "or materials at height, another may prevent worker exposure by establishing an exclusion zone below "
            "overhead work, and another may verify ongoing effectiveness through inspection of dropped-object controls."
        )
    if pid == "vehicle_pedestrian":
        return (
            "In this vehicle or mobile-equipment scenario, one control may separate pedestrians from moving equipment, "
            "another may organise vehicle movement through traffic management rules, and another may improve awareness "
            "through warning, visibility, or spotter arrangements."
        )
    if pid == "confined_space":
        return (
            "In this confined-space scenario, one control may prevent unsafe entry through permit and pre-entry testing, "
            "another may manage changing atmospheric conditions through monitoring and ventilation, and another may "
            "reduce consequence severity through prepared rescue and emergency arrangements."
        )
    if pid == "chemical_release_exposure":
        return (
            "In this chemical release or exposure scenario, one control may stop or contain the release at source, "
            "another may prevent worker exposure by isolating the affected area, and another may reduce injury severity "
            "through suitable PPE, decontamination, or emergency response measures."
        )
    if pid == "lifting":
        return (
            "In this lifting scenario, one control may ensure the lift is planned and within load limits, another may "
            "verify lifting equipment and rigging integrity before use, and another may prevent exposure by keeping "
            "people clear of the suspended-load area."
        )
    return (
        "For this incident, multiple controls may be appropriate because different controls can address the hazard source, "
        "the exposure pathway, worker awareness or procedures, and consequence reduction. The recommended controls should "
        "therefore be interpreted as complementary decision-support measures rather than a single exclusive label."
    )


def build_supporting_event_statement(description: str, pathway_id: str, control_function: str, control_text: str) -> str:
    """Return a complete event-specific statement for one recommended control.

    This field is designed for the application layer. It explains how the control
    connects to the event mechanism in natural language, while `evidence_from_description`
    remains available as a compact backend/audit field.
    """
    pid = pathway_id or detect_hazard_pathway(description).get("pathway_id", "generic")

    if pid == "wet_slip_fall":
        if control_function == "hazard_source_control":
            return (
                "In this wet-floor fall scenario, the immediate hazard is the wet bathroom floor near the shower area. "
                f"{control_text} directly addresses the low-friction surface condition that contributed to the slip/fall pathway."
            )
        if control_function == "isolation_access_control":
            return (
                "In this wet-floor fall scenario, workers may be exposed to a slippery walking surface before it is made safe. "
                f"{control_text} prevents exposure by keeping people away from the affected area until the floor is cleaned, dried, or otherwise controlled."
            )
        if control_function == "warning_signage":
            return (
                "In this wet-floor fall scenario, workers may not recognise the temporary slip hazard before entering the area. "
                f"{control_text} provides administrative warning before exposure, but should support rather than replace cleaning, drying, or access restriction."
            )
        if control_function == "engineering_environmental_control":
            return (
                "In this wet-floor fall scenario, the shower-area environment may repeatedly create wet or low-traction walking surfaces. "
                f"{control_text} addresses the environmental condition that may allow the same slip/fall pathway to recur."
            )

    if pid == "dropped_object":
        if control_function == "hazard_source_control":
            return f"In this dropped-object scenario, {control_text} reduces the chance that tools or materials can fall from height."
        if control_function == "isolation_access_control":
            return f"In this dropped-object scenario, {control_text} prevents people from being exposed below or near the potential drop zone."
        if control_function == "inspection_maintenance":
            return f"In this dropped-object scenario, {control_text} checks that prevention controls remain present, suitable, and effective."

    if pid == "vehicle_pedestrian":
        if control_function == "isolation_access_control":
            return f"In this vehicle or mobile-equipment scenario, {control_text} reduces collision exposure by separating people from moving equipment."
        if control_function == "administrative_procedure":
            return f"In this vehicle or mobile-equipment scenario, {control_text} manages movement rules, communication, and supervision for shared work areas."
        if control_function == "warning_signage":
            return f"In this vehicle or mobile-equipment scenario, {control_text} improves awareness before workers enter reversing, blind-spot, or traffic interaction areas."

    if pid == "confined_space":
        if control_function == "administrative_procedure":
            return f"In this confined-space scenario, {control_text} helps prevent unsafe entry by confirming permit, testing, and entry requirements before exposure occurs."
        if control_function == "inspection_maintenance":
            return f"In this confined-space scenario, {control_text} manages atmospheric conditions that may change during the work."
        if control_function == "emergency_response":
            return f"In this confined-space scenario, {control_text} reduces consequence severity if rescue or emergency intervention becomes necessary."

    if pid == "chemical_release_exposure":
        if control_function == "hazard_source_control":
            return f"In this chemical release or exposure scenario, {control_text} reduces the release at its source before more people or surfaces are affected."
        if control_function == "isolation_access_control":
            return f"In this chemical release or exposure scenario, {control_text} prevents unnecessary access to the affected area and reduces exposure."
        if control_function == "ppe_personal_protection":
            return f"In this chemical release or exposure scenario, {control_text} helps reduce injury severity, but should supplement containment and isolation controls."

    if pid == "lifting":
        if control_function == "administrative_procedure":
            return f"In this lifting scenario, {control_text} ensures that the lift is planned and controlled before the load is moved."
        if control_function == "inspection_maintenance":
            return f"In this lifting scenario, {control_text} verifies equipment condition before failure can occur under load."
        if control_function == "isolation_access_control":
            return f"In this lifting scenario, {control_text} prevents people from being exposed to a suspended or moving load."

    return (
        f"In this incident scenario, {control_text} is recommended because it addresses a relevant part of the hazard pathway, "
        "such as the hazard source, exposure pathway, worker awareness, procedural control, or consequence reduction."
    )


CONTROL_FUNCTION_KEYWORDS = [
    # Source/hazard removal is intentionally early so housekeeping/cleaning is not swallowed by generic admin controls.
    ("hazard_source_control", ["housekeeping", "clean", "cleaning", "dry", "drying", "spill", "slip", "slippery", "floor", "walkway", "surface condition", "traction", "drainage", "secondary containment", "containment", "earth bund", "bund", "dropped objects", "dropped objs"]),
    ("warning_signage", ["sign", "signage", "warning", "label", "tag", "notice", "demarcation"]),
    ("isolation_access_control", ["isolation", "isolate", "access", "barricade", "exclusion", "segregation", "separation", "lockout", "loto", "pedestrian separation", "vehicle pedestrian", "traffic management", "cordon", "restricted"]),
    ("engineering_environmental_control", ["engineering", "guard", "guarding", "edge protection", "ventilation", "drainage", "surface", "design", "platform", "scaffold", "ladder", "plant", "equipment", "anti slip", "anti-slip", "nonslip", "non slip"]),
    ("inspection_maintenance", ["inspection", "inspect", "maintenance", "pre-start", "prestart", "testing", "test", "monitoring", "audit", "service", "integrity", "dropped objs inspctns"]),
    ("administrative_procedure", ["procedure", "permit", "ptw", "training", "competency", "communication", "supervision", "jsa", "swms", "driver behavior", "driver behaviour", "traffic mgmt"]),
    ("emergency_response", ["emergency", "crisis", "rescue", "first aid", "evacuation", "alarm", "shower", "eyewash", "fire fighting", "firefighting"]),
    ("ppe_personal_protection", ["ppe", "personal protective", "glove", "goggles", "respirator", "mask", "helmet", "hard hat", "harness", "lanyard", "footwear", "boots", "hearing protection"]),
    ("fitness_competency", ["fitness for work", "fatigue", "alcohol", "drug", "competent", "authorised", "authorized", "licence", "license"]),
]


def infer_control_function(category: str, definition: str = "") -> str:
    text = normalise_key(f"{category} {definition}")
    for fn, keywords in CONTROL_FUNCTION_KEYWORDS:
        for kw in keywords:
            if normalise_key(kw) in text:
                return fn
    return "general_control"

def control_role_from_function(control_function: str) -> str:
    if control_function in {"hazard_source_control", "engineering_environmental_control", "isolation_access_control"}:
        return "primary"
    if control_function in {"inspection_maintenance", "emergency_response"}:
        return "primary_or_supporting"
    return "supplementary"


def priority_from_score_and_role(relative_score: float, role: str, provisional: bool) -> str:
    if provisional:
        return "provisional"
    if role == "primary" and relative_score >= CONTROL_MIN_REL_SCORE:
        return "high"
    if relative_score >= CONTROL_MIN_REL_SCORE:
        return "medium"
    return "low"


def template_rationale(
    description: str,
    risk_category: str,
    candidate: ControlCandidate,
    control_function: str,
    evidence: List[str],
) -> str:
    evidence_text = "; ".join(evidence) if evidence else "the supplied incident description"
    definition_text = truncate_text(candidate.definition, 220) if candidate.definition else "the available risk-specific control definition"
    pathway = detect_hazard_pathway(description)
    pid = pathway.get("pathway_id", "generic")

    # Pathway-specific rationales are used where possible. This is the main fix
    # over the previous v3 output, which produced generic rationales that merely
    # repeated the selected category and did not explain the wet-floor mechanism.
    if pid == "wet_slip_fall":
        if control_function == "hazard_source_control":
            return (
                "The incident describes a worker slipping/falling on a wet bathroom floor, so the immediate hazard pathway is "
                "wet surface -> reduced traction -> slip/fall. This control is relevant because it removes or reduces the "
                "wet/slippery surface condition rather than only responding after the fall. "
                f"Supporting evidence: {evidence_text}. Matched definition/category context: {definition_text}."
            )
        if control_function == "isolation_access_control":
            return (
                "The incident describes exposure to a wet walking surface. This control is relevant because access restriction, "
                "barricading, or isolation can prevent people from entering the wet area until the surface has been made safe. "
                f"Supporting evidence: {evidence_text}. Matched definition/category context: {definition_text}."
            )
        if control_function == "warning_signage":
            return (
                "The incident describes a wet-floor slip/fall pathway. Warning signage is relevant as a supplementary control "
                "because it alerts people before they step onto the low-friction surface, although it should not replace cleaning, "
                "drying, or access control. "
                f"Supporting evidence: {evidence_text}. Matched definition/category context: {definition_text}."
            )
        if control_function == "engineering_environmental_control":
            return (
                "The incident indicates a walking-surface condition that allowed slipping. This control is relevant if it improves "
                "the physical environment, such as surface condition, drainage, anti-slip design, or other engineering measures that "
                "reduce recurrence. "
                f"Supporting evidence: {evidence_text}. Matched definition/category context: {definition_text}."
            )

    function_explanations = {
        "hazard_source_control": "addresses the hazard source or immediate unsafe condition before further exposure occurs",
        "engineering_environmental_control": "modifies the work environment, equipment, or physical condition that contributes to the event",
        "isolation_access_control": "reduces exposure by separating people from the hazardous area, plant, equipment, or activity",
        "inspection_maintenance": "supports earlier detection and correction of defective, missing, or degraded controls",
        "administrative_procedure": "supports safer work execution through procedures, permits, supervision, communication, or competency controls",
        "warning_signage": "improves hazard awareness before a person enters or continues the hazardous task or area",
        "ppe_personal_protection": "reduces injury severity or exposure at the worker level, but should not replace stronger controls",
        "emergency_response": "supports mitigation after an event or alarm condition has occurred",
        "fitness_competency": "addresses worker readiness, authorisation, competency, or fitness-for-work factors",
        "general_control": "matches the incident and risk-specific definition but does not yet have a more specific control-function mapping",
    }
    fn_text = function_explanations.get(control_function, function_explanations["general_control"])
    return (
        f"This control is recommended for the {risk_category} pathway because it {fn_text}. "
        f"The supporting incident evidence is: {evidence_text}. "
        f"The matched control definition/category context is: {definition_text}."
    )

def coverage_summary(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    fns = {r.get("control_function") for r in recommendations}
    return {
        "hazard_source_control_included": "hazard_source_control" in fns,
        "engineering_or_environmental_control_included": "engineering_environmental_control" in fns,
        "isolation_or_access_control_included": "isolation_access_control" in fns,
        "administrative_or_warning_control_included": bool({"administrative_procedure", "warning_signage"} & fns),
        "ppe_included": "ppe_personal_protection" in fns,
        "ppe_only": bool(fns) and fns == {"ppe_personal_protection"},
        "functions_covered": sorted([x for x in fns if x]),
    }


def confined_space_guardrail(description: str, top_risks: List[Dict[str, Any]]) -> Optional[str]:
    low = str(description or "").lower()
    evidence_terms = ["confined", "tank", "vessel", "low oxygen", "oxygen alarm", "gas test", "gas testing", "atmosphere", "entry permit"]
    has_evidence = any(t in low for t in evidence_terms)
    if not has_evidence:
        return None
    top_names = [normalise_key(x.get("label", "")) for x in top_risks[:3]]
    if not any("confined space" in x for x in top_names):
        return "Confined-space evidence was detected, but Confined Space was not in the Risk Top-3. Route to high-priority review."
    return None


# -----------------------------------------------------------------------------
# Loading risk labels and control definitions
# -----------------------------------------------------------------------------


def try_extract_labels_from_json(obj: Any) -> Optional[List[str]]:
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return obj
    if isinstance(obj, dict):
        for key in ["risk_labels", "labels", "class_names", "classes", "id2label"]:
            if key in obj:
                value = obj[key]
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    return value
                if isinstance(value, dict):
                    # id2label can be {"0": "...", "1": "..."}
                    try:
                        return [value[str(i)] for i in range(len(value))]
                    except Exception:
                        pass
    return None


def load_risk_labels() -> List[str]:
    candidates = []
    if RISK_LABELS_FILE:
        candidates.append(Path(RISK_LABELS_FILE))
    adapter_dir = Path(RISK_ADAPTER_DIR)
    candidates.extend(
        [
            adapter_dir / "risk_labels.json",
            adapter_dir / "labels.json",
            adapter_dir / "label_names.json",
            adapter_dir / "class_names.json",
            adapter_dir / "training_metadata.json",
            Path(MODEL_DIR) / "risk_labels.json",
        ]
    )
    for p in candidates:
        if p.exists():
            try:
                labels = try_extract_labels_from_json(load_json(p))
                if labels:
                    return [str(x) for x in labels]
            except Exception:
                continue
    return DEFAULT_RISK_LABELS


def option_to_category(option: Any) -> Optional[str]:
    if isinstance(option, str):
        return option
    if isinstance(option, dict):
        for key in ["control_category", "category", "name", "label", "control"]:
            if key in option and option[key]:
                return str(option[key])
    return None


def extract_definition_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ["definition", "description", "text", "meaning", "control_definition"]:
            if key in value and value[key]:
                return str(value[key])
    return ""


def load_control_candidates() -> Dict[str, List[ControlCandidate]]:
    risk_options_obj = load_json(RISK_OPTIONS_JSON)
    definitions_obj = load_json(CONTROL_DEFINITIONS_JSON)

    by_risk: Dict[str, Dict[str, ControlCandidate]] = {}

    def add_candidate(risk_name: str, category: str, definition: str = "") -> None:
        risk_key = normalise_key(risk_name)
        if not risk_key or not category:
            return
        cat = str(category).strip()
        cat_key = normalise_key(cat)
        by_risk.setdefault(risk_key, {})
        if cat_key not in by_risk[risk_key]:
            by_risk[risk_key][cat_key] = ControlCandidate(category=cat, definition=definition or "", risk_key=risk_key)
        elif definition and not by_risk[risk_key][cat_key].definition:
            by_risk[risk_key][cat_key].definition = definition

    # risk_to_control_options.json: usually {risk: [control categories]}
    if isinstance(risk_options_obj, dict):
        for risk_name, options in risk_options_obj.items():
            if isinstance(options, list):
                for opt in options:
                    cat = option_to_category(opt)
                    if cat:
                        add_candidate(risk_name, cat, extract_definition_from_value(opt))
            elif isinstance(options, dict):
                for cat, value in options.items():
                    add_candidate(risk_name, str(cat), extract_definition_from_value(value))

    # control_definitions_for_prompt.json: support several possible structures.
    if isinstance(definitions_obj, dict):
        for risk_name, value in definitions_obj.items():
            if isinstance(value, dict):
                # {risk: {control: definition}}
                for cat, def_value in value.items():
                    add_candidate(risk_name, str(cat), extract_definition_from_value(def_value))
            elif isinstance(value, list):
                # {risk: [{category:..., definition:...}]}
                for item in value:
                    cat = option_to_category(item)
                    if cat:
                        add_candidate(risk_name, cat, extract_definition_from_value(item))
            elif isinstance(value, str):
                # Could be a flattened key. Try risk||category or risk::category.
                key = str(risk_name)
                if "||" in key:
                    r, c = key.split("||", 1)
                    add_candidate(r, c, value)
                elif "::" in key:
                    r, c = key.split("::", 1)
                    add_candidate(r, c, value)
    elif isinstance(definitions_obj, list):
        for item in definitions_obj:
            if isinstance(item, dict):
                risk = item.get("risk") or item.get("risk_category") or item.get("risk_key")
                cat = option_to_category(item)
                if risk and cat:
                    add_candidate(str(risk), cat, extract_definition_from_value(item))

    return {risk_key: list(cat_map.values()) for risk_key, cat_map in by_risk.items()}


# Global state loaded at startup.
RISK_LABELS: List[str] = []
CONTROL_BY_RISK: Dict[str, List[ControlCandidate]] = {}
TOKENIZER = None
MODEL = None


# -----------------------------------------------------------------------------
# Model loading and scoring
# -----------------------------------------------------------------------------

@app.on_event("startup")
def startup_load_model() -> None:
    global RISK_LABELS, CONTROL_BY_RISK, TOKENIZER, MODEL

    RISK_LABELS = load_risk_labels()
    CONTROL_BY_RISK = load_control_candidates()

    print(f"[startup] Risk labels loaded: {len(RISK_LABELS)}")
    print(f"[startup] Control risk keys loaded: {len(CONTROL_BY_RISK)}")
    print(f"[startup] Base model: {BASE_MODEL_NAME}")
    print(f"[startup] Risk adapter: {RISK_ADAPTER_DIR}")

    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    if Path(RISK_ADAPTER_DIR).exists():
        MODEL = PeftModel.from_pretrained(base_model, RISK_ADAPTER_DIR)
        MODEL.eval()
        print("[startup] Risk LoRA adapter attached.")
    else:
        MODEL = base_model
        print("[startup] WARNING: Risk adapter directory not found. Running with base model only.")


@contextmanager
def maybe_disable_adapter(disable: bool = True):
    if not disable or MODEL is None:
        yield
        return
    if hasattr(MODEL, "disable_adapter"):
        try:
            with MODEL.disable_adapter():
                yield
            return
        except Exception:
            pass
    # Fallback: if adapter cannot be disabled, continue with the active model.
    yield


def build_chat_prompt(system_text: str, user_text: str) -> str:
    assert TOKENIZER is not None
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Safe fallback for non-chat tokenizers.
        return f"System: {system_text}\nUser: {user_text}\nAssistant:"


def risk_prompt(description: str, location: str) -> str:
    system_text = (
        "You are a workplace safety risk classification assistant. "
        "Select the single most appropriate Risk Category from the allowed candidates. "
        "Return only the category name."
    )
    user_text = (
        f"Location: {location or 'Unknown'}\n"
        f"Incident description: {description}\n\n"
        "Allowed Risk Categories:\n"
        + "\n".join([f"- {x}" for x in RISK_LABELS])
        + "\n\nRisk Category:"
    )
    return build_chat_prompt(system_text, user_text)


def control_scoring_prompt(description: str, risk_category: str, candidates: List[ControlCandidate]) -> str:
    defs = []
    for i, c in enumerate(candidates, start=1):
        if c.definition:
            defs.append(f"{i}. {c.category}: {truncate_text(c.definition, 420)}")
        else:
            defs.append(f"{i}. {c.category}")
    system_text = (
        "You are a workplace safety control recommendation assistant. "
        "Use only the allowed Control Categories. Select categories that are relevant to the incident and Risk Category."
    )
    user_text = (
        f"Incident description: {description}\n"
        f"Risk Category: {risk_category}\n\n"
        "Allowed Control Categories and definitions:\n"
        + "\n".join(defs)
        + "\n\nMost relevant Control Category:"
    )
    return build_chat_prompt(system_text, user_text)


def score_candidates_by_nll(prompt: str, candidate_texts: List[str], batch_size: int, disable_adapter: bool) -> List[float]:
    assert TOKENIZER is not None and MODEL is not None
    if not candidate_texts:
        return []

    prompt_ids = TOKENIZER(prompt, add_special_tokens=False).input_ids
    results: List[float] = []

    with torch.inference_mode(), maybe_disable_adapter(disable_adapter):
        for start in range(0, len(candidate_texts), batch_size):
            batch = candidate_texts[start : start + batch_size]
            input_ids_list = []
            labels_list = []
            max_len = 0

            for cand in batch:
                cand_ids = TOKENIZER(" " + cand.strip(), add_special_tokens=False).input_ids
                if not cand_ids:
                    cand_ids = [TOKENIZER.eos_token_id]
                ids = prompt_ids + cand_ids
                labels = [-100] * len(prompt_ids) + cand_ids
                input_ids_list.append(ids)
                labels_list.append(labels)
                max_len = max(max_len, len(ids))

            pad_id = TOKENIZER.pad_token_id or TOKENIZER.eos_token_id
            input_tensor = []
            label_tensor = []
            attention_mask = []
            for ids, labels in zip(input_ids_list, labels_list):
                pad_len = max_len - len(ids)
                input_tensor.append(ids + [pad_id] * pad_len)
                label_tensor.append(labels + [-100] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)

            device = next(MODEL.parameters()).device
            input_tensor_t = torch.tensor(input_tensor, dtype=torch.long, device=device)
            labels_t = torch.tensor(label_tensor, dtype=torch.long, device=device)
            attention_t = torch.tensor(attention_mask, dtype=torch.long, device=device)

            out = MODEL(input_ids=input_tensor_t, attention_mask=attention_t)
            # Shift because token t predicts token t+1.
            logits = out.logits[:, :-1, :].float()
            shifted_labels = labels_t[:, 1:]
            mask = shifted_labels.ne(-100)

            log_probs = torch.log_softmax(logits, dim=-1)
            safe_labels = shifted_labels.masked_fill(~mask, 0)
            token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            token_nll = -token_log_probs.masked_fill(~mask, 0.0)
            denom = mask.sum(dim=1).clamp(min=1)
            avg_nll = token_nll.sum(dim=1) / denom
            results.extend([float(x) for x in avg_nll.detach().cpu().tolist()])

            del input_tensor_t, labels_t, attention_t, out, logits, log_probs, token_log_probs, token_nll, avg_nll
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def rank_text_candidates(prompt: str, candidates: List[str], batch_size: int, disable_adapter: bool) -> List[Dict[str, Any]]:
    nlls = score_candidates_by_nll(prompt, candidates, batch_size=batch_size, disable_adapter=disable_adapter)
    rel_scores = softmax_from_negative_nll(nlls)
    rows = []
    for label, nll, score in zip(candidates, nlls, rel_scores):
        rows.append({"label": label, "avg_nll": round(nll, 6), "relative_score": round(score, 6)})
    rows.sort(key=lambda x: (x["relative_score"], -x["avg_nll"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def score_risk(description: str, location: str, top_k: int) -> Dict[str, Any]:
    p = risk_prompt(description, location)
    ranked = rank_text_candidates(
        p,
        RISK_LABELS,
        batch_size=RISK_CANDIDATE_BATCH_SIZE,
        disable_adapter=False,
    )
    top = ranked[:top_k]
    status, priority, reason = review_from_scores(top, RISK_MIN_REL_SCORE, RISK_MIN_MARGIN)
    guardrail = confined_space_guardrail(description, top)
    if guardrail:
        status, priority, reason = "review", "high", guardrail
    return {
        "predicted_risk_category": top[0]["label"] if top else None,
        "top_k": top,
        "review_status": status,
        "review_priority": priority,
        "reason": reason,
        "guardrail": guardrail,
    }


def get_controls_for_risk(risk_category: str) -> List[ControlCandidate]:
    key = normalise_key(risk_category)
    if key in CONTROL_BY_RISK:
        return CONTROL_BY_RISK[key]
    # Fuzzy containment fallback.
    for k, values in CONTROL_BY_RISK.items():
        if key and (key in k or k in key):
            return values
    return []


def score_control_candidates(description: str, risk_category: str, controls: List[ControlCandidate]) -> List[Dict[str, Any]]:
    prompt = control_scoring_prompt(description, risk_category, controls)
    candidate_labels = [c.category for c in controls]
    ranked = rank_text_candidates(
        prompt,
        candidate_labels,
        batch_size=CONTROL_CANDIDATE_BATCH_SIZE,
        disable_adapter=True,  # Report 7 used base model for definition-guided Control scoring.
    )
    candidate_by_label = {normalise_key(c.category): c for c in controls}
    for row in ranked:
        c = candidate_by_label.get(normalise_key(row["label"]))
        if c:
            row["definition"] = c.definition
            row["control_function"] = infer_control_function(c.category, c.definition)
    return ranked


def select_complementary_controls(
    ranked_controls: List[Dict[str, Any]],
    description: str,
    risk_category: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select a rationale-useful multi-control set.

    Unlike the earlier implementation, this is not just one candidate per function
    from the likelihood Top-K. It uses a detected hazard pathway to promote
    controls that explain distinct prevention logics. For the wet-bathroom case,
    this explicitly promotes source control, access restriction, and signage.
    """
    pathway = detect_hazard_pathway(description)
    desired = pathway.get("desired_control_functions") or PATHWAY_FUNCTION_PRIORITY["generic"]
    enriched: List[Dict[str, Any]] = []

    for row in ranked_controls:
        row = dict(row)
        fn = row.get("control_function") or infer_control_function(row.get("label", ""), row.get("definition", ""))
        row["control_function"] = fn
        boost, penalty, reasons = pathway_keyword_boost(row.get("label", ""), row.get("definition", ""), fn, pathway)
        rel = float(row.get("relative_score", 0.0))
        # Likelihood remains relevant, but pathway usefulness can override a weak raw Top-K
        # when the raw scorer ranks generic/emergency controls above direct pathway controls.
        row["heuristic_boost"] = round(boost, 6)
        row["heuristic_penalty"] = round(penalty, 6)
        row["selection_score"] = round(rel + boost - penalty, 6)
        row["pathway_match_reasons"] = reasons
        enriched.append(row)

    enriched.sort(key=lambda x: (float(x.get("selection_score", 0.0)), float(x.get("relative_score", 0.0))), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    # Pass 1: select one best candidate for each desired control function.
    for fn in desired:
        candidates = [r for r in enriched if r.get("control_function") == fn and normalise_key(r.get("label", "")) not in selected_keys]
        if not candidates:
            continue
        best = candidates[0]
        # Avoid adding a clearly irrelevant candidate only to satisfy diversity.
        if float(best.get("selection_score", 0.0)) <= 0 and selected:
            continue
        selected.append(best)
        selected_keys.add(normalise_key(best.get("label", "")))
        if len(selected) >= top_k:
            break

    # Pass 2: fill remaining slots by selection score.
    for row in enriched:
        key = normalise_key(row.get("label", ""))
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)
        if len(selected) >= top_k:
            break

    # Keep output in selected order: this reflects complementary pathway priority,
    # not pure likelihood rank.
    for i, row in enumerate(selected, start=1):
        row["selected_rank"] = i

    return selected, pathway

def llm_generate_rationale(
    description: str,
    risk_category: str,
    candidate: ControlCandidate,
    control_function: str,
    evidence: List[str],
) -> str:
    assert TOKENIZER is not None and MODEL is not None
    system_text = (
        "You are a workplace safety control recommendation assistant. "
        "Write one concise, evidence-grounded rationale. Do not invent facts."
    )
    user_text = (
        f"Incident description: {description}\n"
        f"Risk Category: {risk_category}\n"
        f"Control Category: {candidate.category}\n"
        f"Control Definition: {candidate.definition or 'No definition provided.'}\n"
        f"Control Function: {control_function}\n"
        f"Evidence snippets: {json.dumps(evidence, ensure_ascii=False)}\n\n"
        "Write a concise rationale explaining why this Control Category is relevant. "
        "The rationale must connect the incident evidence to the control definition. "
        "Return only the rationale sentence(s)."
    )
    prompt = build_chat_prompt(system_text, user_text)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(next(MODEL.parameters()).device)
    with torch.inference_mode(), maybe_disable_adapter(True):
        out = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_RATIONALE_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1] :]
    text = TOKENIZER.decode(generated, skip_special_tokens=True).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text or template_rationale(description, risk_category, candidate, control_function, evidence)




# -----------------------------------------------------------------------------
# Application-safe generated-control layer
# -----------------------------------------------------------------------------

# Internal origin flags. These are for backend/audit use. The application UI can
# hide them and simply display every item as a "Recommended Control".
CONTROL_ORIGIN_TAXONOMY = 0   # existing source Control Category / definition
CONTROL_ORIGIN_GENERATED = 1  # generated practical control measure, not an official category
CONTROL_ORIGIN_HYBRID = 2     # official category plus generated practical wording


def generated_controls_for_pathway(description: str, risk_category: str, pathway: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Return practical control measures when the source taxonomy has no usable category.

    These controls are deliberately returned as application-facing Recommended
    Controls, not as official Control Category labels. The origin flag remains in
    the payload for backend/audit traceability.
    """
    pid = pathway.get("pathway_id", "generic")
    evidence_general = [truncate_text(description, 180)] if description else []

    def item(
        rank: int,
        text: str,
        fn: str,
        role: str,
        priority: str,
        rationale: str,
        evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        supporting_statement = build_supporting_event_statement(description, pid, fn, text)
        return {
            "rank": rank,
            "control_text": text,
            "frontend_label": "Recommended Control",
            "official_control_category": None,
            "control_category": None,  # backward-compatible field, but not an official label for generated items
            "control_function": fn,
            "control_role": role,
            "priority": priority,
            "provisional": True,
            "origin_flag": CONTROL_ORIGIN_GENERATED,
            "origin_label": "generated_measure",
            "taxonomy_backed": False,
            "evidence_from_description": evidence or evidence_general,
            "supporting_event_statement": supporting_statement,
            "rationale": rationale,
        }

    controls: List[Dict[str, Any]] = []

    if pid == "wet_slip_fall":
        controls = [
            item(
                1,
                "Clean or dry the wet floor surface",
                "hazard_source_control",
                "primary",
                "high",
                "The incident indicates a wet bathroom floor and a slip/fall outcome. Cleaning or drying the surface directly addresses the low-friction condition that created the fall pathway.",
                extract_evidence(description, "hazard_source_control"),
            ),
            item(
                2,
                "Restrict access to the wet area until the surface is safe",
                "isolation_access_control",
                "primary_or_supporting",
                "high",
                "If the floor cannot be made safe immediately, restricting access prevents workers from entering the wet area and interrupts the exposure pathway.",
                extract_evidence(description, "isolation_access_control"),
            ),
            item(
                3,
                "Place wet-floor warning signage before the affected area",
                "warning_signage",
                "supplementary",
                "medium",
                "Wet-floor signage helps workers recognise the temporary slip hazard before stepping onto the low-friction surface. It should support, not replace, cleaning or access restriction.",
                extract_evidence(description, "warning_signage"),
            ),
            item(
                4,
                "Review drainage, matting, or anti-slip surface controls near the shower area",
                "engineering_environmental_control",
                "primary_or_supporting",
                "medium",
                "Because the incident occurred near a shower area, environmental controls such as drainage, matting, or anti-slip surfaces may reduce recurrence of the same wet-floor pathway.",
                extract_evidence(description, "engineering_environmental_control"),
            ),
        ]
    elif pid == "dropped_object":
        controls = [
            item(1, "Secure tools and materials at height", "hazard_source_control", "primary", "high", "The event involves a dropped or falling object. Securing objects at height directly reduces the source of the falling-object hazard.", extract_evidence(description, "hazard_source_control")),
            item(2, "Establish an exclusion zone below overhead work", "isolation_access_control", "primary", "high", "Separating people from the drop zone prevents exposure if an object falls.", extract_evidence(description, "isolation_access_control")),
            item(3, "Inspect tool lanyards, toe boards, and dropped-object prevention controls", "inspection_maintenance", "primary_or_supporting", "medium", "Inspection helps verify that dropped-object prevention controls remain in place and effective.", extract_evidence(description, "inspection_maintenance")),
        ]
    elif pid == "vehicle_pedestrian":
        controls = [
            item(1, "Separate pedestrians from mobile equipment routes", "isolation_access_control", "primary", "high", "The incident indicates a vehicle or mobile-equipment interaction pathway. Physical or procedural separation reduces exposure to moving equipment.", extract_evidence(description, "isolation_access_control")),
            item(2, "Use traffic management controls such as spotters, speed limits, and designated routes", "administrative_procedure", "primary_or_supporting", "medium", "Traffic management reduces the likelihood of conflict between vehicles and workers in shared areas.", extract_evidence(description, "administrative_procedure")),
            item(3, "Improve visibility and warning arrangements around reversing or blind-spot areas", "warning_signage", "supplementary", "medium", "Warnings and visibility controls help people recognise and avoid mobile-equipment movement hazards.", extract_evidence(description, "warning_signage")),
        ]
    elif pid == "confined_space":
        controls = [
            item(1, "Confirm confined-space entry permit and atmospheric testing before entry", "administrative_procedure", "primary", "high", "The incident contains confined-space or atmospheric hazard evidence. Permit and gas-testing controls are central to safe entry decisions.", extract_evidence(description, "administrative_procedure")),
            item(2, "Use continuous gas monitoring and ventilation where required", "inspection_maintenance", "primary_or_supporting", "high", "Atmospheric hazards can change during work, so monitoring and ventilation help detect and reduce exposure.", extract_evidence(description, "inspection_maintenance")),
            item(3, "Prepare rescue and emergency response arrangements before entry", "emergency_response", "primary_or_supporting", "medium", "Confined-space events may require rapid rescue, so emergency arrangements should be ready before entry.", extract_evidence(description, "emergency_response")),
        ]
    elif pid == "chemical_release_exposure":
        controls = [
            item(1, "Stop or contain the leak, spill, or release at source", "hazard_source_control", "primary", "high", "The incident indicates a chemical release or exposure pathway. Source control reduces the release before further exposure occurs.", extract_evidence(description, "hazard_source_control")),
            item(2, "Isolate the affected area and prevent unnecessary access", "isolation_access_control", "primary_or_supporting", "high", "Access restriction reduces the number of people exposed to the chemical hazard.", extract_evidence(description, "isolation_access_control")),
            item(3, "Use suitable chemical PPE and emergency decontamination arrangements", "ppe_personal_protection", "supplementary", "medium", "PPE and decontamination help reduce injury severity, but should support containment and isolation rather than replace them.", extract_evidence(description, "ppe_personal_protection")),
        ]
    elif pid == "lifting":
        controls = [
            item(1, "Verify lifting plan, load limits, and rigging method before the lift", "administrative_procedure", "primary", "high", "A lifting pathway requires planning and verification so the load is controlled throughout the task.", extract_evidence(description, "administrative_procedure")),
            item(2, "Inspect lifting equipment, slings, and attachments before use", "inspection_maintenance", "primary_or_supporting", "high", "Inspection helps identify damaged or unsuitable lifting equipment before it fails under load.", extract_evidence(description, "inspection_maintenance")),
            item(3, "Keep people clear of the suspended load or lifting exclusion zone", "isolation_access_control", "primary", "high", "Exclusion prevents exposure if the load shifts, drops, or fails.", extract_evidence(description, "isolation_access_control")),
        ]
    else:
        controls = [
            item(1, "Identify and control the immediate hazard source", "hazard_source_control", "primary", "medium", "The incident description indicates an unsafe condition. The first control should address the immediate source of harm where practicable.", evidence_general),
            item(2, "Prevent further worker exposure to the hazardous area or activity", "isolation_access_control", "primary_or_supporting", "medium", "Where the source cannot be immediately removed, exposure should be controlled by separating people from the hazard.", evidence_general),
            item(3, "Apply task instructions, supervision, or warning controls for the remaining risk", "administrative_procedure", "supplementary", "low", "Administrative controls can support awareness and consistent work execution, but should not replace stronger controls.", evidence_general),
        ]

    return controls[: max(1, limit)]


def taxonomy_recommendation_to_app_item(row: Dict[str, Any], candidate: ControlCandidate, description: str, risk_category: str, provisional: bool, include_rationale: bool, mode: str) -> Dict[str, Any]:
    fn = row.get("control_function") or infer_control_function(candidate.category, candidate.definition)
    evidence = extract_evidence(description, fn)
    role = control_role_from_function(fn)
    rationale = ""
    if include_rationale:
        if mode == "llm":
            try:
                rationale = llm_generate_rationale(description, risk_category, candidate, fn, evidence)
            except Exception as exc:
                rationale = template_rationale(description, risk_category, candidate, fn, evidence)
                rationale += f" [LLM rationale generation failed and template rationale was used: {type(exc).__name__}.]"
        else:
            rationale = template_rationale(description, risk_category, candidate, fn, evidence)

    rel_score = float(row.get("relative_score", 0.0))
    return {
        "rank": row.get("selected_rank"),
        "likelihood_rank": row.get("rank"),
        "control_text": candidate.category,
        "frontend_label": "Recommended Control",
        "official_control_category": candidate.category,
        "control_category": candidate.category,
        "control_function": fn,
        "control_role": role,
        "priority": priority_from_score_and_role(rel_score, role, provisional),
        "provisional": provisional,
        "origin_flag": CONTROL_ORIGIN_TAXONOMY,
        "origin_label": "taxonomy_backed",
        "taxonomy_backed": True,
        "relative_score": row.get("relative_score"),
        "selection_score": row.get("selection_score"),
        "heuristic_boost": row.get("heuristic_boost"),
        "heuristic_penalty": row.get("heuristic_penalty"),
        "pathway_match_reasons": row.get("pathway_match_reasons", []),
        "avg_nll": row.get("avg_nll"),
        "definition": candidate.definition,
        "evidence_from_description": evidence,
        "supporting_event_statement": build_supporting_event_statement(
            description,
            detect_hazard_pathway(description).get("pathway_id", "generic"),
            fn,
            candidate.category,
        ),
        "rationale": rationale,
    }


def make_control_source_summary(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "taxonomy_backed_count": sum(1 for r in recs if r.get("origin_flag") == CONTROL_ORIGIN_TAXONOMY),
        "generated_count": sum(1 for r in recs if r.get("origin_flag") == CONTROL_ORIGIN_GENERATED),
        "hybrid_count": sum(1 for r in recs if r.get("origin_flag") == CONTROL_ORIGIN_HYBRID),
        "origin_flags_present": sorted({r.get("origin_flag") for r in recs if r.get("origin_flag") is not None}),
        "ui_should_display_origin": False,
    }


def generated_fallback_response(description: str, risk_category: str, control_top_k: int, risk_review_priority: str, reason_internal: str) -> Dict[str, Any]:
    pathway = detect_hazard_pathway(description)
    recs = generated_controls_for_pathway(description, risk_category, pathway, limit=control_top_k)
    # Ensure ranks are consecutive after truncation.
    for i, r in enumerate(recs, start=1):
        r["rank"] = i
    priority = "medium" if risk_review_priority in {"low", "medium"} else "high"
    return {
        "mode": "application_safe_recommended_controls",
        "display_mode": "recommended_controls",
        "risk_category_used": risk_category,
        "hazard_pathway": pathway,
        "overall_control_logic_statement": overall_control_logic_statement(description, pathway),
        "recommended_controls": recs,
        # Backward-compatible alias for earlier testing code.
        "recommended_control_set": recs,
        "taxonomy_backed_controls": [],
        "generated_control_measures": recs,
        "raw_likelihood_top_k": [],
        "coverage_summary": coverage_summary(recs),
        "internal_control_source_summary": make_control_source_summary(recs),
        "control_source_status": "generated_controls_used",
        "review_required": True,
        "review_status": "review",
        "review_priority": priority,
        # Application-facing reason should not expose taxonomy coverage failure.
        "reason": "Recommended controls were produced from the incident hazard pathway and should be reviewed before operational use.",
        "internal_reason": reason_internal,
        "control_skipped": False,
        "selection_note": (
            "The application-facing output is a unified Recommended Controls list. Internal origin flags are retained for audit, "
            "but the UI does not need to expose whether an item came from the source taxonomy or the generated-control layer."
        ),
    }

def build_control_recommendation_set(
    description: str,
    risk_category: str,
    control_top_k: int,
    risk_review_priority: str,
    include_rationale: bool,
    rationale_mode: Optional[str],
) -> Dict[str, Any]:
    """Build an application-safe Recommended Controls response.

    Frontend-facing output is always `recommended_controls`. Backend/audit fields
    keep origin flags so we can distinguish official taxonomy categories from
    generated measures without exposing taxonomy gaps to the client organisation.
    """
    pathway = detect_hazard_pathway(description)
    controls = get_controls_for_risk(risk_category)

    if risk_review_priority == "high" and SUSPEND_CONTROL_ON_HIGH_RISK:
        return generated_fallback_response(
            description=description,
            risk_category=risk_category,
            control_top_k=control_top_k,
            risk_review_priority=risk_review_priority,
            reason_internal="Risk review priority is high and automatic taxonomy-backed Control scoring was suspended; generated decision-support controls were returned instead.",
        )

    if not controls:
        return generated_fallback_response(
            description=description,
            risk_category=risk_category,
            control_top_k=control_top_k,
            risk_review_priority=risk_review_priority,
            reason_internal="No risk-specific source Control Category definitions were available for the selected Risk context; generated controls were used as application-safe fallback.",
        )

    ranked = score_control_candidates(description, risk_category, controls)
    selected, pathway = select_complementary_controls(ranked, description, risk_category, control_top_k)

    if not selected:
        return generated_fallback_response(
            description=description,
            risk_category=risk_category,
            control_top_k=control_top_k,
            risk_review_priority=risk_review_priority,
            reason_internal="Risk-specific Control definitions were loaded, but the selection layer did not identify a usable control set; generated controls were used as application-safe fallback.",
        )

    status, priority, reason = review_from_scores(ranked[: max(control_top_k, 2)], CONTROL_MIN_REL_SCORE, CONTROL_MIN_MARGIN)
    provisional = risk_review_priority in {"medium", "high"}
    if provisional and priority != "high":
        status = "review"
        priority = "medium" if risk_review_priority == "medium" else "high"
        reason = f"Control recommendations are provisional because upstream Risk review priority is {risk_review_priority}. {reason}"

    mode = (rationale_mode or RATIONALE_MODE or "template").lower().strip()
    candidate_by_label = {normalise_key(c.category): c for c in controls}
    recs: List[Dict[str, Any]] = []
    for row in selected:
        candidate = candidate_by_label.get(normalise_key(row.get("label", ""))) or ControlCandidate(category=row.get("label", ""), definition=row.get("definition", ""))
        recs.append(taxonomy_recommendation_to_app_item(row, candidate, description, risk_category, provisional, include_rationale, mode))

    # If the taxonomy-backed list is weak or fails to cover the detected high-confidence
    # pathway, append generated measures. This is shown to the UI as the same unified
    # Recommended Controls list, while origin_flag preserves internal traceability.
    generated_added: List[Dict[str, Any]] = []
    covered = coverage_summary(recs)
    if pathway.get("pathway_id") == "wet_slip_fall":
        needs_generated = not (
            covered.get("hazard_source_control_included")
            and covered.get("isolation_or_access_control_included")
            and covered.get("administrative_or_warning_control_included")
        )
        if needs_generated:
            existing_texts = {normalise_key(r.get("control_text", "")) for r in recs}
            gen = generated_controls_for_pathway(description, risk_category, pathway, limit=control_top_k)
            for g in gen:
                if len(recs) >= control_top_k:
                    break
                if normalise_key(g.get("control_text", "")) in existing_texts:
                    continue
                g["rank"] = len(recs) + 1
                recs.append(g)
                generated_added.append(g)

    # If after the above the list is still empty, use generated fallback.
    if not recs:
        return generated_fallback_response(
            description=description,
            risk_category=risk_category,
            control_top_k=control_top_k,
            risk_review_priority=risk_review_priority,
            reason_internal="No application-facing controls remained after taxonomy scoring and diversity selection; generated controls were used as fallback.",
        )

    return {
        "mode": "application_safe_recommended_controls",
        "display_mode": "recommended_controls",
        "risk_category_used": risk_category,
        "hazard_pathway": pathway,
        "overall_control_logic_statement": overall_control_logic_statement(description, pathway),
        "recommended_controls": recs,
        # Backward-compatible alias for earlier testing/report scripts.
        "recommended_control_set": recs,
        "taxonomy_backed_controls": [r for r in recs if r.get("origin_flag") == CONTROL_ORIGIN_TAXONOMY],
        "generated_control_measures": [r for r in recs if r.get("origin_flag") == CONTROL_ORIGIN_GENERATED],
        "raw_likelihood_top_k": ranked[:control_top_k],
        "coverage_summary": coverage_summary(recs),
        "internal_control_source_summary": make_control_source_summary(recs),
        "control_source_status": (
            "taxonomy_and_generated_controls_used" if any(r.get("origin_flag") == CONTROL_ORIGIN_GENERATED for r in recs) and any(r.get("origin_flag") == CONTROL_ORIGIN_TAXONOMY for r in recs)
            else "generated_controls_used" if any(r.get("origin_flag") == CONTROL_ORIGIN_GENERATED for r in recs)
            else "taxonomy_controls_used"
        ),
        "review_required": bool(priority in {"medium", "high"} or any(r.get("origin_flag") == CONTROL_ORIGIN_GENERATED for r in recs)),
        "review_status": status,
        "review_priority": priority,
        "reason": reason,
        "internal_reason": (
            "Taxonomy-backed controls were scored and selected. Generated controls were appended only if needed to cover the detected hazard pathway."
            if generated_added else
            "Taxonomy-backed controls were scored and selected."
        ),
        "control_skipped": False,
        "selection_note": (
            "The application-facing output is a unified Recommended Controls list. Internal origin flags are retained for audit, "
            "but the UI does not need to expose whether an item came from the source taxonomy or the generated-control layer."
        ),
    }


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    definition_pairs = sum(len(v) for v in CONTROL_BY_RISK.values()) if CONTROL_BY_RISK else 0
    return {
        "status": "ok" if MODEL is not None and TOKENIZER is not None else "loading_or_unavailable",
        "api_version": "v3.4-report8-control-closure-frontend-safe",
        "base_model": BASE_MODEL_NAME,
        "risk_adapter_dir": RISK_ADAPTER_DIR,
        "risk_adapter_loaded": isinstance(MODEL, PeftModel) if MODEL is not None else False,
        "risk_label_count": len(RISK_LABELS),
        "control_risk_key_count": len(CONTROL_BY_RISK),
        "control_definition_pair_count": definition_pairs,
        "rationale_mode_default": RATIONALE_MODE,
        "risk_candidate_batch_size": RISK_CANDIDATE_BATCH_SIZE,
        "control_candidate_batch_size": CONTROL_CANDIDATE_BATCH_SIZE,
        "auth_enabled": bool(API_TOKEN),
        "main_endpoint": "/predict_risk_control",
    }


@app.get("/frontend_contract")
def frontend_contract(_: None = Depends(require_auth)) -> Dict[str, Any]:
    return {
        "api_version": "v3.4-report8-control-closure-frontend-safe",
        "endpoint_role": "Control recommendation endpoint for Report 8 closure",
        "risk_display_policy": {
            "display_raw_risk_prediction": False,
            "display_final_risk_from_this_endpoint": False,
            "reason": (
                "The raw Risk scoring path in this Control endpoint is retained for internal audit only. "
                "Frontend Risk display should use the Report 7 validated Risk API or a separately validated Risk pipeline."
            ),
        },
        "control_display_policy": {
            "display_recommended_controls": True,
            "display_control_context_as_routing_context_only": True,
            "do_not_label_control_context_as_final_predicted_risk": True,
            "recommended_frontend_label": "Recommended Control",
            "hide_internal_origin_fields": [
                "origin_flag",
                "origin_label",
                "taxonomy_backed",
                "raw_likelihood_top_k",
                "raw_risk_prediction_internal",
            ],
        },
        "recommended_frontend_fields": [
            "frontend_output.control_context_used",
            "frontend_output.overall_control_logic_statement",
            "frontend_output.recommended_controls",
            "frontend_output.review_status",
            "frontend_output.review_priority",
        ],
    }


@app.post("/predict_risk")
def predict_risk(req: PredictRiskRequest, _: None = Depends(require_auth)) -> Dict[str, Any]:
    started = time.time()
    risk = score_risk(req.description, req.location, req.risk_top_k)
    risk["latency_seconds"] = round(time.time() - started, 3)
    return risk


@app.post("/predict_control_from_risk")
def predict_control_from_risk(req: PredictControlFromRiskRequest, _: None = Depends(require_auth)) -> Dict[str, Any]:
    started = time.time()
    out = build_control_recommendation_set(
        description=req.description,
        risk_category=req.risk_category,
        control_top_k=req.control_top_k,
        risk_review_priority=req.risk_review_priority,
        include_rationale=req.include_rationale,
        rationale_mode=req.rationale_mode,
    )
    out["latency_seconds"] = round(time.time() - started, 3)
    return out


@app.post("/predict_risk_control")
def predict_risk_control(req: PredictRiskControlRequest, _: None = Depends(require_auth)) -> Dict[str, Any]:
    started = time.time()
    risk = score_risk(req.description, req.location, req.risk_top_k)

    selected_risk, risk_for_control_selection = choose_risk_for_control(
        description=req.description,
        risk_prediction=risk,
        confirmed_risk=req.confirmed_risk,
    )

    # Do not silently hide a Risk/Pathway conflict. The Risk prediction remains visible,
    # and review priority is raised when Control uses a pathway-preferred Risk.
    if risk_for_control_selection.get("override_applied"):
        old_reason = risk.get("reason", "")
        risk["review_status"] = "review"
        if risk.get("review_priority") != "high":
            risk["review_priority"] = "medium"
        risk["reason"] = (old_reason + " " + risk_for_control_selection.get("reason", "")).strip()

    if not selected_risk:
        control = {
            "mode": "multi_control_rationale_grounded",
            "risk_category_used": None,
            "hazard_pathway": detect_hazard_pathway(req.description),
            "recommended_control_set": [],
            "review_status": "review",
            "review_priority": "high",
            "reason": "Risk prediction was unavailable, so Control recommendation could not be produced.",
            "control_skipped": True,
        }
    else:
        control = build_control_recommendation_set(
            description=req.description,
            risk_category=selected_risk,
            control_top_k=req.control_top_k,
            risk_review_priority=risk.get("review_priority", "high"),
            include_rationale=req.include_rationale,
            rationale_mode=req.rationale_mode,
        )

    combined_priority = "low"
    if risk.get("review_priority") == "high" or control.get("review_priority") == "high":
        combined_priority = "high"
    elif risk.get("review_priority") == "medium" or control.get("review_priority") == "medium":
        combined_priority = "medium"

    # v3.4 frontend-safe response policy:
    # - The raw Risk output is retained only for internal audit/debugging.
    # - It should not be displayed as the final Risk prediction by the frontend.
    # - Formal Risk display should continue to use the Report 7 validated Risk API / pipeline.
    raw_risk_internal = dict(risk)
    raw_risk_internal["field_status"] = "internal_diagnostic_only"
    raw_risk_internal["display_to_frontend"] = False
    raw_risk_internal["interpretation"] = (
        "Raw pre-guardrail Risk scoring output from the Control endpoint. "
        "Do not use this field as the final displayed Risk Category. "
        "Use the Report 7 validated Risk API or a separately validated Risk pipeline for frontend Risk display."
    )

    frontend_recommended_controls = []
    for rec in control.get("recommended_controls", control.get("recommended_control_set", [])) or []:
        clean = {
            "rank": rec.get("rank"),
            "frontend_label": rec.get("frontend_label", "Recommended Control"),
            "control_text": rec.get("control_text"),
            "control_function": rec.get("control_function"),
            "control_role": rec.get("control_role"),
            "priority": rec.get("priority"),
            "provisional": rec.get("provisional"),
            "supporting_event_statement": rec.get("supporting_event_statement"),
            "rationale": rec.get("rationale"),
        }
        if rec.get("official_control_category"):
            clean["official_control_category"] = rec.get("official_control_category")
        frontend_recommended_controls.append(clean)

    return {
        "api_version": "v3.4-report8-control-closure-frontend-safe",
        "input": {
            "description": req.description,
            "location": req.location,
            "confirmed_risk_used": req.confirmed_risk,
        },
        "frontend_contract": {
            "endpoint_role": "Control recommendation endpoint",
            "display_raw_risk_prediction": False,
            "display_final_risk_from_this_endpoint": False,
            "risk_display_source": "Use Report 7 validated Risk API / validated Risk pipeline for final Risk display.",
            "display_control_context_as_routing_context_only": True,
            "do_not_label_control_context_as_final_predicted_risk": True,
            "display_recommended_controls": True,
            "hide_internal_origin_fields": True,
        },
        "frontend_output": {
            "control_context_used": risk_for_control_selection.get("risk_used_for_control"),
            "control_context_source": risk_for_control_selection.get("source"),
            "control_context_is_provisional": risk_for_control_selection.get("provisional"),
            "pathway_id": (risk_for_control_selection.get("pathway") or {}).get("pathway_id"),
            "overall_control_logic_statement": control.get("overall_control_logic_statement"),
            "recommended_controls": frontend_recommended_controls,
            "review_status": "auto" if combined_priority == "low" else "review",
            "review_priority": combined_priority,
        },
        "risk_for_control_selection": risk_for_control_selection,
        "control_recommendation": control,
        "raw_risk_prediction_internal": raw_risk_internal,
        # Legacy field retained for compatibility with existing test scripts; marked as internal.
        "risk_prediction": raw_risk_internal,
        "combined_review_status": "auto" if combined_priority == "low" else "review",
        "combined_review_priority": combined_priority,
        "latency_seconds": round(time.time() - started, 3),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_risk_control_combined_v3_multicontrol_appsafe_statement_guardrails_frontendsafe:app", host="0.0.0.0", port=9001, reload=False)


# =============================================================================
# External mining guardrail configuration override (Report 8 v3.4)
# =============================================================================
# This section is intentionally appended after the original v3.2 functions so the
# existing API can be extended without rewriting the full file. Python resolves
# global function names at call time, so the endpoint functions above will use the
# definitions below when requests are processed.

GUARDRAIL_RULES_JSON = os.getenv(
    "GUARDRAIL_RULES_JSON",
    "config/risk_pathway_control_guardrails_mining_v3_manual_handling.json",
)

_GUARDRAIL_CACHE: Optional[Dict[str, Any]] = None
_ORIGINAL_DETECT_HAZARD_PATHWAY = detect_hazard_pathway
_ORIGINAL_OVERALL_CONTROL_LOGIC_STATEMENT = overall_control_logic_statement
_ORIGINAL_BUILD_SUPPORTING_EVENT_STATEMENT = build_supporting_event_statement
_ORIGINAL_GENERATED_CONTROLS_FOR_PATHWAY = generated_controls_for_pathway
_ORIGINAL_PATHWAY_KEYWORD_BOOST = pathway_keyword_boost


def load_guardrail_config() -> Dict[str, Any]:
    """Load external mining guardrails once. If the file is missing, fail safe by using built-in v3.2 rules."""
    global _GUARDRAIL_CACHE
    if _GUARDRAIL_CACHE is not None:
        return _GUARDRAIL_CACHE
    path = Path(GUARDRAIL_RULES_JSON)
    if not path.exists():
        _GUARDRAIL_CACHE = {"version": "missing", "rules": []}
        return _GUARDRAIL_CACHE
    try:
        _GUARDRAIL_CACHE = load_json(path)
    except Exception as exc:
        _GUARDRAIL_CACHE = {"version": f"load_failed:{type(exc).__name__}", "rules": []}
    return _GUARDRAIL_CACHE


def _external_rule_matches(text: str, rule: Dict[str, Any]) -> bool:
    low = str(text or "").lower()
    include_any = [str(x).lower() for x in rule.get("include_any", []) if str(x).strip()]
    exclude_any = [str(x).lower() for x in rule.get("exclude_any", []) if str(x).strip()]
    if include_any and not any(term in low for term in include_any):
        return False
    if exclude_any and any(term in low for term in exclude_any):
        return False
    return True


def _rule_to_pathway(rule: Dict[str, Any]) -> Dict[str, Any]:
    desired = rule.get("desired_control_functions") or PATHWAY_FUNCTION_PRIORITY.get("generic", [])
    return {
        "pathway_id": rule.get("pathway_id", "generic"),
        "preferred_risk": rule.get("preferred_risk"),
        "secondary_risk": rule.get("secondary_risk"),
        "confidence": rule.get("confidence", "high"),
        "summary": rule.get("summary") or rule.get("overall_control_logic_statement") or "External mining guardrail matched.",
        "hazard_factors": rule.get("hazard_factors", []),
        "desired_control_functions": desired,
        "overall_control_logic_statement": rule.get("overall_control_logic_statement"),
        "generated_fallback_controls": rule.get("generated_fallback_controls", []),
        "taxonomy_preferred_keywords": rule.get("taxonomy_preferred_keywords", []),
        "guardrail_rule_priority": rule.get("priority"),
        "guardrail_source": "external_json",
    }


def detect_hazard_pathway(description: str) -> Dict[str, Any]:  # type: ignore[override]
    """External guardrail-aware pathway detector.

    The external JSON file is evaluated first by priority. If no rule matches, the
    built-in v3.2 detector remains available as a fallback.
    """
    cfg = load_guardrail_config()
    rules = cfg.get("rules", []) if isinstance(cfg, dict) else []
    text = str(description or "")
    matched = [r for r in rules if isinstance(r, dict) and _external_rule_matches(text, r)]
    if matched:
        matched.sort(key=lambda r: int(r.get("priority", 0)), reverse=True)
        return _rule_to_pathway(matched[0])
    return _ORIGINAL_DETECT_HAZARD_PATHWAY(description)


def overall_control_logic_statement(description: str, pathway: Optional[Dict[str, Any]] = None) -> str:  # type: ignore[override]
    pathway = pathway or detect_hazard_pathway(description)
    statement = pathway.get("overall_control_logic_statement") if isinstance(pathway, dict) else None
    if statement:
        return str(statement)
    return _ORIGINAL_OVERALL_CONTROL_LOGIC_STATEMENT(description, pathway)


def _external_support_statement(pathway_id: str, control_function: str, control_text: str) -> Optional[str]:
    # Function-specific statements for mining pathways that are not covered by the original v3.2 builder.
    pid = pathway_id or "generic"
    if pid == "fall_from_height":
        if control_function == "engineering_environmental_control":
            return f"In this fall-from-height scenario, {control_text} helps prevent a worker from reaching or falling from an elevated edge or platform."
        if control_function == "ppe_personal_protection":
            return f"In this fall-from-height scenario, {control_text} reduces the likelihood or consequence of a fall when work near an exposed edge cannot be fully eliminated."
        if control_function == "isolation_access_control":
            return f"In this fall-from-height scenario, {control_text} reduces exposure by keeping workers away from the open-edge or elevated work area until controls are verified."
        if control_function == "administrative_procedure":
            return f"In this fall-from-height scenario, {control_text} supports work-at-height planning, authorisation, supervision, and pre-start verification."
        if control_function == "inspection_maintenance":
            return f"In this fall-from-height scenario, {control_text} helps confirm that platforms, ladders, scaffolds, anchorage points, or edge controls remain suitable before work continues."
    if pid == "hot_work_fire":
        if control_function == "hazard_source_control":
            return f"In this hot-work fire scenario, {control_text} controls the fuel or ignition pathway before sparks can ignite nearby material."
        if control_function == "administrative_procedure":
            return f"In this hot-work fire scenario, {control_text} verifies that the task is authorised and that pre-start fire controls are in place before hot work begins."
        if control_function == "engineering_environmental_control":
            return f"In this hot-work fire scenario, {control_text} reduces the chance that sparks or heat can spread to combustible materials."
        if control_function == "emergency_response":
            return f"In this hot-work fire scenario, {control_text} reduces consequence severity if ignition occurs during or after the hot work."
        if control_function == "isolation_access_control":
            return f"In this hot-work fire scenario, {control_text} separates people, combustibles, or other activities from the hot-work area."
    if pid == "geotechnical_ground_failure":
        if control_function == "isolation_access_control":
            return f"In this geotechnical or ground-failure scenario, {control_text} keeps workers and vehicles away from the unstable edge or cracked ground until it is assessed."
        if control_function == "inspection_maintenance":
            return f"In this geotechnical or ground-failure scenario, {control_text} verifies ground stability before access or work continues."
        if control_function == "engineering_environmental_control":
            return f"In this geotechnical or ground-failure scenario, {control_text} provides physical separation or stabilisation around the unstable area."
        if control_function == "administrative_procedure":
            return f"In this geotechnical or ground-failure scenario, {control_text} redirects movement or controls work planning around the unstable ground condition."
        if control_function == "warning_signage":
            return f"In this geotechnical or ground-failure scenario, {control_text} warns workers before they enter the unstable ground or pit-edge area."
    if pid == "mechanical_energy_isolation":
        if control_function == "isolation_access_control":
            return f"In this mechanical energy-isolation scenario, {control_text} removes the uncontrolled energy pathway before clearing, maintenance, or blockage removal."
        if control_function == "administrative_procedure":
            return f"In this mechanical energy-isolation scenario, {control_text} verifies that workers follow isolation, lockout, and intervention procedures before contact with plant."
        if control_function == "engineering_environmental_control":
            return f"In this mechanical energy-isolation scenario, {control_text} reduces contact with moving, rotating, or crushing points if access control fails."
        if control_function == "inspection_maintenance":
            return f"In this mechanical energy-isolation scenario, {control_text} helps identify defective or uncontrolled plant conditions before work continues."
    if pid == "electrical_energy":
        if control_function == "isolation_access_control":
            return f"In this electrical energy scenario, {control_text} prevents exposure to live or potentially live electrical sources."
        if control_function == "administrative_procedure":
            return f"In this electrical energy scenario, {control_text} verifies authorisation, isolation, and test-for-dead requirements before electrical work continues."
        if control_function == "ppe_personal_protection":
            return f"In this electrical energy scenario, {control_text} reduces arc-flash or shock consequence where exposure cannot be fully eliminated by isolation."
    if pid == "vehicle_pedestrian_interaction":
        # Reuse meaning from the old built-in vehicle_pedestrian id.
        return _ORIGINAL_BUILD_SUPPORTING_EVENT_STATEMENT("", "vehicle_pedestrian", control_function, control_text)
    if pid == "lifting_operation":
        return _ORIGINAL_BUILD_SUPPORTING_EVENT_STATEMENT("", "lifting", control_function, control_text)
    if pid == "loss_of_containment_chemical":
        return _ORIGINAL_BUILD_SUPPORTING_EVENT_STATEMENT("", "chemical_release_exposure", control_function, control_text)
    return None


def build_supporting_event_statement(description: str, pathway_id: str, control_function: str, control_text: str) -> str:  # type: ignore[override]
    # If a generated fallback control carries a custom template in the external rule, use it exactly.
    pathway = detect_hazard_pathway(description)
    for ctrl in pathway.get("generated_fallback_controls", []) or []:
        if normalise_key(ctrl.get("control_text", "")) == normalise_key(control_text):
            template = ctrl.get("supporting_event_statement_template")
            if template:
                return str(template)
    custom = _external_support_statement(pathway_id, control_function, control_text)
    if custom:
        return custom
    return _ORIGINAL_BUILD_SUPPORTING_EVENT_STATEMENT(description, pathway_id, control_function, control_text)


def generated_controls_for_pathway(description: str, risk_category: str, pathway: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:  # type: ignore[override]
    # External JSON-generated controls are used when provided. They are app-facing
    # recommended controls, not official taxonomy categories.
    fallback = pathway.get("generated_fallback_controls") if isinstance(pathway, dict) else None
    if fallback:
        evidence_general = [truncate_text(description, 180)] if description else []
        recs: List[Dict[str, Any]] = []
        for idx, ctrl in enumerate(fallback[: max(1, limit)], start=1):
            fn = ctrl.get("control_function", "general_control")
            text = ctrl.get("control_text", "Recommended control")
            supporting = ctrl.get("supporting_event_statement_template") or build_supporting_event_statement(description, pathway.get("pathway_id", "generic"), fn, text)
            rationale = (
                f"This control is recommended for the {risk_category} pathway because it addresses the {fn.replace('_', ' ')} part of the incident control logic. "
                f"The supporting incident evidence is: {'; '.join(extract_evidence(description, fn) or evidence_general)}."
            )
            recs.append({
                "rank": idx,
                "control_text": text,
                "frontend_label": "Recommended Control",
                "official_control_category": None,
                "control_category": None,
                "control_function": fn,
                "control_role": ctrl.get("control_role", control_role_from_function(fn)),
                "priority": ctrl.get("priority", "medium"),
                "provisional": True,
                "origin_flag": CONTROL_ORIGIN_GENERATED,
                "origin_label": "generated_measure",
                "taxonomy_backed": False,
                "evidence_from_description": extract_evidence(description, fn) or evidence_general,
                "supporting_event_statement": supporting,
                "rationale": rationale,
            })
        return recs
    return _ORIGINAL_GENERATED_CONTROLS_FOR_PATHWAY(description, risk_category, pathway, limit)


def pathway_keyword_boost(label: str, definition: str, control_function: str, pathway: Dict[str, Any]) -> Tuple[float, float, List[str]]:  # type: ignore[override]
    boost, penalty, reasons = _ORIGINAL_PATHWAY_KEYWORD_BOOST(label, definition, control_function, pathway)
    text = normalise_key(f"{label} {definition}")
    pref = [normalise_key(x) for x in pathway.get("taxonomy_preferred_keywords", []) or []]
    if pref:
        matches = [x for x in pref if x and x in text]
        if matches:
            boost += min(0.55, 0.18 + 0.05 * len(matches))
            reasons.append("matches external guardrail preferred taxonomy keywords")
    return boost, penalty, reasons
