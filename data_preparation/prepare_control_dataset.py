#!/usr/bin/env python3
"""
Prepare a curated Fall-from-height Control Category dataset.

Purpose
-------
1) Audit Fall-from-height rows in the risk-enriched 87k table.
2) Treat placeholder values such as "0" as missing Control Category labels.
3) Apply reviewed Control Category corrections from the previous review-audit output.
4) Export clean training candidates for the next Control Category modelling stage.

No human-identifying wording is used in filenames or output fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Canonical Control Category labels used in the Fall-from-height task
# ---------------------------------------------------------------------

CANONICAL_CONTROL_LABELS = [
    "Earth Bunds, Signage and Truck loader stops",
    "Scaffolding and Portable Ladders",
    "Structural Integrity Inspect",
    "Fall Prevention Systems and Rope Access",
    "Mobile Working at Heights Equipment",
    "Other",
    "Fitness for Work",
    "Crisis and Emergency Management",
]

# Normalized aliases -> canonical label
CONTROL_ALIAS_MAP: Dict[str, str] = {
    # Earth bunds / signage / truck loader stops
    "earth bunds signage and truck loader stops": "Earth Bunds, Signage and Truck loader stops",
    "earth bunds signage truck loader stops": "Earth Bunds, Signage and Truck loader stops",
    "earth bunds": "Earth Bunds, Signage and Truck loader stops",
    "bunds signage and truck loader stops": "Earth Bunds, Signage and Truck loader stops",
    "bund signage and truck loader stops": "Earth Bunds, Signage and Truck loader stops",
    "signage and truck loader stops": "Earth Bunds, Signage and Truck loader stops",
    "truck loader stops": "Earth Bunds, Signage and Truck loader stops",

    # Scaffolding / ladders
    "scaffolding and portable ladders": "Scaffolding and Portable Ladders",
    "scaffolding portable ladders": "Scaffolding and Portable Ladders",
    "scaffolding ladders": "Scaffolding and Portable Ladders",
    "portable ladders": "Scaffolding and Portable Ladders",
    "ladders": "Scaffolding and Portable Ladders",

    # Structural integrity
    "structural integrity inspect": "Structural Integrity Inspect",
    "structural integrity": "Structural Integrity Inspect",
    "structure integrity": "Structural Integrity Inspect",
    "structural": "Structural Integrity Inspect",
    "integrity": "Structural Integrity Inspect",

    # Fall prevention / rope access
    "fall prevention systems and rope access": "Fall Prevention Systems and Rope Access",
    "fall prevention system and rope access": "Fall Prevention Systems and Rope Access",
    "fall prevention systems rope access": "Fall Prevention Systems and Rope Access",
    "fall prevention and rope access": "Fall Prevention Systems and Rope Access",
    "fall prevention systems": "Fall Prevention Systems and Rope Access",
    "fall prevention system": "Fall Prevention Systems and Rope Access",
    "fall prevention": "Fall Prevention Systems and Rope Access",
    "rope access": "Fall Prevention Systems and Rope Access",
    "fall prevention systems and rope acces": "Fall Prevention Systems and Rope Access",
    "fall prevention systems rope acces": "Fall Prevention Systems and Rope Access",

    # Mobile working at heights equipment
    "mobile working at heights equipment": "Mobile Working at Heights Equipment",
    "mobile working at height equipment": "Mobile Working at Heights Equipment",
    "mobile work at heights equipment": "Mobile Working at Heights Equipment",
    "mobile equipment at heights": "Mobile Working at Heights Equipment",
    "mobile elevated work platform": "Mobile Working at Heights Equipment",
    "ewp": "Mobile Working at Heights Equipment",
    "mewp": "Mobile Working at Heights Equipment",

    # Other
    "other": "Other",

    # Fitness / crisis
    "fitness for work": "Fitness for Work",
    "fitness": "Fitness for Work",
    "crisis and emergency management": "Crisis and Emergency Management",
    "crisis emergency management": "Crisis and Emergency Management",
    "emergency management": "Crisis and Emergency Management",
}

MISSING_CONTROL_VALUES = {
    "",
    "0",
    "0.0",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "not applicable",
    "unknown",
    "missing",
    "-",
    "--",
}


def normalize_text(x: object) -> str:
    """Lowercase, remove punctuation-like separators, and collapse spaces."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def is_missing_control(x: object) -> bool:
    """Return True for empty / placeholder Control Category values."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    raw = str(x).strip()
    if not raw:
        return True
    norm = normalize_text(raw)
    return norm in MISSING_CONTROL_VALUES


def canonicalize_control_label(x: object) -> Optional[str]:
    """Map a raw control label/comment fragment into a canonical Control Category."""
    if is_missing_control(x):
        return None

    raw = str(x).strip()

    # Exact canonical match first
    for label in CANONICAL_CONTROL_LABELS:
        if raw == label:
            return label

    norm = normalize_text(raw)
    if not norm or norm in MISSING_CONTROL_VALUES:
        return None

    if norm in CONTROL_ALIAS_MAP:
        return CONTROL_ALIAS_MAP[norm]

    # More tolerant contains-based mapping.
    # Order matters: specific categories before generic ones.
    contains_rules: List[Tuple[List[str], str]] = [
        (["fall prevention", "rope access"], "Fall Prevention Systems and Rope Access"),
        (["fall prevention"], "Fall Prevention Systems and Rope Access"),
        (["rope access"], "Fall Prevention Systems and Rope Access"),
        (["structural integrity"], "Structural Integrity Inspect"),
        (["structural"], "Structural Integrity Inspect"),
        (["integrity"], "Structural Integrity Inspect"),
        (["scaffolding", "ladder"], "Scaffolding and Portable Ladders"),
        (["scaffolding"], "Scaffolding and Portable Ladders"),
        (["portable ladder"], "Scaffolding and Portable Ladders"),
        (["ladder"], "Scaffolding and Portable Ladders"),
        (["earth bund"], "Earth Bunds, Signage and Truck loader stops"),
        (["truck loader stop"], "Earth Bunds, Signage and Truck loader stops"),
        (["signage"], "Earth Bunds, Signage and Truck loader stops"),
        (["ewp"], "Mobile Working at Heights Equipment"),
        (["mobile working"], "Mobile Working at Heights Equipment"),
        (["elevated work platform"], "Mobile Working at Heights Equipment"),
        (["fitness for work"], "Fitness for Work"),
        (["emergency management"], "Crisis and Emergency Management"),
        (["other"], "Other"),
    ]

    for keys, label in contains_rules:
        if all(k in norm for k in keys):
            return label

    return None


def clean_review_comment(comment: object) -> str:
    """
    Normalize review comments such as:
      "This is Structural Integrity."
      "Worng control: This is fall prevention systems and rope acces."
      "Correct: Earth Bunds..."
    into a shorter candidate phrase before canonical mapping.
    """
    if comment is None:
        return ""
    if isinstance(comment, float) and math.isnan(comment):
        return ""

    s = str(comment).strip()
    if not s:
        return ""

    # Keep original content but remove common review prefixes.
    s = re.sub(r"(?i)\bworng\b", "wrong", s)
    s = re.sub(r"(?i)\bwrong control\s*:\s*", "", s)
    s = re.sub(r"(?i)\bcorrect\s*:\s*", "", s)
    s = re.sub(r"(?i)^\s*this\s+is\s+", "", s)
    s = re.sub(r"(?i)^\s*it\s+is\s+", "", s)
    s = re.sub(r"(?i)^\s*should\s+be\s+", "", s)
    s = re.sub(r"(?i)^\s*should\s+be\s+classified\s+as\s+", "", s)
    s = re.sub(r"(?i)^\s*classified\s+as\s+", "", s)
    s = re.sub(r"[.。]+$", "", s).strip()
    return s


def event_key(x: object) -> str:
    """Create a stable event key from int-like or float-like event IDs."""
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        if x.is_integer():
            return str(int(x))
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    # Convert "1000123.0" -> "1000123"
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(f"Could not find any of these columns: {list(candidates)}")
    return None


def safe_value_counts(series: pd.Series) -> List[Dict[str, object]]:
    vc = series.value_counts(dropna=False)
    out = []
    for key, count in vc.items():
        if pd.isna(key):
            key = None
        out.append({"value": key, "count": int(count)})
    return out


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def to_jsonl(df: pd.DataFrame, path: Path, text_col: str, label_col: str, risk_col: str, location_col: Optional[str]) -> None:
    """Write simple instruction-style JSONL for later SFT scripts."""
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            desc = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
            risk = "" if pd.isna(row.get(risk_col)) else str(row.get(risk_col)).strip()
            location = ""
            if location_col and location_col in df.columns:
                location = "" if pd.isna(row.get(location_col)) else str(row.get(location_col)).strip()
            answer = str(row.get(label_col)).strip()

            input_parts = [f"Risk Category: {risk}", f"Incident description: {desc}"]
            if location:
                input_parts.append(f"Location: {location}")

            item = {
                "instruction": (
                    "Classify the failed control category for a Fall from height incident. "
                    "Return only one allowed Control Category name."
                ),
                "input": "\n".join(input_parts),
                "output": answer,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curated Fall-from-height Control Category dataset.")
    parser.add_argument(
        "--main-file",
        default="outputs/risk_category_prediction_curated_23class/full_prediction/risk_category_predictions_87k_curated.csv",
        help="Risk-enriched 87k CSV file.",
    )
    parser.add_argument(
        "--review-corrections",
        default="outputs/control_category_audit/fall_from_height_review_audit_v2/fh_reviewed_control_corrections.csv",
        help="Reviewed control correction CSV produced by the v2 audit script.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/control_category_audit/fall_from_height_control_curated_dataset",
        help="Output directory.",
    )
    parser.add_argument("--risk-col", default="risk_for_control_curated")
    parser.add_argument("--risk-value", default="Fall from height")
    parser.add_argument("--control-col", default=None, help="Control label column. If omitted, auto-detected.")
    parser.add_argument("--text-col", default=None, help="Incident description column. If omitted, auto-detected.")
    parser.add_argument("--event-col", default=None, help="Event number/id column. If omitted, auto-detected.")
    parser.add_argument("--location-col", default=None, help="Location column. If omitted, auto-detected if available.")
    parser.add_argument("--min-class-count", type=int, default=10, help="Minimum count for main-class training dataset.")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main_path = Path(args.main_file)
    review_path = Path(args.review_corrections)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not main_path.exists():
        raise FileNotFoundError(f"Main file not found: {main_path}")

    df = pd.read_csv(main_path, low_memory=False)

    risk_col = find_column(df, [args.risk_col, "risk_for_control_curated", "SAFETY_RISK_CATEGORY"])
    control_col = args.control_col or find_column(
        df,
        [
            "What Control Failed",
            "WHAT_CONTROL_FAILED",
            "what_control_failed",
            "Control Category",
            "control_category",
            "control_name",
        ],
    )
    text_col = args.text_col or find_column(
        df,
        ["WHAT_HAPPENED_ENGLISH", "what_happened_english", "input_text", "description", "Description"],
    )
    event_col = args.event_col or find_column(
        df,
        ["event_number", "EVENT_NUMBER", "Event Number", "event_id", "id", "ID"],
        required=False,
    )
    location_col = args.location_col or find_column(
        df,
        ["Location", "location", "LOCATION", "site", "Site"],
        required=False,
    )

    # Filter Fall-from-height rows.
    risk_norm = df[risk_col].astype(str).str.strip().str.lower()
    target_norm = args.risk_value.strip().lower()
    fh = df[risk_norm == target_norm].copy()

    # Clean existing control labels, treating "0" and similar placeholders as missing.
    fh["existing_control_category_raw"] = fh[control_col]
    fh["existing_control_category_clean"] = fh[control_col].apply(canonicalize_control_label)
    fh["existing_control_label_is_valid"] = fh["existing_control_category_clean"].notna()

    fh["curated_control_category"] = fh["existing_control_category_clean"]
    fh["control_curation_source"] = fh["existing_control_label_is_valid"].map(
        {True: "existing_control_label", False: "missing_control_label"}
    )
    fh["review_correction_applied"] = False
    fh["review_decision_clean"] = None
    fh["review_comment"] = None
    fh["review_original_control_category"] = None
    fh["review_model_top1_label"] = None

    # Prepare match keys.
    if event_col:
        fh["_event_key"] = fh[event_col].apply(event_key)
    else:
        fh["_event_key"] = ""
    fh["_text_key"] = fh[text_col].astype(str).map(lambda x: normalize_text(x)[:500])

    applied_rows = []
    unmatched_rows = []

    if review_path.exists():
        rev = pd.read_csv(review_path, low_memory=False)

        rev_event_col = find_column(
            rev,
            ["event_number", "EVENT_NUMBER", "Event Number", "event_id", "id", "ID"],
            required=False,
        )
        rev_text_col = find_column(rev, ["input_text", "WHAT_HAPPENED_ENGLISH", "description"], required=False)
        rev_curated_col = find_column(rev, ["curated_control_category"], required=True)
        rev_decision_col = find_column(rev, ["review_decision_clean", "0.72", "Correct / Wrong"], required=False)
        rev_comment_col = find_column(rev, ["Comments", "comments", "review_comment"], required=False)
        rev_orig_col = find_column(rev, ["true_label", "original_control_category"], required=False)
        rev_top1_col = find_column(rev, ["pred_top1_label"], required=False)

        if rev_event_col:
            rev["_event_key"] = rev[rev_event_col].apply(event_key)
        else:
            rev["_event_key"] = ""
        if rev_text_col:
            rev["_text_key"] = rev[rev_text_col].astype(str).map(lambda x: normalize_text(x)[:500])
        else:
            rev["_text_key"] = ""

        # Apply mapped correction labels.
        for _, r in rev.iterrows():
            raw_curated = r.get(rev_curated_col)
            curated_label = canonicalize_control_label(raw_curated)
            if curated_label is None:
                # Try comment parsing as fallback.
                if rev_comment_col:
                    curated_label = canonicalize_control_label(clean_review_comment(r.get(rev_comment_col)))

            if curated_label is None:
                row_dict = r.to_dict()
                row_dict["unmatched_reason"] = "curated_label_unmapped"
                unmatched_rows.append(row_dict)
                continue

            match_idx = pd.Index([])

            ev_key = r.get("_event_key", "")
            if ev_key:
                match_idx = fh.index[fh["_event_key"] == ev_key]

            # Fallback to text matching if event id is missing or not matched.
            if len(match_idx) == 0:
                txt_key = r.get("_text_key", "")
                if txt_key:
                    match_idx = fh.index[fh["_text_key"] == txt_key]

            if len(match_idx) == 0:
                row_dict = r.to_dict()
                row_dict["unmatched_reason"] = "no_matching_main_row"
                unmatched_rows.append(row_dict)
                continue

            for idx in match_idx:
                fh.at[idx, "curated_control_category"] = curated_label
                fh.at[idx, "control_curation_source"] = "review_correction"
                fh.at[idx, "review_correction_applied"] = True
                if rev_decision_col:
                    fh.at[idx, "review_decision_clean"] = r.get(rev_decision_col)
                if rev_comment_col:
                    fh.at[idx, "review_comment"] = r.get(rev_comment_col)
                if rev_orig_col:
                    fh.at[idx, "review_original_control_category"] = r.get(rev_orig_col)
                if rev_top1_col:
                    fh.at[idx, "review_model_top1_label"] = r.get(rev_top1_col)

                applied = {
                    "main_index": int(idx),
                    "curated_control_category": curated_label,
                    "event_key": fh.at[idx, "_event_key"],
                    "text_key": fh.at[idx, "_text_key"],
                }
                if rev_event_col:
                    applied["review_event_number"] = r.get(rev_event_col)
                if rev_decision_col:
                    applied["review_decision_clean"] = r.get(rev_decision_col)
                if rev_comment_col:
                    applied["review_comment"] = r.get(rev_comment_col)
                applied_rows.append(applied)

    else:
        print(f"[WARN] Review corrections file not found: {review_path}. Continuing without review corrections.")

    # Training candidates
    candidates_all = fh[fh["curated_control_category"].notna()].copy()
    missing_or_unlabelled = fh[fh["curated_control_category"].isna()].copy()

    label_counts_all = candidates_all["curated_control_category"].value_counts().rename_axis("control_category").reset_index(name="count")
    retained_labels = label_counts_all.loc[label_counts_all["count"] >= args.min_class_count, "control_category"].tolist()
    candidates_main = candidates_all[candidates_all["curated_control_category"].isin(retained_labels)].copy()
    removed_low_frequency = candidates_all[~candidates_all["curated_control_category"].isin(retained_labels)].copy()

    # Save audit tables
    fh.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / "fh_control_rows_with_curated_labels.csv", index=False)
    candidates_all.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / "fh_control_training_candidates_curated_all_classes.csv", index=False)
    missing_or_unlabelled.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / "fh_control_missing_or_unlabelled_records.csv", index=False)
    candidates_main.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / f"fh_control_training_candidates_main_classes_min{args.min_class_count}.csv", index=False)
    removed_low_frequency.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / f"fh_control_removed_low_frequency_min{args.min_class_count}.csv", index=False)

    label_counts_all.to_csv(out_dir / "fh_control_label_distribution_curated_all_classes.csv", index=False)
    candidates_main["curated_control_category"].value_counts().rename_axis("control_category").reset_index(name="count").to_csv(
        out_dir / f"fh_control_label_distribution_main_classes_min{args.min_class_count}.csv", index=False
    )
    fh["existing_control_category_raw"].value_counts(dropna=False).rename_axis("raw_control_value").reset_index(name="count").to_csv(
        out_dir / "fh_control_label_distribution_raw_values.csv", index=False
    )
    fh["existing_control_category_clean"].value_counts(dropna=False).rename_axis("clean_existing_control_category").reset_index(name="count").to_csv(
        out_dir / "fh_control_label_distribution_existing_clean.csv", index=False
    )

    pd.DataFrame(applied_rows).to_csv(out_dir / "fh_control_review_corrections_applied.csv", index=False)
    pd.DataFrame(unmatched_rows).to_csv(out_dir / "fh_control_review_corrections_unmatched.csv", index=False)

    # Make a simple train/test split for the retained main classes.
    train_rows = 0
    test_rows = 0
    split_status = "not_created"

    if len(candidates_main) > 0 and candidates_main["curated_control_category"].nunique() >= 2:
        try:
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(
                candidates_main,
                test_size=args.test_size,
                random_state=args.seed,
                stratify=candidates_main["curated_control_category"],
            )
            split_status = "created_stratified"
        except Exception as e:
            # Fallback if stratification fails for any unexpected reason.
            train_df = candidates_main.sample(frac=1 - args.test_size, random_state=args.seed)
            test_df = candidates_main.drop(train_df.index)
            split_status = f"created_random_fallback: {type(e).__name__}: {e}"

        train_rows = len(train_df)
        test_rows = len(test_df)

        train_df.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / "risk_fh_control_train.csv", index=False)
        test_df.drop(columns=["_event_key", "_text_key"], errors="ignore").to_csv(out_dir / "risk_fh_control_test.csv", index=False)

        to_jsonl(train_df, out_dir / "risk_fh_control_train.jsonl", text_col, "curated_control_category", risk_col, location_col)
        to_jsonl(test_df, out_dir / "risk_fh_control_test.jsonl", text_col, "curated_control_category", risk_col, location_col)

        train_df["curated_control_category"].value_counts().rename_axis("control_category").reset_index(name="count").to_csv(
            out_dir / "train_label_distribution.csv", index=False
        )
        test_df["curated_control_category"].value_counts().rename_axis("control_category").reset_index(name="count").to_csv(
            out_dir / "test_label_distribution.csv", index=False
        )

    summary = {
        "main_file": str(main_path),
        "review_corrections_file": str(review_path),
        "output_dir": str(out_dir),
        "risk_column": risk_col,
        "risk_value": args.risk_value,
        "control_label_column": control_col,
        "text_column": text_col,
        "event_column": event_col,
        "location_column": location_col,
        "total_rows": int(len(df)),
        "fall_from_height_rows": int(len(fh)),
        "raw_control_valid_after_zero_as_missing": int(fh["existing_control_label_is_valid"].sum()),
        "raw_control_missing_after_zero_as_missing": int((~fh["existing_control_label_is_valid"]).sum()),
        "review_corrections_applied_rows": int(len(applied_rows)),
        "review_corrections_unmatched_rows": int(len(unmatched_rows)),
        "curated_control_available_rows": int(len(candidates_all)),
        "curated_control_missing_rows": int(len(missing_or_unlabelled)),
        "unique_curated_control_labels_all_classes": int(candidates_all["curated_control_category"].nunique()),
        "curated_control_distribution_all_classes": label_counts_all.to_dict(orient="records"),
        "min_class_count": int(args.min_class_count),
        "retained_main_control_labels": retained_labels,
        "main_class_training_candidate_rows": int(len(candidates_main)),
        "removed_low_frequency_rows": int(len(removed_low_frequency)),
        "split_status": split_status,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "missing_control_placeholders_treated_as_missing": sorted(MISSING_CONTROL_VALUES),
        "canonical_control_labels": CANONICAL_CONTROL_LABELS,
    }
    write_json(out_dir / "fh_control_curated_dataset_summary.json", summary)

    print("Done. Curated Fall-from-height Control Category dataset saved to:")
    print(out_dir)
    print("\nKey summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
