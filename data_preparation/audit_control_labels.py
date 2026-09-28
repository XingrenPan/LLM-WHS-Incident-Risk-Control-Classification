#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_audit_all_control_labels.py

Purpose
-------
Audit Control Category labels across the full risk-enriched Events Inc. table
before training a global / multi-risk Control Category adapter.

This script does NOT train a model. It only checks whether the existing
"What Control Failed" labels are usable as supervised Control Category labels.

Main checks:
1. Treat placeholder values such as 0 / 0.0 / nan / blank as missing.
2. Count valid vs missing control labels overall.
3. Count valid vs missing control labels by risk_for_control_curated.
4. Produce overall Control Category distribution.
5. Produce Risk Category x Control Category distribution.
6. Identify low-frequency global control labels.
7. Identify low-frequency risk-control pairs.
8. Export valid control-labelled records and missing-control samples.
9. Export recommended main-control labels after min-count filtering.

No human-identifying wording is used in output file names or column names.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_MAIN_FILE = (
    "outputs/risk_category_prediction_curated_23class/"
    "full_prediction/risk_category_predictions_87k_curated.csv"
)
DEFAULT_OUTPUT_DIR = "outputs/control_category_audit/all_control_label_audit"

DEFAULT_RISK_COL = "risk_for_control_curated"
DEFAULT_RISK_SOURCE_COL = "risk_for_control_source"
DEFAULT_CONTROL_COL = "What Control Failed"
DEFAULT_TEXT_COL = "WHAT_HAPPENED_ENGLISH"
DEFAULT_LOCATION_COL = "Location"
DEFAULT_EVENT_COL = "Event Number"

MISSING_CONTROL_VALUES = {
    "",
    "-",
    "--",
    "0",
    "0.0",
    "0.00",
    "nan",
    "na",
    "n/a",
    "none",
    "null",
    "missing",
    "unknown",
    "not applicable",
    "not_applicable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all Control Category labels in the full risk-enriched table."
    )
    parser.add_argument("--main-file", type=str, default=DEFAULT_MAIN_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--risk-col", type=str, default=DEFAULT_RISK_COL)
    parser.add_argument("--risk-source-col", type=str, default=DEFAULT_RISK_SOURCE_COL)
    parser.add_argument("--control-col", type=str, default=DEFAULT_CONTROL_COL)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--location-col", type=str, default=DEFAULT_LOCATION_COL)
    parser.add_argument("--event-col", type=str, default=DEFAULT_EVENT_COL)

    parser.add_argument(
        "--min-control-count",
        type=int,
        default=10,
        help="Minimum global count for a Control Category to be retained as a main class.",
    )
    parser.add_argument(
        "--min-risk-control-pair-count",
        type=int,
        default=5,
        help="Minimum count for a risk-control pair to be treated as sufficiently represented.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of missing-control records to sample for inspection.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding for output files.",
    )
    return parser.parse_args()


def normalize_text_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Normalize repeated internal whitespace.
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_control_label(x: Any) -> str:
    """
    Normalize raw Control Category labels without forcing synonym mapping.
    This is an audit-stage normalization, not a final taxonomy mapping.
    """
    s = normalize_text_value(x)
    if not s:
        return ""
    # Remove accidental surrounding quotes.
    s = s.strip("\"'").strip()
    # Normalize common punctuation spacing.
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_missing_control_label(x: Any) -> bool:
    s = normalize_control_label(x)
    if not s:
        return True
    return s.lower() in MISSING_CONTROL_VALUES


def find_column(df: pd.DataFrame, preferred: str, candidates: Optional[List[str]] = None) -> Optional[str]:
    if preferred in df.columns:
        return preferred

    candidates = candidates or []
    for c in candidates:
        if c in df.columns:
            return c

    # Case-insensitive fallback.
    lookup = {str(c).lower(): c for c in df.columns}
    if preferred.lower() in lookup:
        return lookup[preferred.lower()]
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]

    return None


def value_counts_df(series: pd.Series, name_col: str, count_col: str = "count") -> pd.DataFrame:
    vc = series.value_counts(dropna=False).reset_index()
    vc.columns = [name_col, count_col]
    return vc


def safe_records_for_export(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    return df[cols].copy()


def main() -> None:
    args = parse_args()

    main_file = Path(args.main_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not main_file.exists():
        raise FileNotFoundError(f"Main file not found: {main_file}")

    df = pd.read_csv(main_file, low_memory=False)
    total_rows = len(df)

    risk_col = find_column(df, args.risk_col, ["risk_for_control", "SAFETY_RISK_CATEGORY", "predicted_risk_category_23class"])
    risk_source_col = find_column(df, args.risk_source_col, ["risk_label_source", "risk_for_control_source"])
    control_col = find_column(df, args.control_col, ["WHAT_CONTROL_FAILED", "what_control_failed", "Control Category", "control_category"])
    text_col = find_column(df, args.text_col, ["Description", "description", "input_text"])
    location_col = find_column(df, args.location_col, ["LOCATION", "location"])
    event_col = find_column(df, args.event_col, ["event_number", "Event_Number", "EVENT_NUMBER"])

    if risk_col is None:
        raise ValueError("Could not find a risk column. Please pass --risk-col.")
    if control_col is None:
        raise ValueError("Could not find a control label column. Please pass --control-col.")
    if text_col is None:
        raise ValueError("Could not find a text column. Please pass --text-col.")

    # Core normalization.
    df["_risk_for_audit"] = df[risk_col].apply(normalize_text_value)
    df["_control_raw_for_audit"] = df[control_col].apply(normalize_control_label)
    df["_control_is_missing_for_audit"] = df[control_col].apply(is_missing_control_label)
    df["_control_category_for_audit"] = df["_control_raw_for_audit"].where(
        ~df["_control_is_missing_for_audit"], pd.NA
    )

    valid_mask = ~df["_control_is_missing_for_audit"]
    missing_mask = df["_control_is_missing_for_audit"]

    valid_df = df.loc[valid_mask].copy()
    missing_df = df.loc[missing_mask].copy()

    # Overall label distribution.
    overall_control_dist = value_counts_df(
        valid_df["_control_category_for_audit"], "control_category", "count"
    )
    overall_control_dist.to_csv(out_dir / "control_label_distribution_overall.csv", index=False, encoding=args.encoding)

    # Risk coverage summary.
    risk_total = df.groupby("_risk_for_audit", dropna=False).size().rename("total_rows")
    risk_valid = valid_df.groupby("_risk_for_audit", dropna=False).size().rename("valid_control_rows")
    risk_missing = missing_df.groupby("_risk_for_audit", dropna=False).size().rename("missing_control_rows")
    risk_unique_controls = (
        valid_df.groupby("_risk_for_audit", dropna=False)["_control_category_for_audit"]
        .nunique(dropna=True)
        .rename("unique_valid_control_labels")
    )

    risk_summary = pd.concat([risk_total, risk_valid, risk_missing, risk_unique_controls], axis=1).fillna(0)
    for c in ["total_rows", "valid_control_rows", "missing_control_rows", "unique_valid_control_labels"]:
        risk_summary[c] = risk_summary[c].astype(int)
    risk_summary["valid_control_rate"] = (
        risk_summary["valid_control_rows"] / risk_summary["total_rows"].replace(0, pd.NA)
    ).fillna(0.0)
    risk_summary = risk_summary.reset_index().rename(columns={"_risk_for_audit": "risk_category"})
    risk_summary = risk_summary.sort_values(["valid_control_rows", "total_rows"], ascending=[False, False])
    risk_summary.to_csv(out_dir / "risk_control_coverage_summary.csv", index=False, encoding=args.encoding)

    # Risk x control distribution.
    risk_control_dist = (
        valid_df.groupby(["_risk_for_audit", "_control_category_for_audit"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"_risk_for_audit": "risk_category", "_control_category_for_audit": "control_category"})
        .sort_values(["risk_category", "count"], ascending=[True, False])
    )
    risk_control_dist.to_csv(out_dir / "control_label_distribution_by_risk.csv", index=False, encoding=args.encoding)

    # Wide matrix, useful for quick inspection.
    risk_control_matrix = pd.pivot_table(
        valid_df,
        index="_risk_for_audit",
        columns="_control_category_for_audit",
        values=text_col,
        aggfunc="count",
        fill_value=0,
    )
    risk_control_matrix.index.name = "risk_category"
    risk_control_matrix.to_csv(out_dir / "risk_control_matrix.csv", encoding=args.encoding)

    # Main labels after global min count.
    main_control_labels_df = overall_control_dist.loc[
        overall_control_dist["count"] >= args.min_control_count
    ].copy()
    main_control_labels = main_control_labels_df["control_category"].astype(str).tolist()
    main_control_labels_df.to_csv(out_dir / "main_control_labels_min_count.csv", index=False, encoding=args.encoding)

    low_frequency_control_labels = overall_control_dist.loc[
        overall_control_dist["count"] < args.min_control_count
    ].copy()
    low_frequency_control_labels.to_csv(out_dir / "low_frequency_control_labels.csv", index=False, encoding=args.encoding)

    # Low-frequency risk-control pairs.
    low_pair_df = risk_control_dist.loc[
        risk_control_dist["count"] < args.min_risk_control_pair_count
    ].copy()
    low_pair_df.to_csv(out_dir / "low_frequency_risk_control_pairs.csv", index=False, encoding=args.encoding)

    # Main control training candidates.
    main_candidate_mask = valid_df["_control_category_for_audit"].isin(main_control_labels)
    main_candidates = valid_df.loc[main_candidate_mask].copy()
    removed_low_freq_records = valid_df.loc[~main_candidate_mask].copy()

    # Add explicit audit columns to exports.
    for out_df in [valid_df, missing_df, main_candidates, removed_low_freq_records]:
        out_df["control_category_audit"] = out_df["_control_category_for_audit"]
        out_df["control_label_status_audit"] = out_df["_control_is_missing_for_audit"].map(
            {True: "missing_or_placeholder", False: "valid_control_label"}
        )

    export_cols = [
        event_col,
        text_col,
        location_col,
        risk_col,
        risk_source_col,
        control_col,
        "control_category_audit",
        "control_label_status_audit",
    ]
    export_cols = [c for c in export_cols if c is not None]

    safe_records_for_export(valid_df, export_cols).to_csv(
        out_dir / "valid_control_label_records.csv", index=False, encoding=args.encoding
    )
    safe_records_for_export(main_candidates, export_cols).to_csv(
        out_dir / "control_training_candidates_main_labels.csv", index=False, encoding=args.encoding
    )
    safe_records_for_export(removed_low_freq_records, export_cols).to_csv(
        out_dir / "removed_low_frequency_control_records.csv", index=False, encoding=args.encoding
    )

    if len(missing_df) > 0:
        sample_n = min(args.sample_size, len(missing_df))
        missing_sample = missing_df.sample(n=sample_n, random_state=42)
    else:
        missing_sample = missing_df
    safe_records_for_export(missing_sample, export_cols).to_csv(
        out_dir / "missing_control_label_sample.csv", index=False, encoding=args.encoding
    )

    # Risk source distribution if available.
    risk_source_distribution = []
    if risk_source_col is not None and risk_source_col in df.columns:
        source_dist = value_counts_df(df[risk_source_col].apply(normalize_text_value), "risk_for_control_source", "count")
        source_dist.to_csv(out_dir / "risk_for_control_source_distribution.csv", index=False, encoding=args.encoding)
        risk_source_distribution = source_dist.to_dict(orient="records")

    # Recommended training scale by risk category after global main-label filtering.
    main_by_risk = (
        main_candidates.groupby("_risk_for_audit", dropna=False)
        .size()
        .reset_index(name="main_control_training_rows")
        .rename(columns={"_risk_for_audit": "risk_category"})
        .sort_values("main_control_training_rows", ascending=False)
    )
    main_by_risk.to_csv(out_dir / "main_control_training_rows_by_risk.csv", index=False, encoding=args.encoding)

    summary: Dict[str, Any] = {
        "main_file": str(main_file),
        "output_dir": str(out_dir),
        "total_rows": int(total_rows),
        "risk_column": risk_col,
        "risk_source_column": risk_source_col,
        "control_label_column": control_col,
        "text_column": text_col,
        "location_column": location_col,
        "event_column": event_col,
        "missing_control_placeholders_treated_as_missing": sorted(MISSING_CONTROL_VALUES),
        "control_label_audit": {
            "valid_control_rows": int(valid_mask.sum()),
            "missing_control_rows": int(missing_mask.sum()),
            "valid_control_rate": float(valid_mask.sum() / total_rows) if total_rows else 0.0,
            "unique_valid_control_labels": int(valid_df["_control_category_for_audit"].nunique(dropna=True)),
        },
        "main_label_filter": {
            "min_control_count": int(args.min_control_count),
            "retained_main_control_labels": main_control_labels,
            "num_retained_main_control_labels": int(len(main_control_labels)),
            "main_control_training_candidate_rows": int(len(main_candidates)),
            "removed_low_frequency_control_rows": int(len(removed_low_freq_records)),
        },
        "risk_control_coverage_top": risk_summary.head(50).to_dict(orient="records"),
        "overall_control_distribution_top": overall_control_dist.head(50).to_dict(orient="records"),
        "main_control_training_rows_by_risk_top": main_by_risk.head(50).to_dict(orient="records"),
        "risk_for_control_source_distribution": risk_source_distribution,
        "output_files": {
            "control_label_distribution_overall": str(out_dir / "control_label_distribution_overall.csv"),
            "risk_control_coverage_summary": str(out_dir / "risk_control_coverage_summary.csv"),
            "control_label_distribution_by_risk": str(out_dir / "control_label_distribution_by_risk.csv"),
            "risk_control_matrix": str(out_dir / "risk_control_matrix.csv"),
            "main_control_labels_min_count": str(out_dir / "main_control_labels_min_count.csv"),
            "low_frequency_control_labels": str(out_dir / "low_frequency_control_labels.csv"),
            "low_frequency_risk_control_pairs": str(out_dir / "low_frequency_risk_control_pairs.csv"),
            "valid_control_label_records": str(out_dir / "valid_control_label_records.csv"),
            "control_training_candidates_main_labels": str(out_dir / "control_training_candidates_main_labels.csv"),
            "removed_low_frequency_control_records": str(out_dir / "removed_low_frequency_control_records.csv"),
            "missing_control_label_sample": str(out_dir / "missing_control_label_sample.csv"),
            "main_control_training_rows_by_risk": str(out_dir / "main_control_training_rows_by_risk.csv"),
        },
    }

    with open(out_dir / "all_control_label_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone. All-control label audit outputs saved to:")
    print(out_dir)
    print("\nKey summary:")
    print(json.dumps({
        "total_rows": summary["total_rows"],
        "valid_control_rows": summary["control_label_audit"]["valid_control_rows"],
        "missing_control_rows": summary["control_label_audit"]["missing_control_rows"],
        "unique_valid_control_labels": summary["control_label_audit"]["unique_valid_control_labels"],
        "min_control_count": summary["main_label_filter"]["min_control_count"],
        "num_retained_main_control_labels": summary["main_label_filter"]["num_retained_main_control_labels"],
        "main_control_training_candidate_rows": summary["main_label_filter"]["main_control_training_candidate_rows"],
        "removed_low_frequency_control_rows": summary["main_label_filter"]["removed_low_frequency_control_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
