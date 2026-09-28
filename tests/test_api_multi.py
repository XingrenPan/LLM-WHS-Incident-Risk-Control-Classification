#!/usr/bin/env python3
"""
Multi-case test for the Report 8 v3.4 frontend-safe Control API.

Purpose:
1. Test multiple mining / WHS incident scenarios.
2. Verify that raw Risk output is marked internal-only and not frontend-facing.
3. Verify that frontend_output provides a Control-routing context and Recommended Controls.
4. Save outputs for review/reporting.

Run:
  cd WHS-Risk-Control-LLM
  source env_report7.sh
  python run_frontendsafe_multicase_test.py

Outputs:
  frontendsafe_multicase_results.json
  frontendsafe_multicase_summary.txt
  frontendsafe_multicase_table.csv
"""

import csv
import json
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter

API_BASE = "http://127.0.0.1:9001"
API_TOKEN = "abc123"

CASES = [
    {
        "case_id": "case_01_fall_from_height",
        "expected_control_context": "Fall from height",
        "description": "A worker was working on an elevated platform near an open edge without properly attaching the safety harness. The worker lost balance but was stopped by a nearby coworker before falling.",
        "location": "Mine processing plant",
    },
    {
        "case_id": "case_02_dropped_object",
        "expected_control_context": "Dropped / Falling Object",
        "description": "A spanner was dropped from a maintenance platform during work at height and landed close to a worker walking below.",
        "location": "Mine maintenance area",
    },
    {
        "case_id": "case_03_vehicle_pedestrian",
        "expected_control_context": "Vehicles & Mobile Equipment",
        "description": "A light vehicle reversed near a workshop entrance and almost struck a pedestrian who was walking behind the vehicle.",
        "location": "Mine workshop",
    },
    {
        "case_id": "case_04_energy_isolation",
        "expected_control_context": "Energy release (excl. Electrical)",
        "description": "A worker was clearing material from a conveyor belt when the equipment unexpectedly started because the energy isolation was not correctly applied.",
        "location": "Conveyor maintenance area",
    },
    {
        "case_id": "case_05_confined_space",
        "expected_control_context": "Confined Space",
        "description": "A worker entered a tank for inspection before gas testing was completed. A low oxygen alarm activated while the worker was inside.",
        "location": "Processing plant tank",
    },
    {
        "case_id": "case_06_lifting_operation",
        "expected_control_context": "Lifting",
        "description": "During a lifting operation, a worn sling was used to lift a heavy pump component. The suspended load swung close to workers standing inside the lifting area.",
        "location": "Mine workshop lifting area",
    },
    {
        "case_id": "case_07_loss_of_containment_chemical",
        "expected_control_context": "Loss of Containment",
        "description": "An acid transfer hose disconnected during pumping, releasing liquid onto the floor and splashing a nearby worker on the arm.",
        "location": "Chemical transfer area",
    },
    {
        "case_id": "case_08_ground_failure_edge",
        "expected_control_context": "Geotechnical / Ground Failure",
        "description": "Cracks were observed near the edge of an open pit haul road, but light vehicles continued to travel close to the unstable edge.",
        "location": "Open pit haul road",
    },
    {
        "case_id": "case_09_crusher_blockage",
        "expected_control_context": "Energy release (excl. Electrical)",
        "description": "A worker attempted to manually clear a blockage from a crusher without completing isolation. The machine could have moved while the worker's hands were near the crushing point.",
        "location": "Crusher area",
    },
    {
        "case_id": "case_10_hot_work_fire",
        "expected_control_context": "Non Process Fire & Explosion",
        "description": "Welding work was carried out near combustible packaging materials in the workshop. Sparks from the hot work landed close to the stored materials.",
        "location": "Mine workshop",
    },
    {
        "case_id": "case_11_electrical_arc_flash",
        "expected_control_context": "Electrical (incl. Arc Flash/Blast)",
        "description": "An electrician opened a live switchboard for inspection and was exposed to an arc flash hazard because the electrical isolation was incomplete.",
        "location": "Electrical room",
    },
    {
        "case_id": "case_12_slip_trip_fall",
        "expected_control_context": "Slip / Trip / Fall",
        "description": "A worker slipped on wet concrete near the wash bay and fell while walking through the area.",
        "location": "Wash bay",
    },
    {
        "case_id": "case_13_manual_handling",
        "expected_control_context": "Manual Handling / Ergonomics",
        "description": "A worker strained their lower back while manually lifting a heavy pump component without using mechanical assistance.",
        "location": "Maintenance workshop",
    },
]


def request_json(path, method="GET", payload=None, timeout=180):
    url = API_BASE + path
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {
            "error": "HTTPError",
            "status": e.code,
            "body": e.read().decode("utf-8", errors="replace"),
        }
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


def safe_get(obj, path, default=None):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def control_names(frontend_output):
    controls = frontend_output.get("recommended_controls", []) if isinstance(frontend_output, dict) else []
    names = []
    for c in controls:
        names.append(c.get("control_text") or c.get("official_control_category") or c.get("frontend_label") or "")
    return names


def main():
    health = request_json("/health", method="GET", timeout=60)
    contract_endpoint = request_json("/frontend_contract", method="GET", timeout=60)

    records = []
    full_responses = []

    for idx, case in enumerate(CASES, 1):
        print(f"Running {idx}/{len(CASES)}: {case['case_id']}")

        payload = {
            "description": case["description"],
            "location": case["location"],
            "risk_top_k": 5,
            "control_top_k": 5,
            "include_rationale": True,
        }
        response = request_json("/predict_risk_control", method="POST", payload=payload, timeout=180)

        full_responses.append({
            "case": case,
            "request_payload": payload,
            "response": response,
        })

        if "error" in response:
            rec = {
                "case_id": case["case_id"],
                "expected_control_context": case["expected_control_context"],
                "error": json.dumps(response, ensure_ascii=False),
                "api_version": "",
                "raw_top1": "",
                "raw_display_to_frontend": "",
                "frontend_contract_display_raw_risk": "",
                "frontend_contract_display_final_risk": "",
                "control_context_used": "",
                "control_context_source": "",
                "pathway_id": "",
                "override_applied": "",
                "review_status": "",
                "review_priority": "",
                "recommended_controls_count": 0,
                "recommended_controls": "",
                "expected_context_match": False,
                "frontend_safe_pass": False,
                "overall_pass": False,
            }
            records.append(rec)
            continue

        raw = response.get("raw_risk_prediction_internal", {})
        frontend_contract = response.get("frontend_contract", {})
        frontend_output = response.get("frontend_output", {})
        rsel = response.get("risk_for_control_selection", {})
        pathway = rsel.get("pathway", {}) if isinstance(rsel, dict) else {}

        controls = control_names(frontend_output)
        expected = case["expected_control_context"]
        actual_context = frontend_output.get("control_context_used")

        expected_context_match = actual_context == expected

        # Frontend-safety checks: these are more important than raw Risk correctness in v3.4.
        raw_display = raw.get("display_to_frontend")
        display_raw_flag = frontend_contract.get("display_raw_risk_prediction")
        display_final_risk_flag = frontend_contract.get("display_final_risk_from_this_endpoint")
        do_not_label_context = frontend_contract.get("do_not_label_control_context_as_final_predicted_risk")
        has_frontend_controls = len(controls) > 0

        frontend_safe_pass = (
            raw_display is False
            and display_raw_flag is False
            and display_final_risk_flag is False
            and do_not_label_context is True
            and has_frontend_controls
        )

        overall_pass = frontend_safe_pass and expected_context_match

        rec = {
            "case_id": case["case_id"],
            "expected_control_context": expected,
            "error": "",
            "api_version": response.get("api_version"),
            "raw_top1": raw.get("predicted_risk_category"),
            "raw_display_to_frontend": raw_display,
            "frontend_contract_display_raw_risk": display_raw_flag,
            "frontend_contract_display_final_risk": display_final_risk_flag,
            "control_context_used": actual_context,
            "control_context_source": frontend_output.get("control_context_source"),
            "pathway_id": frontend_output.get("pathway_id") or pathway.get("pathway_id"),
            "override_applied": rsel.get("override_applied"),
            "review_status": frontend_output.get("review_status"),
            "review_priority": frontend_output.get("review_priority"),
            "recommended_controls_count": len(controls),
            "recommended_controls": " | ".join(controls),
            "expected_context_match": expected_context_match,
            "frontend_safe_pass": frontend_safe_pass,
            "overall_pass": overall_pass,
            "overall_control_logic_statement": frontend_output.get("overall_control_logic_statement", ""),
            "risk_selection_reason": rsel.get("reason", ""),
        }
        records.append(rec)

    output = {
        "health": health,
        "frontend_contract_endpoint": contract_endpoint,
        "cases": CASES,
        "records": records,
        "full_responses": full_responses,
    }

    Path("frontendsafe_multicase_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "case_id",
        "expected_control_context",
        "api_version",
        "raw_top1",
        "raw_display_to_frontend",
        "frontend_contract_display_raw_risk",
        "frontend_contract_display_final_risk",
        "control_context_used",
        "control_context_source",
        "pathway_id",
        "override_applied",
        "review_status",
        "review_priority",
        "recommended_controls_count",
        "recommended_controls",
        "expected_context_match",
        "frontend_safe_pass",
        "overall_pass",
        "overall_control_logic_statement",
        "risk_selection_reason",
        "error",
    ]

    with open("frontendsafe_multicase_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    raw_counts = Counter(r.get("raw_top1") for r in records if not r.get("error"))
    context_counts = Counter(r.get("control_context_used") for r in records if not r.get("error"))
    pathway_counts = Counter(r.get("pathway_id") for r in records if not r.get("error"))

    n = len(records)
    n_error = sum(1 for r in records if r.get("error"))
    n_frontend_safe = sum(1 for r in records if r.get("frontend_safe_pass") is True)
    n_context_match = sum(1 for r in records if r.get("expected_context_match") is True)
    n_overall_pass = sum(1 for r in records if r.get("overall_pass") is True)

    with open("frontendsafe_multicase_summary.txt", "w", encoding="utf-8") as f:
        f.write("Report 8 v3.4 Frontend-Safe Multi-Case Test Summary\n")
        f.write("=" * 100 + "\n\n")

        f.write("Health\n")
        f.write("-" * 100 + "\n")
        f.write(json.dumps(health, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Frontend contract endpoint\n")
        f.write("-" * 100 + "\n")
        f.write(json.dumps(contract_endpoint, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Aggregate results\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total cases: {n}\n")
        f.write(f"Errors: {n_error}\n")
        f.write(f"Frontend-safe pass: {n_frontend_safe}/{n}\n")
        f.write(f"Expected control-context match: {n_context_match}/{n}\n")
        f.write(f"Overall pass: {n_overall_pass}/{n}\n\n")

        f.write("Raw Risk Top-1 distribution\n")
        f.write(json.dumps(dict(raw_counts), indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Control context used distribution\n")
        f.write(json.dumps(dict(context_counts), indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Pathway distribution\n")
        f.write(json.dumps(dict(pathway_counts), indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Case-level table\n")
        f.write("=" * 100 + "\n\n")
        for r in records:
            f.write(f"Case: {r.get('case_id')}\n")
            f.write(f"Expected control context: {r.get('expected_control_context')}\n")
            f.write(f"Raw Risk Top-1 internal: {r.get('raw_top1')}\n")
            f.write(f"Raw Risk display_to_frontend: {r.get('raw_display_to_frontend')}\n")
            f.write(f"Frontend contract display_raw_risk_prediction: {r.get('frontend_contract_display_raw_risk')}\n")
            f.write(f"Frontend contract display_final_risk_from_this_endpoint: {r.get('frontend_contract_display_final_risk')}\n")
            f.write(f"Control context used: {r.get('control_context_used')}\n")
            f.write(f"Control context source: {r.get('control_context_source')}\n")
            f.write(f"Pathway ID: {r.get('pathway_id')}\n")
            f.write(f"Override applied: {r.get('override_applied')}\n")
            f.write(f"Review: {r.get('review_status')} / {r.get('review_priority')}\n")
            f.write(f"Recommended controls count: {r.get('recommended_controls_count')}\n")
            f.write(f"Recommended controls: {r.get('recommended_controls')}\n")
            f.write(f"Expected context match: {r.get('expected_context_match')}\n")
            f.write(f"Frontend-safe pass: {r.get('frontend_safe_pass')}\n")
            f.write(f"Overall pass: {r.get('overall_pass')}\n")
            if r.get("error"):
                f.write(f"ERROR: {r.get('error')}\n")
            f.write(f"Control logic: {r.get('overall_control_logic_statement')}\n")
            f.write(f"Risk selection reason: {r.get('risk_selection_reason')}\n")
            f.write("\n" + "-" * 100 + "\n\n")

        f.write("Interpretation guide\n")
        f.write("-" * 100 + "\n")
        f.write("Frontend-safe pass means the endpoint marks raw Risk as internal-only, blocks frontend Risk display from this endpoint, and returns Recommended Controls.\n")
        f.write("Expected control-context match checks whether the guardrail/control-routing context matches the manually expected context for this test case.\n")
        f.write("Overall pass requires both frontend-safe pass and expected control-context match.\n")
        f.write("For Report 8 closure, frontend-safe pass is the critical API contract check; context mismatches should be reviewed as guardrail coverage limitations rather than raw Risk display failures.\n")

    print("Saved: frontendsafe_multicase_results.json")
    print("Saved: frontendsafe_multicase_summary.txt")
    print("Saved: frontendsafe_multicase_table.csv")
    print()
    print(f"Frontend-safe pass: {n_frontend_safe}/{n}")
    print(f"Expected control-context match: {n_context_match}/{n}")
    print(f"Overall pass: {n_overall_pass}/{n}")
    print("Raw Risk Top-1 distribution:", dict(raw_counts))
    print("Control context distribution:", dict(context_counts))


if __name__ == "__main__":
    main()
