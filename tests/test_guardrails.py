#!/usr/bin/env python3
import json
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://127.0.0.1:9001/predict_risk_control"
API_TOKEN = "abc123"

CASES = [
    {
        "case_id": "case_01_fall_from_height",
        "case_name": "Fall from height near open edge",
        "description": "A worker was working on an elevated platform near an open edge without properly attaching the safety harness. The worker lost balance but was stopped by a nearby coworker before falling.",
        "location": "Mine processing plant",
        "expected_risk_direction": "Fall from height"
    },
    {
        "case_id": "case_02_dropped_object",
        "case_name": "Dropped object from maintenance platform",
        "description": "A spanner was dropped from a maintenance platform during work at height and landed close to a worker walking below.",
        "location": "Mine maintenance area",
        "expected_risk_direction": "Dropped / Falling Object"
    },
    {
        "case_id": "case_03_vehicle_pedestrian",
        "case_name": "Vehicle pedestrian near miss",
        "description": "A light vehicle reversed near a workshop entrance and almost struck a pedestrian who was walking behind the vehicle.",
        "location": "Mine workshop",
        "expected_risk_direction": "Vehicles & Mobile Equipment"
    },
    {
        "case_id": "case_04_energy_isolation",
        "case_name": "Unexpected conveyor start during maintenance",
        "description": "A worker was clearing material from a conveyor belt when the equipment unexpectedly started because the energy isolation was not correctly applied.",
        "location": "Conveyor maintenance area",
        "expected_risk_direction": "Energy release / Isolation"
    },
    {
        "case_id": "case_05_confined_space",
        "case_name": "Tank entry before gas testing",
        "description": "A worker entered a tank for inspection before gas testing was completed. A low oxygen alarm activated while the worker was inside.",
        "location": "Processing plant tank",
        "expected_risk_direction": "Confined Space"
    },
    {
        "case_id": "case_06_lifting_operation",
        "case_name": "Worn sling and swinging load",
        "description": "During a lifting operation, a worn sling was used to lift a heavy pump component. The suspended load swung close to workers standing inside the lifting area.",
        "location": "Mine workshop lifting area",
        "expected_risk_direction": "Lifting"
    },
    {
        "case_id": "case_07_loss_of_containment_chemical",
        "case_name": "Acid hose disconnection and splash exposure",
        "description": "An acid transfer hose disconnected during pumping, releasing liquid onto the floor and splashing a nearby worker on the arm.",
        "location": "Chemical transfer area",
        "expected_risk_direction": "Loss of Containment / Acute Chemical Exposure"
    },
    {
        "case_id": "case_08_ground_failure_edge",
        "case_name": "Cracked pit edge with vehicle access",
        "description": "Cracks were observed near the edge of an open pit haul road, but light vehicles continued to travel close to the unstable edge.",
        "location": "Open pit haul road",
        "expected_risk_direction": "Geotechnical / Ground Failure"
    },
    {
        "case_id": "case_09_crusher_blockage",
        "case_name": "Crusher blockage cleared without isolation",
        "description": "A worker attempted to manually clear a blockage from a crusher without completing isolation. The machine could have moved while the worker's hands were near the crushing point.",
        "location": "Crusher area",
        "expected_risk_direction": "Energy release / Mechanical hazard"
    },
    {
        "case_id": "case_10_hot_work_fire",
        "case_name": "Hot work near combustible materials",
        "description": "Welding work was carried out near combustible packaging materials in the workshop. Sparks from the hot work landed close to the stored materials.",
        "location": "Mine workshop",
        "expected_risk_direction": "Non Process Fire & Explosion / Hot work"
    }
]

def post_case(case):
    payload = {
        "description": case["description"],
        "location": case["location"],
        "risk_top_k": 3,
        "control_top_k": 5,
        "include_rationale": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        return {"error": "HTTPError", "status": e.code, "body": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}

results = []

for i, case in enumerate(CASES, 1):
    print(f"Running {i}/{len(CASES)}: {case['case_id']} - {case['case_name']}")
    response = post_case(case)
    results.append({
        "case_id": case["case_id"],
        "case_name": case["case_name"],
        "expected_risk_direction": case["expected_risk_direction"],
        "input": {"description": case["description"], "location": case["location"]},
        "response": response
    })

Path("mining_10case_guardrail_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

with open("mining_10case_guardrail_summary.txt", "w", encoding="utf-8") as f:
    f.write("Mining 10-Case Report 8 Guardrail API Test Summary\n")
    f.write("=" * 80 + "\n\n")

    for item in results:
        response = item["response"]

        f.write("=" * 80 + "\n")
        f.write(f"{item['case_id']}: {item['case_name']}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Expected risk direction: {item['expected_risk_direction']}\n")
        f.write(f"Input description: {item['input']['description']}\n")
        f.write(f"Location: {item['input']['location']}\n\n")

        if "error" in response:
            f.write("ERROR RESPONSE\n")
            f.write(json.dumps(response, indent=2, ensure_ascii=False))
            f.write("\n\n")
            continue

        f.write(f"API version: {response.get('api_version')}\n\n")

        risk = response.get("risk_prediction", {})
        f.write("Risk Prediction\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-1 Risk: {risk.get('predicted_risk_category')}\n")
        f.write(f"Review: {risk.get('review_status')} / {risk.get('review_priority')}\n")
        f.write(f"Reason: {risk.get('reason')}\n")
        f.write("Top-K Risk:\n")
        for r in risk.get("top_k", []):
            f.write(
                f"  - rank {r.get('rank')}: {r.get('label')} "
                f"(relative_score={r.get('relative_score')}, avg_nll={r.get('avg_nll')})\n"
            )
        f.write("\n")

        rsel = response.get("risk_for_control_selection", {})
        pathway = rsel.get("pathway", {}) if isinstance(rsel, dict) else {}
        f.write("Risk Used for Control Selection\n")
        f.write("-" * 40 + "\n")
        f.write(f"Risk used for control: {rsel.get('risk_used_for_control')}\n")
        f.write(f"Source: {rsel.get('source')}\n")
        f.write(f"Override applied: {rsel.get('override_applied')}\n")
        f.write(f"Provisional: {rsel.get('provisional')}\n")
        f.write(f"Pathway ID: {pathway.get('pathway_id')}\n")
        f.write(f"Guardrail source: {pathway.get('guardrail_source')}\n")
        f.write(f"Guardrail priority: {pathway.get('guardrail_rule_priority')}\n")
        f.write(f"Reason: {rsel.get('reason')}\n\n")

        ctrl = response.get("control_recommendation", {})
        f.write("Control Recommendation\n")
        f.write("-" * 40 + "\n")
        f.write(f"Display mode: {ctrl.get('display_mode')}\n")
        f.write(f"Risk category used: {ctrl.get('risk_category_used')}\n")
        f.write(f"Control source status: {ctrl.get('control_source_status')}\n")
        f.write(f"Review status: {ctrl.get('review_status')} / {ctrl.get('review_priority')}\n")
        f.write(f"Review required: {ctrl.get('review_required')}\n")
        f.write(f"Reason: {ctrl.get('reason')}\n")
        f.write(f"Internal reason: {ctrl.get('internal_reason')}\n\n")

        f.write("Overall control logic statement:\n")
        f.write((ctrl.get("overall_control_logic_statement") or "None") + "\n\n")

        f.write("Coverage summary:\n")
        f.write(json.dumps(ctrl.get("coverage_summary", {}), indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Internal source summary:\n")
        f.write(json.dumps(ctrl.get("internal_control_source_summary", {}), indent=2, ensure_ascii=False))
        f.write("\n\n")

        recs = ctrl.get("recommended_controls", [])
        f.write(f"Recommended controls count: {len(recs)}\n\n")

        for rec in recs:
            f.write(f"[{rec.get('rank')}] {rec.get('control_text')}\n")
            f.write(f"frontend_label: {rec.get('frontend_label')}\n")
            f.write(f"official_control_category: {rec.get('official_control_category')}\n")
            f.write(f"control_function: {rec.get('control_function')}\n")
            f.write(f"control_role: {rec.get('control_role')}\n")
            f.write(f"priority: {rec.get('priority')}\n")
            f.write(f"provisional: {rec.get('provisional')}\n")
            f.write(f"origin_flag: {rec.get('origin_flag')}\n")
            f.write(f"origin_label: {rec.get('origin_label')}\n")
            f.write(f"taxonomy_backed: {rec.get('taxonomy_backed')}\n")
            f.write(f"evidence: {rec.get('evidence_from_description')}\n")
            f.write("supporting_event_statement:\n")
            f.write((rec.get("supporting_event_statement") or "None") + "\n")
            f.write("rationale:\n")
            f.write((rec.get("rationale") or "None") + "\n")
            f.write("\n")

        f.write("Combined review:\n")
        f.write(f"{response.get('combined_review_status')} / {response.get('combined_review_priority')}\n\n")

print("Saved: mining_10case_guardrail_results.json")
print("Saved: mining_10case_guardrail_summary.txt")
