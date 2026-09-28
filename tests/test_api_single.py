#!/usr/bin/env python3
"""
Single-case test for the Report 8 v3.4 frontend-safe Control API.

Run:
  cd WHS-Risk-Control-LLM
  source env_report7.sh
  python run_frontendsafe_single_case_test.py

Outputs:
  frontendsafe_single_case_result.json
  frontendsafe_single_case_summary.txt
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://127.0.0.1:9001/predict_risk_control"
HEALTH_URL = "http://127.0.0.1:9001/health"
CONTRACT_URL = "http://127.0.0.1:9001/frontend_contract"
API_TOKEN = "abc123"

CASE = {
    "case_id": "frontendsafe_confined_space_case",
    "case_name": "Confined Space low oxygen alarm case",
    "description": "A worker entered a tank for inspection before gas testing was completed. A low oxygen alarm activated while the worker was inside.",
    "location": "Processing plant tank",
    "expected_control_context": "Confined Space"
}


def request_json(url, method="GET", payload=None, timeout=180):
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


def main():
    health = request_json(HEALTH_URL, method="GET", timeout=60)
    frontend_contract = request_json(CONTRACT_URL, method="GET", timeout=60)

    payload = {
        "description": CASE["description"],
        "location": CASE["location"],
        "risk_top_k": 5,
        "control_top_k": 5,
        "include_rationale": True,
    }

    response = request_json(API_URL, method="POST", payload=payload, timeout=180)

    output = {
        "case": CASE,
        "request_payload": payload,
        "health": health,
        "frontend_contract_endpoint": frontend_contract,
        "prediction_response": response,
    }

    Path("frontendsafe_single_case_result.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open("frontendsafe_single_case_summary.txt", "w", encoding="utf-8") as f:
        f.write("Report 8 v3.4 Frontend-Safe Single Case Test Summary\n")
        f.write("=" * 90 + "\n\n")

        f.write("Input case\n")
        f.write("-" * 90 + "\n")
        f.write(f"Case ID: {CASE['case_id']}\n")
        f.write(f"Case name: {CASE['case_name']}\n")
        f.write(f"Expected control context: {CASE['expected_control_context']}\n")
        f.write(f"Description: {CASE['description']}\n")
        f.write(f"Location: {CASE['location']}\n\n")

        f.write("Health\n")
        f.write("-" * 90 + "\n")
        f.write(json.dumps(health, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Frontend contract endpoint\n")
        f.write("-" * 90 + "\n")
        f.write(json.dumps(frontend_contract, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Prediction response summary\n")
        f.write("-" * 90 + "\n")
        if "error" in response:
            f.write("ERROR RESPONSE\n")
            f.write(json.dumps(response, indent=2, ensure_ascii=False))
            f.write("\n")
            print("Saved with error response.")
        else:
            f.write(f"API version: {response.get('api_version')}\n")
            f.write(f"Endpoint role: {response.get('endpoint_role')}\n\n")

            contract = response.get("frontend_contract", {})
            f.write("Frontend contract inside prediction response\n")
            f.write(f"display_raw_risk_prediction: {contract.get('display_raw_risk_prediction')}\n")
            f.write(f"display_final_risk_from_this_endpoint: {contract.get('display_final_risk_from_this_endpoint')}\n")
            f.write(f"risk_display_source: {contract.get('risk_display_source')}\n")
            f.write(f"display_recommended_controls: {contract.get('display_recommended_controls')}\n")
            f.write(f"do_not_label_control_context_as_final_predicted_risk: {contract.get('do_not_label_control_context_as_final_predicted_risk')}\n\n")

            raw = response.get("raw_risk_prediction_internal", {})
            f.write("Raw risk prediction internal\n")
            f.write(f"display_to_frontend: {raw.get('display_to_frontend')}\n")
            f.write(f"field_status: {raw.get('field_status')}\n")
            f.write(f"raw predicted risk category: {raw.get('predicted_risk_category')}\n")
            f.write(f"interpretation: {raw.get('interpretation')}\n")
            f.write("raw top-k:\n")
            for r in raw.get("top_k", []):
                f.write(
                    f"  - rank {r.get('rank')}: {r.get('label')} "
                    f"(relative_score={r.get('relative_score')}, avg_nll={r.get('avg_nll')})\n"
                )
            f.write("\n")

            rsel = response.get("risk_for_control_selection", {})
            pathway = rsel.get("pathway", {}) if isinstance(rsel, dict) else {}
            f.write("Risk context used for Control selection\n")
            f.write(f"risk_used_for_control: {rsel.get('risk_used_for_control')}\n")
            f.write(f"source: {rsel.get('source')}\n")
            f.write(f"override_applied: {rsel.get('override_applied')}\n")
            f.write(f"provisional: {rsel.get('provisional')}\n")
            f.write(f"pathway_id: {pathway.get('pathway_id')}\n")
            f.write(f"guardrail_source: {pathway.get('guardrail_source')}\n")
            f.write(f"reason: {rsel.get('reason')}\n\n")

            front = response.get("frontend_output", {})
            f.write("Frontend output\n")
            f.write(f"control_context_used: {front.get('control_context_used')}\n")
            f.write(f"control_context_source: {front.get('control_context_source')}\n")
            f.write(f"control_context_is_provisional: {front.get('control_context_is_provisional')}\n")
            f.write(f"pathway_id: {front.get('pathway_id')}\n")
            f.write(f"review_status: {front.get('review_status')}\n")
            f.write(f"review_priority: {front.get('review_priority')}\n")
            f.write(f"overall_control_logic_statement: {front.get('overall_control_logic_statement')}\n\n")

            controls = front.get("recommended_controls", [])
            f.write(f"Recommended controls count: {len(controls)}\n\n")
            for c in controls:
                f.write(f"[{c.get('rank')}] {c.get('control_text')}\n")
                f.write(f"frontend_label: {c.get('frontend_label')}\n")
                f.write(f"official_control_category: {c.get('official_control_category')}\n")
                f.write(f"control_function: {c.get('control_function')}\n")
                f.write(f"control_role: {c.get('control_role')}\n")
                f.write(f"priority: {c.get('priority')}\n")
                f.write(f"provisional: {c.get('provisional')}\n")
                f.write(f"supporting_event_statement: {c.get('supporting_event_statement')}\n")
                f.write(f"rationale: {c.get('rationale')}\n")
                f.write("\n")

    print("Saved: frontendsafe_single_case_result.json")
    print("Saved: frontendsafe_single_case_summary.txt")


if __name__ == "__main__":
    main()
