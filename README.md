# LLM-WHS-Incident-Risk-Control-Classification

This repository contains the implementation of an LLM-based framework for workplace health and safety (WHS) incident risk classification and associated control classification/recommendation.

The project includes data preparation, model training, inference, definition-guided control scoring, guardrails, evaluation, and a combined FastAPI service. The current implementation uses Llama 3.1 8B Instruct for the main LLM-based workflows and RoBERTa as a conventional transformer baseline.

## Features

- 23-class WHS Risk Category classification
- Llama 3.1 8B Instruct with 4-bit QLoRA adaptation
- RoBERTa-based Risk classification baseline
- Top-K Risk prediction with rationale generation
- Risk-conditioned Control classification
- Definition-guided Control candidate scoring
- Multi-control recommendation
- Deterministic Risk-to-Control guardrails
- Human-review routing for uncertain or unsupported cases
- Combined FastAPI interface for Risk and Control inference
- Single-case and multi-case API test scripts

## Repository Structure

```text
LLM-WHS-Incident-Risk-Control-Classification/
├── api/
│   └── main.py
├── config/
│   └── risk_pathway_control_guardrails_mining_v3_manual_handling.json
├── data_preparation/
│   ├── audit_control_labels.py
│   └── prepare_control_dataset.py
├── inference/
│   ├── evaluate_control.py
│   ├── predict_control.py
│   ├── predict_risk.py
│   └── predict_risk_topk.py
├── tests/
│   ├── test_api_multi.py
│   ├── test_api_single.py
│   └── test_guardrails.py
├── training/
│   ├── train_control_llama.py
│   ├── train_risk_llama.py
│   └── train_risk_roberta.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Models

The main LLM experiments use **Meta Llama 3.1 8B Instruct** with 4-bit QLoRA adaptation.

The repository also includes **RoBERTa-base** implementations used as conventional transformer baselines for both Risk and restricted Control classification experiments.

Fine-tuned model weights are not required to inspect the implementation and are not included in this repository.

## Data

The WHS incident dataset used during development is not included because of data access and privacy restrictions.

Users should provide their own data following the input formats expected by the data preparation, training, and inference scripts.

The code uses the following general local directory layout:

```text
data/
models/
outputs/
```

## Experimental Results

### Risk Category Classification

Risk Category classification was evaluated on a common held-out set of **4,697 labelled records**.

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Llama-3.1-8B-Instruct + 4-bit QLoRA | **0.86** | **0.86** | **0.70** |
| RoBERTa-base + weighted cross-entropy | 0.79 | 0.80 | 0.63 |
| Word2Vec + logistic regression | 0.71 | 0.74 | 0.56 |
| FastText + logistic regression | 0.72 | 0.75 | 0.57 |
| FastText + CNN | 0.79 | 0.80 | 0.64 |

The Llama Risk model achieved the highest observed overall performance among the compared models on this held-out split. The difference between weighted F1 and macro F1 also indicates that class imbalance and minority-category performance remain important considerations.

Risk outputs are treated as decision-support candidates rather than automatically accepted final labels. Uncertain, unsupported, or ambiguous cases can be routed for human review.

### Risk-Conditioned Control Classification

A restricted supervised Control experiment was conducted for the **Fall from height** Risk Category.

The dataset contained **633 records**, with **443 training records** and **190 held-out test records**. Five Control Categories were evaluated under a fixed Fall-from-height Risk context.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| RoBERTa-base + CB-Focal Loss | 0.89 | 0.71 | 0.76 | 0.73 | 0.90 |
| Llama-3.1-8B + QLoRA candidate scoring | **0.91** | **0.80** | **0.81** | **0.78** | **0.94** |

For the Llama candidate-scoring approach:

- **Top-2 accuracy:** 0.96
- **Top-3 accuracy:** 0.98

The Llama Control experiment used closed-set candidate likelihood scoring rather than unconstrained free-text generation. Each permitted Control label was scored under the same incident, location, Risk context, and candidate set, and the lowest-loss candidate was selected as Top-1.

These statistical results apply only to the **risk-conditioned five-class Control experiment** and should not be interpreted as performance across the complete organisational Control taxonomy.

### Broad Control Recommendation and Guardrail Validation

The broader Control component is designed as a **definition-guided, guarded, multi-control recommendation workflow** rather than a single-label supervised classifier across the complete taxonomy.

The application-facing workflow is:

```text
Incident description + optional location
        ↓
Internal Risk scoring
        ↓
Hazard-pathway guardrail routing
        ↓
Control-routing context
        ↓
Definition-guided Control ranking
        ↓
Fallback / complementary Control assembly
        ↓
Frontend-safe recommended controls
        ↓
Review status and priority
```

The final frontend-safe API was evaluated using **13 deliberately designed WHS/mining scenarios** after the manual-handling guardrail update.

| Validation item | Result |
|---|---:|
| Frontend-safe pass | 13 / 13 |
| Expected Control-context match | 13 / 13 |
| Overall targeted scenario pass | 13 / 13 |

This targeted validation verifies the expected API contract, guardrail routing, and selected Control-context behaviours for the designed scenarios. It is an acceptance/sanity check and **not** a statistical estimate of general performance across all WHS incidents.

The final system therefore keeps Risk classification and Control recommendation conceptually separated: Risk predictions support routing, while the Control component combines bounded candidate ranking, deterministic guardrails, complementary multi-control assembly, fallback handling, and human-review routing.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Access to Llama 3.1 8B Instruct may require a Hugging Face account, acceptance of the corresponding model license, and authentication in the execution environment.

## Training

### Risk classification with Llama

```bash
python training/train_risk_llama.py
```

### Control classification with Llama

```bash
python training/train_control_llama.py
```

### RoBERTa Risk baseline

```bash
python training/train_risk_roberta.py
```

Command-line arguments can be used to override the default data, model, and output paths defined in the scripts.

## Inference

### Batch Risk prediction

```bash
python inference/predict_risk.py
```

### Top-K Risk prediction with rationale

```bash
python inference/predict_risk_topk.py
```

### Definition-guided Control prediction

```bash
python inference/predict_control.py
```

### Control evaluation

```bash
python inference/evaluate_control.py
```

## API

The combined Risk-Control API is implemented in:

```text
api/main.py
```

A typical launch command is:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API supports the Risk and Control inference workflows used in the project. Model locations and supporting configuration files can be adjusted according to the deployment environment.

## Guardrails

Risk-to-Control pathway guardrails are defined in:

```text
config/risk_pathway_control_guardrails_mining_v3_manual_handling.json
```

The guardrail layer is used to constrain Control routing to plausible hazard pathways and to identify cases requiring review. Raw Risk scores are retained for internal decision support rather than treated as the final Control decision.

## Testing

The repository contains scripts for single-case, multi-case, and guardrail testing:

```bash
python tests/test_api_single.py
python tests/test_api_multi.py
python tests/test_guardrails.py
```

Some test scripts assume that the API service and required model resources are already available.
