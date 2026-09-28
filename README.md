# WHS Risk and Control Classification with Large Language Models

This repository contains the implementation of an LLM-based framework for workplace health and safety (WHS) incident risk classification and associated control classification.

The repository focuses on the core research code, including data preparation, model training, inference, guardrails, evaluation, and API deployment. Private WHS records, model weights, checkpoints, intermediate outputs, and authentication credentials are intentionally excluded.

## Features

- Risk category classification using Llama 3.1 8B Instruct
- RoBERTa-based risk classification baseline
- Control category classification
- Definition-guided control candidate scoring
- Top-K risk prediction with rationale generation
- Risk-control guardrails
- Combined FastAPI interface for risk and control inference
- Single-case and multi-case API test scripts

## Repository Structure

```text
WHS-Risk-Control-LLM/
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

The LLM experiments use Meta Llama 3.1 8B Instruct as the base model.

The repository also includes a RoBERTa-based risk classification baseline.

Fine-tuned model weights and checkpoints are not included in this repository.

## Data

The WHS incident dataset used during development is not included because of data access and privacy restrictions.

Users should provide their own data following the input formats expected by the training, data preparation, and inference scripts.

The repository assumes the following general local directory layout when running experiments:

```text
data/
models/
outputs/
```

These directories are excluded from version control where appropriate.

## Installation

Clone the repository and install the required packages:

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

### RoBERTa risk baseline

```bash
python training/train_risk_roberta.py
```

Command-line arguments can be used to override the default data, model, and output paths defined in the scripts.

## Inference

### Batch risk prediction

```bash
python inference/predict_risk.py
```

### Top-K risk prediction with rationale

```bash
python inference/predict_risk_topk.py
```

### Definition-guided control prediction

```bash
python inference/predict_control.py
```

### Control evaluation

```bash
python inference/evaluate_control.py
```

## API

The combined risk-control API is implemented in:

```text
api/main.py
```

A typical launch command is:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API supports the risk and control inference workflow used in the project.

Model locations and supporting configuration files can be adjusted according to the deployment environment.

## Guardrails

Risk-control pathway guardrails are defined in:

```text
config/risk_pathway_control_guardrails_mining_v3_manual_handling.json
```

These rules are used by the combined API to constrain and refine risk-control pathway outputs.

## Testing

The repository contains scripts for testing the API and guardrail behaviour:

```bash
python tests/test_api_single.py
python tests/test_api_multi.py
python tests/test_guardrails.py
```

Some test scripts assume that the API service and required model resources are already available.

## Excluded Files

The following files are intentionally not included in the repository:

- Private WHS incident datasets
- Fine-tuned model weights
- Llama and RoBERTa checkpoints
- Hugging Face cache files
- Authentication tokens and environment secrets
- Large prediction outputs
- Intermediate experiment files
- Manually reviewed private records

This keeps the repository focused on the reproducible implementation while avoiding redistribution of restricted data or large model artifacts.
