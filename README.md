# semeval26-05-scripts
Some scripts for Semeval 2026 Task 5. Equivalent to the scoring script on the CodaBench task. More baselines to be added later.

Link to submission website: https://www.codabench.org/competitions/10877/?secret_key=e3c13419-88c6-4c13-9989-8e694a2bc5c0

# How to evaluate predictions

First, remember to install the requirements.

To evaluate a prediction, please format it like the "predictions/[...].jsonl" files.
Each prediction must be in its own line. The "id" key corresponds to the keys of the samples in the gold data ("0", "1", etc).
The prediction key should be an integer between 1 and 5.

Once you prepared your prediction data, put it in the input/res/ folder (replacing the existing file) and call the evaluation script like this:

```
python scoring.py input/ref/solution.jsonl input/res/predictions.jsonl output/scores.json
```

Scores will be printed and written on output/scores.json. If your predictions file contains bad formatting or is incomplete, it will print an error.

To submit to CodaBench, zip the predictions.jsonl up and upload it to the "My Submissions" tab on the task website.

Test set is yet unreleased, so you can only test on the dev set for now. The samples (including labels) are public here: https://github.com/Janosch-Gehring/ambistory

## Project Overview

This repository contains a collection of small scripts used to evaluate
predictions for Semeval 2026 Task 5, produce simple baselines, and prepare
or validate dataset files (cleaned/minified versions, verification,
format checks, and scoring). The scripts are intentionally lightweight and
can be combined or modified for different evaluation workflows.

## Repository structure (high level)
- `data/` — gold/dev/train data in JSON/JSONL formats
- `input/` — example inputs and predictions used by scripts
- `output/` — generated outputs and scores
- `scripts/` — (top-level) Python scripts used for processing, evaluation,
	and generation of smaller dataset artifacts

The repository's primary scripts are described below.

## Scripts & What they do
- `scoring.py` — Score JSONL predictions against a JSONL reference. Computes
	Spearman correlation and the accuracy-within-standard-deviation metric. Writes
	results to a JSON output file and prints the metrics.
- `evaluate.py` — Alternative evaluation script to compute Spearman and the
	accuracy-within-SD metric against a dataset stored as `data/<split>.json`.
- `format_check.py` — Validate a predictions JSONL for format issues (ids,
	prediction values, duplicates, missing ids).
- `verify.py` — Quick JSONL validator that asserts id order, range of
	predictions (1-5), and number of entries (expected number is 588).
- `chat.py` — Wrapper for calling HF Inference / chat models; sends a JSON
	object describing each sample (context and judged meaning) to a model and
	records the returned likelihood (1–5). Uses `huggingface-hub.InferenceClient`.
- `clean_dev_json.py` — Produce a cleaned (scoring fields dropped) version of
	`data/dev.json` for use in tasks that require anonymized or simplified
	input.
- `make_majority_dev.py` — Generate a simplified dataset that takes the
	majority vote among annotators (ties broken by closeness to the average).
- `minify.py` — Utility to produce compact (minified) JSON files.

## Dependencies & How to install
By default, the repository uses `requirements.txt` to list required
libraries. Install them inside a virtual environment before running the
scripts:

PowerShell example:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt
```

Important notes on dependencies:
- `scipy` (Spearman correlation used by `evaluate.py` and `scoring.py`)
- `huggingface-hub` and `requests` (used by `chat.py` to call HF models)
- `pydantic` is used by `chat.py` for the `PredictionResponse` model when
	requesting structured JSON output; however, some runs might not execute the
	Pydantic-dependent paths, so `pydantic` is optional if you don't use the
	`RESPONSE_FORMAT` constant or the `model_dump_json()` call.
- `numpy` is typically included since SciPy depends on it.

The repository's `requirements.txt` provides a convenient starting point for
the full feature set.

## Running the main evaluation (`scoring.py`) — Example
```powershell
python .\scoring.py input/ref/solution.jsonl input/res/predictions.jsonl output/scores.json
```

This prints Spearman + accuracy results and writes a small `scores.json` with
the results.

## Running the HF `chat.py` script
This script calls a chat model for each entry and writes a JSONL of
predictions. It requires a Hugging Face token if you use remote HF endpoints.

Example :
```python chat.py --token TON_FINE_GRAINED_TOKEN
```


## Validation & Verification
- Use `format_check.py` to ensure your predictions file is formatted properly.
- Use `verify.py` to assert an expected id sequence and count (588 entries by
	default in the script).
