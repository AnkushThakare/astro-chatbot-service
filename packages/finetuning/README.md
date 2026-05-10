# Finetuning Module

This module is a beginner-friendly starting point for fine-tuning a small open-source instruct model on **astrology interpretation style**.

The model in this package does **not** calculate kundli, chart positions, dashas, or astrology math. Your backend or chart engine should generate chart JSON first. The fine-tuned model only learns how to **explain the provided chart data** in a structured, careful, safe style.

## What This Module Does

- validates training and evaluation datasets
- fine-tunes a small instruct model with LoRA / QLoRA using Unsloth
- saves a lightweight LoRA adapter
- tests the adapter on a few sample astrology prompts

## Why We Use LoRA / QLoRA

LoRA and QLoRA are useful here because they:

- reduce GPU memory usage
- train only small adapter weights instead of the full model
- are practical for free Colab or Kaggle GPUs
- keep the base model reusable

## Why The Model Does Not Calculate Kundli

This project separates responsibilities clearly:

- backend/chart engine:
  creates kundli or chart JSON
- fine-tuned model:
  interprets that chart JSON in a human-friendly astrology style

This reduces hallucination risk and keeps training focused on explanation quality instead of chart calculation.

## Folder Layout

```text
packages/finetuning/
  data/
    train.jsonl
    eval.jsonl
  scripts/
    validate_dataset.py
    train_unsloth.py
    test_adapter.py
    evaluate_adapter.py
    compare_models.py
    run_full_cycle.py
  outputs/
    .gitkeep
    eval_runs/
      .gitkeep
    comparison_runs/
      .gitkeep
  requirements.txt
  README.md
  COLAB_QUICKSTART.md
  colab_run_full_cycle.ipynb
  kaggle_run_full_cycle.ipynb
```

## Dataset Format

Each JSONL row must look like this:

```json
{
  "instruction": "Analyze career from this Vedic chart.",
  "input": "Chart JSON or object serialized as text",
  "output": "High-quality astrology interpretation"
}
```

Recommended answer style:

1. Direct Summary
2. Chart Evidence
3. Interpretation
4. Practical Guidance
5. Confidence Level
6. Disclaimer

## Install Dependencies

```bash
pip install -r packages/finetuning/requirements.txt
```

## How To Validate Dataset

```bash
python packages/finetuning/scripts/validate_dataset.py \
  --train packages/finetuning/data/train.jsonl \
  --eval packages/finetuning/data/eval.jsonl
```

What it checks:

- valid JSONL format
- required keys: `instruction`, `input`, `output`
- non-empty strings or JSON-serializable objects
- warns when outputs are very short
- exact duplicate rows

It exits non-zero if invalid rows exist.

## How To Train On Colab / Kaggle

This script expects a CUDA-enabled NVIDIA GPU.

Safe default settings for free GPUs:

- `max_seq_length=2048`
- `epochs=1`
- `learning_rate=2e-4`
- `batch_size=1`
- `grad_accum=4`

Example:

```bash
python packages/finetuning/scripts/train_unsloth.py \
  --train_file packages/finetuning/data/train.jsonl \
  --eval_file packages/finetuning/data/eval.jsonl \
  --output_dir packages/finetuning/outputs/astro-lora-v1 \
  --save_merged_16bit
```

Configurable flags:

- `--model_name`
- `--train_file`
- `--eval_file`
- `--output_dir`
- `--max_seq_length`
- `--epochs`
- `--learning_rate`
- `--batch_size`
- `--grad_accum`
- `--save_merged_16bit`

Recommended workflow:

1. Validate the dataset.
2. Train one small LoRA run on Colab or Kaggle.
3. Test the adapter on 3 sample prompts.
4. Evaluate the adapter on the held-out eval set.
5. Compare base vs adapter before touching production code.

If you want one command for the whole workflow, use:

```bash
python packages/finetuning/scripts/run_full_cycle.py \
  --train_file packages/finetuning/data/train.jsonl \
  --eval_file packages/finetuning/data/eval.jsonl \
  --output_dir packages/finetuning/outputs/astro-lora-v1 \
  --save_merged_16bit
```

There is also a notebook-ready version in
[COLAB_QUICKSTART.md](./COLAB_QUICKSTART.md).

There is also a ready-to-run notebook file at
[colab_run_full_cycle.ipynb](./colab_run_full_cycle.ipynb).

For Kaggle, use
[kaggle_run_full_cycle.ipynb](./kaggle_run_full_cycle.ipynb).

## How To Test Adapter

After training:

```bash
python packages/finetuning/scripts/test_adapter.py \
  --adapter_dir packages/finetuning/outputs/astro-lora-v1
```

The script will:

- load the base model
- load your LoRA adapter
- run 3 sample astrology prompts
- print model outputs
- print latency in milliseconds

This makes it easier to compare adapter behavior between checkpoints.

## How To Evaluate A Trained Adapter

Run a held-out pass on `eval.jsonl` and save the predictions:

```bash
python packages/finetuning/scripts/evaluate_adapter.py \
  --eval_file packages/finetuning/data/eval.jsonl \
  --adapter_dir packages/finetuning/outputs/astro-lora-v1
```

What it reports:

- per-example output
- latency in milliseconds
- whether all 6 expected sections are present
- whether confidence and disclaimer sections were included
- a saved JSONL prediction file under `outputs/eval_runs/`

If you want a baseline before the adapter, leave out `--adapter_dir` to score the base model only.

## How To Compare Base Model vs Adapter

```bash
python packages/finetuning/scripts/compare_models.py \
  --eval_file packages/finetuning/data/eval.jsonl \
  --adapter_dir packages/finetuning/outputs/astro-lora-v1
```

This produces a markdown report under `outputs/comparison_runs/` with:

- the eval input
- reference output
- base-model output
- adapter output
- simple section/disclaimer heuristics for each

This is the easiest way to decide whether the adapter is actually improving explanation style.

## Common Errors

### CUDA / GPU unavailable

Cause:
- running on CPU or a machine without NVIDIA CUDA

Fix:
- run on Colab, Kaggle, or another CUDA-enabled environment

### bitsandbytes import errors

Cause:
- CUDA mismatch or broken wheel

Fix:
- reinstall dependencies
- try a standard Colab or Kaggle notebook environment

### Out of memory

Fixes:

- lower `--max_seq_length`
- keep `--batch_size 1`
- increase `--grad_accum`
- start with `epochs=1`

### Dataset validation errors

Cause:
- invalid JSONL
- missing keys
- empty fields

Fix:
- run `validate_dataset.py`
- correct the reported rows before training

## Next Steps For Connecting To FastAPI

Later, you can connect this module to your backend like this:

- backend generates chart JSON
- FastAPI sends chart JSON plus the user instruction to the fine-tuned model
- model returns the interpretation text
- backend still handles safety, chart generation, and business logic

Suggested request shape for the app:

```json
{
  "instruction": "Analyze career from this Vedic chart.",
  "chart_json": {
    "lagna": "Capricorn",
    "houses": {
      "10": {
        "sign": "Libra",
        "planets": ["Mercury", "Sun"]
      }
    }
  }
}
```

Suggested integration rule:

- keep the current planner, tool routing, and chart engine outside the fine-tuned model
- only call the fine-tuned model after chart JSON already exists
- use the adapter only for interpretation wording and structure
- keep safety, auth, product, consultant, and booking logic in the backend

Recommended next step:

- keep chart generation outside the model
- use the adapter only for interpretation quality
- compare base model vs adapter on real chatbot examples before deployment
- start with `eval.jsonl`, then expand to 20-30 held-out chart examples before production decisions
