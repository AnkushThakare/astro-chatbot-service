# Colab / Kaggle Quickstart

Use this when you want the first end-to-end QLoRA run with minimal setup.

## One Notebook Cell

Paste this into a Colab or Kaggle notebook cell:

```bash
git clone <your-repo-url>
cd astro-chatbot-service
pip install -r packages/finetuning/requirements.txt

python packages/finetuning/scripts/run_full_cycle.py \
  --train_file packages/finetuning/data/train.jsonl \
  --eval_file packages/finetuning/data/eval.jsonl \
  --output_dir packages/finetuning/outputs/astro-lora-v1 \
  --max_seq_length 2048 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --max_new_tokens 400 \
  --temperature 0.2 \
  --save_merged_16bit
```

## What It Runs

The script executes these steps in order:

1. `validate_dataset.py`
2. `train_unsloth.py`
3. `test_adapter.py`
4. `evaluate_adapter.py`
5. `compare_models.py`
6. `package_outputs.py` if you want a validated archive for download

## Important Output Paths

- LoRA adapter:
  `packages/finetuning/outputs/astro-lora-v1`
- Held-out eval predictions:
  `packages/finetuning/outputs/eval_runs/`
- Base-vs-adapter comparison report:
  `packages/finetuning/outputs/comparison_runs/`

## Package And Download Without Google Drive

If `drive.mount('/content/drive')` fails in Colab, use the built-in packaging script instead:

```bash
python packages/finetuning/scripts/package_outputs.py --download
```

This does two useful things:

- refuses to create an empty zip if only `.gitkeep` files exist
- downloads `packages/finetuning/finetuning_outputs.zip` directly from Colab

If you only want to create the archive:

```bash
python packages/finetuning/scripts/package_outputs.py
```

## If You Hit GPU Memory Errors

Retry with:

```bash
python packages/finetuning/scripts/run_full_cycle.py \
  --train_file packages/finetuning/data/train.jsonl \
  --eval_file packages/finetuning/data/eval.jsonl \
  --output_dir packages/finetuning/outputs/astro-lora-v1 \
  --max_seq_length 1536 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_new_tokens 300 \
  --temperature 0.2 \
  --save_merged_16bit
```

## Important Reminder

This model is for **chart explanation only**.

Do not use it to calculate:

- kundli positions
- dashas
- house placements
- planetary math

Those should still come from the backend or chart engine.
