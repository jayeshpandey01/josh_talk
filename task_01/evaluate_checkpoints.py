"""
evaluate_checkpoints.py
-----------------------
Evaluates all checkpoints in the whisper-small-hi folder for WER.

Training context (from hindi_asr_finetuning.ipynb):
  - Base model  : openai/whisper-small
  - PEFT method : LoRA (r=32, alpha=64, target=q_proj+v_proj)
  - Train data  : 93 samples (JoshTalks conversational Hindi)
  - Val data    : 11 samples
  - Max steps   : 1000  (save every 100)
  - Eval set    : google/fleurs hi_in test[:100]

WHY WER IS HIGH (89-147%):
  1. Only 93 training samples -- severely under-trained.
  2. Domain mismatch: trained on JoshTalks coaching audio,
     evaluated on FLEURS read-speech data.
  3. WER > 100% happens when the model hallucinates extra words.

Usage:
  python evaluate_checkpoints.py
  python evaluate_checkpoints.py --checkpoint whisper-small-hi/checkpoint-700
  python evaluate_checkpoints.py --num_samples 50
"""

import os
import re
import argparse
import torch
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel
import evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_ID   = "openai/whisper-small"
CHECKPOINT_DIR  = "whisper-small-hi"
FLEURS_DATASET  = "google/fleurs"
FLEURS_LANG     = "hi_in"
DEFAULT_SAMPLES = 100   # number of FLEURS test samples to evaluate on

# ---------------------------------------------------------------------------
# Text normalisation
# (mirrors the clean_hindi_text used during training)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Remove punctuation and extra whitespace, lowercase Latin characters."""
    text = text.strip()
    # Remove all characters that are not Devanagari, digits, or whitespace
    text = re.sub(r"[^\u0900-\u097F\s0-9]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, processor, device: str):
    base = AutoModelForSpeechSeq2Seq.from_pretrained(BASE_MODEL_ID).to(device)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()

    # Force Hindi transcription (same as training setup)
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    return model


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

def compute_wer_for_model(model, processor, dataset, device: str) -> tuple:
    """Returns (wer_score, predictions_list, references_list)."""
    wer_metric = evaluate.load("wer")
    predictions = []
    references  = []

    for i, sample in enumerate(dataset):
        audio_array = sample["audio"]["array"]

        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                input_features,
                language="hindi",
                task="transcribe",
            )

        pred_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        ref_text  = sample["transcription"]

        pred_norm = normalize_text(pred_text)
        ref_norm  = normalize_text(ref_text)

        predictions.append(pred_norm)
        references.append(ref_norm)

        print(f"  [{i+1}/{len(dataset)}]  REF : {ref_norm}")
        print(f"           PRED: {pred_norm}")

    wer_score = 100.0 * wer_metric.compute(
        predictions=predictions,
        references=references,
    )
    return wer_score, predictions, references


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def get_checkpoints(base_dir: str):
    checkpoints = sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if d.startswith("checkpoint-")
        ],
        key=lambda p: int(p.split("-")[-1]),   # sort numerically
    )
    return checkpoints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WER evaluation for whisper-small-hi checkpoints")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a single checkpoint (default: evaluate all checkpoints)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of FLEURS test samples to evaluate on (default: {DEFAULT_SAMPLES})",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load processor once ----
    print(f"Loading processor from {BASE_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, language="Hindi", task="transcribe")

    # ---- Load FLEURS test set ----
    print(f"Loading FLEURS ({FLEURS_LANG}) test[:{args.num_samples}] ...")
    dataset = load_dataset(
        FLEURS_DATASET,
        FLEURS_LANG,
        split=f"test[:{args.num_samples}]",
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"Loaded {len(dataset)} samples.")

    # ---- Determine which checkpoints to evaluate ----
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = get_checkpoints(CHECKPOINT_DIR)

    if not checkpoints:
        print(f"No checkpoints found in '{CHECKPOINT_DIR}'. Exiting.")
        return

    print(f"\nCheckpoints to evaluate ({len(checkpoints)}):")
    for ckpt in checkpoints:
        print(f"  {ckpt}")

    # ---- Known training-time validation WER (from notebook output) ----
    training_wer = {
        "checkpoint-100":  93.63,
        "checkpoint-200": 101.87,
        "checkpoint-300": 102.25,
        "checkpoint-400": 102.99,
        "checkpoint-500": 102.99,
        "checkpoint-600":  95.13,
        "checkpoint-700":  89.51,
        "checkpoint-800":  90.64,
        "checkpoint-900":  99.44,
        "checkpoint-1000": 91.39,
    }

    # ---- Evaluate ----
    results = {}

    for ckpt in checkpoints:
        ckpt_name = os.path.basename(ckpt)
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt}")

        train_wer = training_wer.get(ckpt_name)
        if train_wer is not None:
            print(f"  Training-time val WER (11 samples): {train_wer:.2f}%")

        print(f"{'='*60}")

        model = load_model(ckpt, processor, device)
        wer, preds, refs = compute_wer_for_model(model, processor, dataset, device)

        results[ckpt_name] = wer
        print(f"\n  FLEURS WER ({args.num_samples} samples): {wer:.2f}%")

        # Free memory between checkpoints
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<25} {'FLEURS WER':>12} {'Train Val WER':>14}")
    print("-" * 55)
    for ckpt_name, wer in sorted(results.items(), key=lambda x: int(x[0].split("-")[-1])):
        train_wer = training_wer.get(ckpt_name)
        train_str = f"{train_wer:.2f}%" if train_wer is not None else "N/A"
        print(f"{ckpt_name:<25} {wer:>11.2f}% {train_str:>14}")

    best = min(results, key=lambda k: results[k])
    print(f"\nBest checkpoint : {best}")
    print(f"Best FLEURS WER : {results[best]:.2f}%")

    # ---- Diagnosis ----
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    print(
        "Training used only 93 samples (JoshTalks conversational Hindi).\n"
        "FLEURS is read-speech (different domain), so WER will be high.\n"
        "\n"
        "Why your friend got ~18% WER:\n"
        "  - Likely trained on a much larger dataset (thousands of samples).\n"
        "  - Or evaluated on in-domain data (same distribution as training).\n"
        "  - May have used a larger model (whisper-medium or whisper-large).\n"
        "\n"
        "To improve your WER:\n"
        "  1. Use more training data (minimum ~500-1000 samples).\n"
        "  2. Evaluate on your own domain (JoshTalks audio) for a fair comparison.\n"
        "  3. Increase max_steps (e.g., 3000-5000) with early stopping on WER.\n"
        "  4. Use a lower learning rate (3e-5) with cosine schedule.\n"
    )


if __name__ == "__main__":
    main()
