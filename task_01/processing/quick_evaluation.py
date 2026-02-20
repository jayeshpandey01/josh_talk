"""
Quick Evaluation Script for Whisper Models on FLEURS Hindi Test Set

This script provides a streamlined way to evaluate both pretrained and fine-tuned
Whisper-small models on the FLEURS Hindi test dataset.

Usage:
    python quick_evaluation.py --model_path ./whisper-small-hi-finetuned/final
"""

import argparse
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import pandas as pd


def load_model_and_processor(model_path, language="hi"):
    """Load Whisper model and processor"""
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    return model, processor


def prepare_dataset(dataset, processor):
    """Prepare FLEURS dataset for evaluation"""
    def prepare_sample(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["reference"] = batch["transcription"]
        return batch
    
    return dataset.map(prepare_sample)


def evaluate_model(model, processor, test_dataset, device="cuda"):
    """Evaluate model and compute WER"""
    model.eval()
    model.to(device)
    
    predictions = []
    references = []
    
    wer_metric = evaluate.load("wer")
    
    for sample in tqdm(test_dataset, desc="Evaluating"):
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
        references.append(sample["reference"])
    
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer, predictions, references


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper models on FLEURS Hindi")
    parser.add_argument("--model_path", type=str, default="openai/whisper-small",
                        help="Path to fine-tuned model or HuggingFace model name")
    parser.add_argument("--baseline", type=str, default="openai/whisper-small",
                        help="Baseline model for comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Whisper Model Evaluation on FLEURS Hindi Test Set")
    print("="*80)
    
    # Load FLEURS Hindi test dataset
    print("\n1. Loading FLEURS Hindi test dataset...")
    fleurs_test = load_dataset("google/fleurs", "hi_in", split="test")
    
    if args.num_samples:
        fleurs_test = fleurs_test.select(range(args.num_samples))
    
    print(f"   Test samples: {len(fleurs_test)}")
    
    # Evaluate baseline model
    print(f"\n2. Evaluating baseline model: {args.baseline}")
    baseline_model, baseline_processor = load_model_and_processor(args.baseline)
    fleurs_prepared = prepare_dataset(fleurs_test, baseline_processor)
    
    baseline_wer, _, _ = evaluate_model(
        baseline_model, 
        baseline_processor, 
        fleurs_prepared, 
        args.device
    )
    print(f"   Baseline WER: {baseline_wer:.4f} ({baseline_wer*100:.2f}%)")
    
    # Evaluate fine-tuned model
    print(f"\n3. Evaluating fine-tuned model: {args.model_path}")
    finetuned_model, finetuned_processor = load_model_and_processor(args.model_path)
    fleurs_prepared = prepare_dataset(fleurs_test, finetuned_processor)
    
    finetuned_wer, predictions, references = evaluate_model(
        finetuned_model, 
        finetuned_processor, 
        fleurs_prepared, 
        args.device
    )
    print(f"   Fine-tuned WER: {finetuned_wer:.4f} ({finetuned_wer*100:.2f}%)")
    
    # Results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame({
        'Model': ['Whisper Small (Pretrained)', 'FT Whisper Small (yours)'],
        'Hindi WER': [baseline_wer, finetuned_wer]
    })
    
    print(results_df.to_string(index=False))
    print("="*80)
    
    improvement = (baseline_wer - finetuned_wer) / baseline_wer * 100
    print(f"\nWER Improvement: {improvement:.2f}%")
    
    # Save results
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to: evaluation_results.csv")
    
    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Reference:  {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-"*80)


if __name__ == "__main__":
    main()
