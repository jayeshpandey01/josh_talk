"""
Complete Training and Evaluation Pipeline for Hindi ASR
This script handles:
1. Data preparation from downloaded files
2. Whisper-small fine-tuning
3. Evaluation on FLEURS Hindi test set
"""

import pandas as pd
import json
import os
from datasets import Dataset, Audio, DatasetDict, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np

print("="*80)
print("HINDI ASR TRAINING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: PREPARE DATASET FROM DOWNLOADED FILES
# ============================================================================
print("\n[STEP 1] Preparing dataset from downloaded files...")

# Load CSV
df = pd.read_csv('../dataset/FT Data - data.csv')
print(f"Total samples in CSV: {len(df)}")

# Process transcriptions and create dataset
data_list = []

for idx, row in df.iterrows():
    recording_id = row['recording_id']
    audio_path = f"audio/{recording_id}.wav"
    trans_path = f"transcriptions/{recording_id}.json"
    
    # Check if files exist
    if not os.path.exists(audio_path) or not os.path.exists(trans_path):
        continue
    
    # Load transcription
    try:
        with open(trans_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)
        
        # Combine all text segments
        text = ' '.join([seg['text'] for seg in trans_data if 'text' in seg])
        
        if text.strip():
            data_list.append({
                'audio': audio_path,
                'text': text.strip(),
                'user_id': row['user_id']
            })
    except Exception as e:
        print(f"Error processing {recording_id}: {e}")
        continue

print(f"Valid samples: {len(data_list)}")

# Create DataFrame
df_processed = pd.DataFrame(data_list)

# Train/val split (90/10) stratified by user
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df_processed,
    test_size=0.1,
    random_state=42,
    stratify=df_processed['user_id']
)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Create HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['audio', 'text']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['audio', 'text']].reset_index(drop=True))

# Cast audio column
train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column('audio', Audio(sampling_rate=16000))

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

print("Dataset prepared successfully!")

# ============================================================================
# STEP 2: LOAD WHISPER MODEL AND PROCESSOR
# ============================================================================
print("\n[STEP 2] Loading Whisper-small model...")

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language="Hindi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Configure for Hindi
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
model.config.suppress_tokens = []

print(f"Model loaded: {model_name}")
print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

# ============================================================================
# STEP 3: PREPARE DATASET FOR TRAINING
# ============================================================================
print("\n[STEP 3] Preparing features...")

def prepare_dataset(batch):
    """Preprocess audio and text"""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Apply preprocessing
dataset_dict = dataset_dict.map(
    prepare_dataset,
    remove_columns=dataset_dict.column_names["train"],
    num_proc=1
)

print("Features prepared!")

# ============================================================================
# STEP 4: DATA COLLATOR
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ============================================================================
# STEP 5: METRICS
# ============================================================================

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ============================================================================
# STEP 6: TRAINING
# ============================================================================
print("\n[STEP 6] Starting training...")

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Training started...")
trainer.train()

# Save model
trainer.save_model("./whisper-small-hi-finetuned/final")
processor.save_pretrained("./whisper-small-hi-finetuned/final")
print("Model saved!")

# ============================================================================
# STEP 7: EVALUATION ON FLEURS
# ============================================================================
print("\n[STEP 7] Evaluating on FLEURS Hindi test set...")

# Load FLEURS
fleurs_test = load_dataset("google/fleurs", "hi_in", split="test")
print(f"FLEURS test samples: {len(fleurs_test)}")

def prepare_fleurs(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["reference"] = batch["transcription"]
    return batch

fleurs_test = fleurs_test.map(prepare_fleurs)

def evaluate_model(model, processor, test_dataset, model_name):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    predictions = []
    references = []
    
    for sample in test_dataset:
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
        references.append(sample["reference"])
    
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer, predictions, references

# Evaluate pretrained
print("\nEvaluating pretrained Whisper-small...")
pretrained_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
pretrained_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")

pretrained_wer, _, _ = evaluate_model(pretrained_model, processor, fleurs_test, "Pretrained")
print(f"Pretrained WER: {pretrained_wer:.4f}")

# Evaluate fine-tuned
print("\nEvaluating fine-tuned model...")
finetuned_model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-hi-finetuned/final")
finetuned_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")

finetuned_wer, predictions, references = evaluate_model(finetuned_model, processor, fleurs_test, "Fine-tuned")
print(f"Fine-tuned WER: {finetuned_wer:.4f}")

# ============================================================================
# STEP 8: RESULTS
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

results_df = pd.DataFrame({
    'Model': ['Whisper Small (Pretrained)', 'FT Whisper Small (yours)'],
    'Hindi WER': [pretrained_wer, finetuned_wer]
})

print(results_df.to_string(index=False))
print("="*80)

improvement = (pretrained_wer - finetuned_wer) / pretrained_wer * 100
print(f"\nWER Improvement: {improvement:.2f}%")

# Save results
results_df.to_csv('evaluation_results.csv', index=False)
print("\nResults saved to evaluation_results.csv")

# Sample predictions
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)
for i in range(min(5, len(predictions))):
    print(f"\nSample {i+1}:")
    print(f"Reference:  {references[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-"*80)

print("\n✓ Training and evaluation complete!")
