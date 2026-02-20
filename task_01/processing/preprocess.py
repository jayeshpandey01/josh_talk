"""
Simple preprocessing script for Hindi ASR dataset
Run this to download and prepare the data
"""

import pandas as pd
import json
import requests
import os
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("Hindi ASR Dataset Preprocessing")
print("="*80)

# Step 1: Load CSV
print("\n1. Loading dataset CSV...")
df = pd.read_csv('../dataset/FT Data - data.csv')
print(f"   Total samples: {len(df)}")
print(f"   Total duration: {df['duration'].sum() / 3600:.2f} hours")

# Step 2: Fix URLs
print("\n2. Fixing URLs...")
def fix_url(url):
    return url.replace('joshtalks-data-collection', 'upload_goai')

df['rec_url_gcp'] = df['rec_url_gcp'].apply(fix_url)
df['transcription_url_gcp'] = df['transcription_url_gcp'].apply(fix_url)
df['metadata_url_gcp'] = df['metadata_url_gcp'].apply(fix_url)
print("   URLs fixed successfully")

# Step 3: Create directories
print("\n3. Creating directories...")
os.makedirs('audio', exist_ok=True)
os.makedirs('transcriptions', exist_ok=True)
print("   Directories created")

# Step 4: Download function
def download_file(url, save_path):
    """Download file from URL"""
    try:
        if os.path.exists(save_path):
            return True
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"   Error downloading {url}: {e}")
        return False

# Step 5: Process transcription
def process_transcription(transcription_data):
    """Combine all text segments"""
    if isinstance(transcription_data, list):
        return ' '.join([segment.get('text', '') for segment in transcription_data])
    return ""

# Step 6: Download and process
print("\n4. Downloading and processing files...")
processed_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    recording_id = row['recording_id']
    audio_path = f"audio/{recording_id}.wav"
    trans_path = f"transcriptions/{recording_id}.json"
    
    # Download audio
    if not download_file(row['rec_url_gcp'], audio_path):
        continue
    
    # Download transcription
    if not download_file(row['transcription_url_gcp'], trans_path):
        continue
    
    # Load and process transcription
    try:
        with open(trans_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)
        text = process_transcription(trans_data)
        
        if not text.strip():
            continue
        
        processed_data.append({
            'audio': audio_path,
            'text': text.strip(),
            'recording_id': recording_id,
            'user_id': row['user_id'],
            'duration': row['duration']
        })
        
    except Exception as e:
        print(f"   Error processing {recording_id}: {e}")
        continue

# Step 7: Save processed data
print(f"\n5. Saving processed data...")
processed_df = pd.DataFrame(processed_data)
processed_df.to_csv('processed_dataset.csv', index=False)

print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"Total processed samples: {len(processed_df)}")
print(f"Total duration: {processed_df['duration'].sum() / 3600:.2f} hours")
print(f"\nFiles saved:")
print(f"  - processed_dataset.csv")
print(f"  - audio/ directory with {len(processed_df)} WAV files")
print(f"  - transcriptions/ directory with {len(processed_df)} JSON files")
print("="*80)
