"""
Dataset Validation Script

This script validates the downloaded and processed dataset before training.
It checks for:
- File existence and accessibility
- Audio quality and format
- Transcription completeness
- Data distribution

Usage:
    python validate_dataset.py --csv_path ../dataset/FT\ Data\ -\ data.csv
"""

import argparse
import pandas as pd
import json
import os
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def fix_url(url):
    """Fix URL by replacing old domain"""
    return url.replace('joshtalks-data-collection', 'upload_goai')


def validate_audio_file(audio_path):
    """Validate audio file and return statistics"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        stats = {
            'exists': True,
            'duration': duration,
            'sample_rate': sr,
            'channels': 1 if audio.ndim == 1 else audio.shape[0],
            'max_amplitude': np.max(np.abs(audio)),
            'is_silent': np.max(np.abs(audio)) < 0.001,
            'valid': True
        }
        
        return stats
    except Exception as e:
        return {
            'exists': os.path.exists(audio_path),
            'error': str(e),
            'valid': False
        }


def validate_transcription_file(trans_path):
    """Validate transcription file and return statistics"""
    try:
        with open(trans_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [seg.get('text', '') for seg in data]
            full_text = ' '.join(texts)
            
            stats = {
                'exists': True,
                'num_segments': len(data),
                'total_chars': len(full_text),
                'total_words': len(full_text.split()),
                'is_empty': len(full_text.strip()) == 0,
                'valid': True
            }
            
            return stats
        else:
            return {'exists': True, 'valid': False, 'error': 'Invalid format'}
            
    except Exception as e:
        return {
            'exists': os.path.exists(trans_path),
            'error': str(e),
            'valid': False
        }


def main():
    parser = argparse.ArgumentParser(description="Validate Hindi ASR dataset")
    parser.add_argument("--csv_path", type=str, default="../dataset/FT Data - data.csv",
                        help="Path to dataset CSV file")
    parser.add_argument("--audio_dir", type=str, default="../processing/audio",
                        help="Directory containing audio files")
    parser.add_argument("--trans_dir", type=str, default="../processing/transcriptions",
                        help="Directory containing transcription files")
    parser.add_argument("--download", action="store_true",
                        help="Download missing files")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Dataset Validation Report")
    print("="*80)
    
    # Load CSV
    print(f"\n1. Loading dataset from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"   Total samples in CSV: {len(df)}")
    print(f"   Total duration: {df['duration'].sum() / 3600:.2f} hours")
    
    # Fix URLs
    df['rec_url_gcp'] = df['rec_url_gcp'].apply(fix_url)
    df['transcription_url_gcp'] = df['transcription_url_gcp'].apply(fix_url)
    
    # Create directories
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.trans_dir, exist_ok=True)
    
    # Validate files
    print("\n2. Validating files...")
    validation_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        recording_id = row['recording_id']
        audio_path = os.path.join(args.audio_dir, f"{recording_id}.wav")
        trans_path = os.path.join(args.trans_dir, f"{recording_id}.json")
        
        result = {
            'recording_id': recording_id,
            'user_id': row['user_id'],
            'expected_duration': row['duration'],
            'audio_exists': os.path.exists(audio_path),
            'trans_exists': os.path.exists(trans_path),
        }
        
        # Validate audio
        if result['audio_exists']:
            audio_stats = validate_audio_file(audio_path)
            result.update({f'audio_{k}': v for k, v in audio_stats.items()})
        
        # Validate transcription
        if result['trans_exists']:
            trans_stats = validate_transcription_file(trans_path)
            result.update({f'trans_{k}': v for k, v in trans_stats.items()})
        
        validation_results.append(result)
    
    # Create validation DataFrame
    val_df = pd.DataFrame(validation_results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\nFile Availability:")
    print(f"  Audio files found: {val_df['audio_exists'].sum()}/{len(val_df)} ({val_df['audio_exists'].mean()*100:.1f}%)")
    print(f"  Transcription files found: {val_df['trans_exists'].sum()}/{len(val_df)} ({val_df['trans_exists'].mean()*100:.1f}%)")
    
    # Audio validation
    if 'audio_valid' in val_df.columns:
        valid_audio = val_df['audio_valid'].sum()
        print(f"\nAudio Quality:")
        print(f"  Valid audio files: {valid_audio}/{val_df['audio_exists'].sum()}")
        
        if 'audio_is_silent' in val_df.columns:
            silent = val_df['audio_is_silent'].sum()
            print(f"  Silent audio files: {silent}")
        
        if 'audio_duration' in val_df.columns:
            durations = val_df[val_df['audio_valid'] == True]['audio_duration']
            print(f"  Duration range: {durations.min():.1f}s - {durations.max():.1f}s")
            print(f"  Average duration: {durations.mean():.1f}s")
    
    # Transcription validation
    if 'trans_valid' in val_df.columns:
        valid_trans = val_df['trans_valid'].sum()
        print(f"\nTranscription Quality:")
        print(f"  Valid transcriptions: {valid_trans}/{val_df['trans_exists'].sum()}")
        
        if 'trans_is_empty' in val_df.columns:
            empty = val_df['trans_is_empty'].sum()
            print(f"  Empty transcriptions: {empty}")
        
        if 'trans_total_words' in val_df.columns:
            words = val_df[val_df['trans_valid'] == True]['trans_total_words']
            print(f"  Words per transcription: {words.min():.0f} - {words.max():.0f}")
            print(f"  Average words: {words.mean():.1f}")
    
    # Both valid
    both_valid = val_df['audio_exists'] & val_df['trans_exists']
    if 'audio_valid' in val_df.columns and 'trans_valid' in val_df.columns:
        both_valid = both_valid & val_df['audio_valid'] & val_df['trans_valid']
        both_valid = both_valid & ~val_df.get('audio_is_silent', False) & ~val_df.get('trans_is_empty', False)
    
    print(f"\nUsable Samples:")
    print(f"  Total usable: {both_valid.sum()}/{len(val_df)} ({both_valid.mean()*100:.1f}%)")
    
    # Save validation report
    val_df.to_csv("validation_report.csv", index=False)
    print(f"\nDetailed validation report saved to: validation_report.csv")
    
    # Plot duration distribution
    if 'audio_duration' in val_df.columns:
        plt.figure(figsize=(10, 6))
        durations = val_df[val_df['audio_valid'] == True]['audio_duration']
        plt.hist(durations, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.title('Audio Duration Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig('duration_distribution.png', dpi=150, bbox_inches='tight')
        print(f"Duration distribution plot saved to: duration_distribution.png")
    
    # Issues summary
    print("\n" + "="*80)
    print("ISSUES FOUND")
    print("="*80)
    
    issues = []
    
    if val_df['audio_exists'].sum() < len(val_df):
        missing = len(val_df) - val_df['audio_exists'].sum()
        issues.append(f"Missing audio files: {missing}")
    
    if val_df['trans_exists'].sum() < len(val_df):
        missing = len(val_df) - val_df['trans_exists'].sum()
        issues.append(f"Missing transcription files: {missing}")
    
    if 'audio_is_silent' in val_df.columns and val_df['audio_is_silent'].sum() > 0:
        issues.append(f"Silent audio files: {val_df['audio_is_silent'].sum()}")
    
    if 'trans_is_empty' in val_df.columns and val_df['trans_is_empty'].sum() > 0:
        issues.append(f"Empty transcriptions: {val_df['trans_is_empty'].sum()}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ No issues found!")
    
    print("\n" + "="*80)
    print(f"Validation complete. Ready for training: {both_valid.sum()} samples")
    print("="*80)


if __name__ == "__main__":
    main()
