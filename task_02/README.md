# Hindi Speech Disfluency Detection Pipeline

**Josh Talks Internship Project - February 2026**

Automated detection and extraction of speech disfluencies from Hindi conversational audio using hybrid ASR + rule-based approach.

---

## Quick Start

### Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install FFmpeg (required for audio processing)
# Windows: Download from https://ffmpeg.org/download.html and add to PATH
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg

# 3. Verify installation
python -c "import whisper_timestamped; print('✓ Ready!')"
```

### Run Pipeline

```bash
# Process first 5 recordings (for testing)
python disfluency_detection_pipeline.py

# Results will be saved to:
# - output/disfluency_results.csv (detection results)
# - output/disfluency_clips/ (audio clips)
```

---

## What This Pipeline Does

1. **Downloads** Hindi audio from GCP URLs
2. **Preprocesses** audio (16kHz mono, normalized)
3. **Transcribes** with word-level timestamps (Whisper)
4. **Detects** 5 types of disfluencies:
   - Fillers (अम्म, उम्म, हम्म)
   - Repetitions (मैं-मैं, वो-वो)
   - False starts (जा—, कर—)
   - Prolongations (सोoooo)
   - Hesitations ([PAUSE])
5. **Extracts** audio clips for each disfluency
6. **Exports** structured CSV with timestamps and clip paths

---

## Output Format

### CSV Columns

| Column | Description | Example |
|--------|-------------|---------|
| `recording_id` | Recording identifier | 825780 |
| `type` | Disfluency category | filler |
| `subtype` | Specific variant | umm |
| `start` | Start time (seconds) | 12.34 |
| `end` | End time (seconds) | 12.89 |
| `text` | Transcribed text | उम्म |
| `confidence` | ASR confidence | 0.87 |
| `clip_filename` | Audio clip name | 825780_disf_001_filler_umm.wav |

### Sample Output

```csv
recording_id,type,subtype,start,end,text,clip_filename
825780,filler,umm,12.34,12.89,उम्म,825780_disf_001_filler_umm.wav
825780,repetition,immediate,45.67,46.23,मैं मैं,825780_disf_002_repetition_immediate.wav
```

---

## Project Structure

```
.
├── disfluency_detection_pipeline.py  # Main pipeline code
├── requirements.txt                   # Python dependencies
├── METHODOLOGY_REPORT.md              # Detailed methodology
├── README.md                          # This file
├── dataset/
│   ├── FT_Data_-_data.csv            # Recording metadata
│   └── Speech Disfluencies List - Sheet1.csv  # Disfluency patterns
└── output/
    ├── disfluency_results.csv        # Detection results
    ├── audio_files/                  # Downloaded audio cache
    └── disfluency_clips/             # Extracted clips
```

---

## Usage Examples

### Process Entire Dataset

```python
from disfluency_detection_pipeline import DisfluencyPipeline

pipeline = DisfluencyPipeline()

# Process all recordings
results = pipeline.process_dataset(max_recordings=None)

print(f"Detected {len(results)} disfluencies")
```

### Process Single Recording

```python
# Process one recording
disfluencies, audio_path = pipeline.process_recording(
    recording_id="825780",
    audio_url="https://storage.googleapis.com/upload_goai/967179/825780_audio.wav"
)

for disf in disfluencies:
    print(f"{disf['type']:15s} {disf['start']:.2f}s - {disf['end']:.2f}s: {disf['text']}")
```

### Custom Detection

```python
from disfluency_detection_pipeline import HindiDisfluencyDetector

detector = HindiDisfluencyDetector()
detector.load_whisper_model(model_size="small")  # Faster model

disfluencies, transcription = detector.detect_all_disfluencies("audio.wav")
```

---

## Configuration

### Whisper Model Size

Trade-off between speed and accuracy:

```python
# Fast but less accurate
detector.load_whisper_model(model_size="small")

# Balanced (default)
detector.load_whisper_model(model_size="medium")

# Most accurate but slow
detector.load_whisper_model(model_size="large")
```

### Hesitation Threshold

Adjust silence detection sensitivity:

```python
# In detect_hesitations() method
silences = detect_silence(
    audio,
    min_silence_len=300,  # Change to 200 for shorter pauses
    silence_thresh=audio.dBFS - 16  # Adjust threshold
)
```

### Clip Padding

Add more context around clips:

```python
clipper.extract_clip(
    audio_path,
    start_sec,
    end_sec,
    output_filename,
    padding_ms=500  # Increase from default 200ms
)
```

---

## Troubleshooting

### FFmpeg Not Found

```bash
# Error: "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"

# Solution: Install FFmpeg and add to PATH
# Windows: https://www.wikihow.com/Install-FFmpeg-on-Windows
# Verify: ffmpeg -version
```

### CUDA Out of Memory

```python
# Error: "RuntimeError: CUDA out of memory"

# Solution 1: Use smaller model
detector.load_whisper_model(model_size="small")

# Solution 2: Process on CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### URL Download Fails

```python
# Error: "Failed to download audio"

# The pipeline automatically fixes URLs:
# OLD: https://storage.googleapis.com/joshtalks-data-collection/hq_data/...
# NEW: https://storage.googleapis.com/upload_goai/...

# If still failing, check network connection or URL validity
```

### Low Detection Rate

```python
# If detecting too few disfluencies:

# 1. Check Whisper initial_prompt includes your target fillers
initial_prompt = "उम्म... मतलब... [add more]..."

# 2. Lower confidence threshold
disfluencies = [d for d in disfluencies if d['confidence'] > 0.5]  # Lower from 0.7

# 3. Add more patterns to disfluency list CSV
```

---

## Performance

| Metric | Value |
|--------|-------|
| Processing speed | ~2-3x real-time (medium model) |
| GPU acceleration | 5-10x faster with CUDA |
| Memory usage | ~4GB (medium model) |
| Disk space | ~100MB per hour of audio |

---

## Uploading Clips to Google Drive

### Option 1: Manual Upload

```bash
# Compress clips
zip -r disfluency_clips.zip output/disfluency_clips/

# Upload to Google Drive via web interface
# Share folder and get link
```

### Option 2: Google Drive API (Automated)

See `METHODOLOGY_REPORT.md` Section 6.2 for code example.

---

## Citation

If using this pipeline in research or publications:

```bibtex
@software{hindi_disfluency_pipeline_2026,
  title={Hindi Speech Disfluency Detection Pipeline},
  author={Josh Talks Internship Project},
  year={2026},
  url={https://github.com/your-repo}
}
```

---

## License

This project is for educational and research purposes as part of Josh Talks internship program.

---

## Acknowledgments

- **OpenAI Whisper** for robust multilingual ASR
- **whisper-timestamped** for word-level timestamps
- **Josh Talks** for providing the dataset and opportunity

---

## Support

For detailed methodology, see `METHODOLOGY_REPORT.md`

For code documentation, see inline comments in `disfluency_detection_pipeline.py`

**System Requirements:**
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- FFmpeg installed
- Optional: CUDA-capable GPU for faster processing
