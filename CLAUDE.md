# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an AI-driven football highlight generator that processes full-length football matches to create dynamic highlight reels. The system supports multiple detection methods: scoreboard-based OCR analysis, audio excitement detection with Gemini Vision validation, and hybrid approaches combining both methods.

## Core Architecture
The project supports three highlight detection modes:

### Scoreboard-Based Pipeline (Original)
1. **Data Extraction** (`get_dataframe.py`): Uses PaddleOCR to extract team names and scores from video frames by cropping the scoreboard area (coordinates 40:65, 110:313). Processes every 200th frame and saves results to CSV.

2. **Video Processing** (`gen_reels.py`): Uses YOLOv8 object detection to track and crop scoreboards dynamically. Implements box tracking algorithms to maintain consistent cropping even when scoreboards move.

3. **Highlight Generation**: Analyzes score changes from the CSV data to identify goal moments, creates clips around each goal event.

### Audio + Gemini Vision Pipeline (New)
1. **Audio Analysis** (`audio_analysis.py`): 
   - Extracts audio using PyAV for fast demuxing
   - Computes RMS envelope with librosa 
   - Applies bandpass filtering for crowd cheering (1000-3000 Hz)
   - Detects amplitude peaks with configurable prominence and minimum distance

2. **Gemini Validation** (`gemini_filter.py`):
   - Extracts video frames at detected audio peaks
   - Uses Gemini Vision API to validate if frames show football excitement/celebration
   - Implements async processing with retry logic and SQLite caching
   - Returns only peaks that pass visual validation

3. **Clip Generation**: Creates highlight clips around validated excitement moments

### Hybrid Mode
Combines both scoreboard and audio detection methods, merges overlapping clips, and ranks by confidence scores.

## Key Dependencies

### Core Libraries
- **OpenCV** (`cv2`): Video processing and frame manipulation
- **MoviePy**: Video editing and concatenation
- **pandas**: Data processing and CSV handling
- **loguru**: Structured logging with JSON summaries

### Scoreboard Analysis
- **PaddleOCR**: Text extraction from scoreboard regions
- **YOLOv8** (`ultralytics`): Object detection for scoreboard tracking

### Audio + AI Analysis
- **PyAV**: Fast audio demuxing from video files
- **librosa**: Audio signal processing and RMS envelope computation
- **scipy**: Signal processing for peak detection and bandpass filtering
- **google-generativeai**: Gemini Vision API for frame validation
- **matplotlib**: Audio analysis visualization (debugging tools)

## Configuration Parameters (`config.py`)

### Audio Analysis
- `AUDIO_SR = 22050`: Target sample rate for audio resampling
- `PEAK_PROMINENCE = 0.25`: Fraction of max RMS that counts as a peak
- `PEAK_MIN_DIST_SEC = 5`: Minimum distance between peaks (seconds)
- `BANDPASS = (1000, 3000)`: Cheering frequency band (Hz)

### Gemini API
- `GEMINI_MODEL = "gemini-1.5-flash"`: Vision model for frame analysis
- `GEMINI_POS_THRESHOLD = 0.6`: Confidence threshold for positive detection
- `GEMINI_MAX_CONCURRENT = 8`: Maximum concurrent API calls
- `GEMINI_CACHE_DB = ".cache/gemini.db"`: SQLite cache for API responses

### Video Clipping
- `PRE_SEC = 12`: Seconds before peak to include in clip
- `POST_SEC = 12`: Seconds after peak to include in clip
- `CLIP_OVERLAP_THRESHOLD = 10`: Merge clips if closer than this (seconds)

### Scoreboard Analysis (Legacy)
- `FRAME_SKIP = 200`: Process every Nth frame
- `OCR_CONFIDENCE = 0.90`: Minimum OCR confidence
- `SCOREBOARD_CROP = (40, 65, 110, 313)`: Y1, Y2, X1, X2 coordinates

## Running the Pipeline

### New CLI Interface
```bash
# Hybrid mode (recommended)
python gen_highlights.py --video match.mp4 --mode hybrid --output highlights.mp4

# Audio-only mode (works on any football video)
python gen_highlights.py --video fan_cam.mov --mode audio

# Scoreboard-only mode (requires visible scoreboard)
python gen_highlights.py --video broadcast.mp4 --mode scoreboard

# Debug mode with detailed logging
python gen_highlights.py --video match.mp4 --mode hybrid --debug
```

### Environment Setup
1. Set `GEMINI_API_KEY` environment variable for audio/hybrid modes
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure ffmpeg is available for audio processing

### Debugging Tools
- `tools/extract_audio.py video.mp4`: Visualize audio peaks and analysis
- JSON run summaries: `run-summary-YYYYMMDD-HHMM.json`
- SQLite cache: `.cache/gemini.db` (for API response caching)

## Testing
Run test suite: `pytest tests/ -v --asyncio-mode=auto`