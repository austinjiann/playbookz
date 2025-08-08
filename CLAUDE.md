# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a universal AI-driven sports highlight generator that works with **any football/sports video** - from phone recordings to broadcast footage. The system combines audio signal processing with Gemini Vision API validation and Expected Goals (xG) modeling to automatically identify and extract exciting moments without requiring visible scoreboards or specific video quality.

## Core Architecture

The system implements a four-stage pipeline that processes **any sports video format**:

### Stage 1: Audio Signal Analysis (`audio_analysis.py`)
- **Direct audio extraction**: Uses librosa to extract audio from any video container format
- **RMS envelope computation**: Identifies amplitude peaks in audio signal over time
- **Bandpass filtering**: Emphasizes crowd excitement frequencies (1000-3000 Hz) 
- **Peak detection**: Uses scipy signal processing with configurable sensitivity thresholds
- **Fallback sampling**: When no audio peaks detected, systematically samples frames every 7 seconds

### Stage 2: Gemini Vision Validation (`gemini_filter.py`)  
- **Frame extraction**: Captures video frames at detected audio peak timestamps
- **AI vision analysis**: Sends frames to Gemini Vision API with sports-specific prompts
- **Action recognition**: Identifies goals, shots, saves, celebrations, tackles, skills
- **Async processing**: Concurrent API calls with rate limiting and retry logic
- **SQLite caching**: Stores frame analysis results to minimize API costs and enable replay

### Stage 3: xG Enhancement (`gemini_filter.py` + `models/xg_model.py`)
- **Shot detection**: Identifies shot events during Gemini validation
- **Coordinate estimation**: Maps frame positions to StatsBomb pitch coordinates (120x80)
- **Context inference**: Extracts body part, set piece status, and play context
- **xG calculation**: Uses trained logistic regression model on StatsBomb data
- **Score blending**: Combines Gemini confidence with xG values using configurable weights

### Stage 4: Intelligent Clip Generation (`gen_highlights.py`)
- **HighlightClip class**: Represents clips with timestamp, confidence, source metadata, and xG data
- **Enhanced ranking**: Sorts clips by blended Gemini+xG scores when enabled
- **Dynamic merging**: Combines nearby exciting moments into extended highlight sequences
- **Overlap resolution**: Merges clips within configurable time windows (4-second default)

### Legacy: Scoreboard Analysis (Deprecated)
- **OCR-based detection**: Uses PaddleOCR + YOLOv8 for scoreboard text extraction
- **Score change detection**: Identifies goal events from scoreboard number changes  
- **Broadcast-only**: Requires visible scoreboards, limited to professional footage
- **Status**: Disabled by default (ENABLE_SCOREBOARD = False), preserved for future use

## Essential Commands

### Development Setup
```bash
# Environment setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configuration validation
python config.py

# Set Gemini API key (required)
export GEMINI_API_KEY="your-key-here"
# Or create .env file: echo "GEMINI_API_KEY=your-key" > .env
```

### Core Usage Patterns
```bash
# Primary mode - works on any sports video
python gen_highlights.py --video match.mp4 --mode audio --output highlights.mp4

# xG-enhanced highlighting with intelligent shot ranking
python gen_highlights.py --video match.mp4 --mode audio --use-xg --output highlights.mp4

# Custom xG weighting (higher = more influence of shot quality)
python gen_highlights.py --video match.mp4 --mode audio --use-xg --xg-weight 0.5

# Debug mode with detailed xG logging
python gen_highlights.py --video match.mp4 --mode audio --use-xg --debug

# Preview tool - analyze without rendering video
python tools/preview_clips.py --video match.mp4 --output debug_preview

# Direct audio analysis testing  
python audio_analysis.py match.mp4

# Direct Gemini frame testing
python gemini_filter.py match.mp4

# Train xG model (first-time setup)
python scripts/train_xg.py
```

### Testing Commands
```bash
# Run full test suite
pytest tests/ -v --asyncio-mode=auto

# Run specific test module
pytest tests/test_audio_pipeline.py -v

# Run single test method
pytest tests/test_audio_pipeline.py::TestAudioPeak::test_peak_creation -v
```

## Key Dependencies and Architecture

### Audio Processing Stack
- **librosa**: Core audio analysis, RMS envelope computation, audio loading
- **scipy**: Signal processing for peak detection and bandpass filtering
- **soundfile**: Audio file I/O, handles various audio formats  
- **numpy**: Numerical operations for signal processing arrays

### Video Processing Stack
- **OpenCV** (`cv2`): Video frame extraction, image encoding for API calls
- **MoviePy**: Video editing, clip concatenation, final output generation
- **av**: Alternative video processing (PyAV), used for audio extraction fallback

### AI Integration Stack  
- **google-generativeai**: Gemini Vision API client for frame analysis
- **scikit-learn**: Logistic regression for xG model training and prediction
- **statsbombpy**: Open football data API for xG model training
- **asyncio**: Concurrent processing for multiple API calls
- **sqlite3**: Local caching database for API responses

### Legacy OCR Stack (Deprecated)
- **PaddleOCR**: Text extraction from scoreboard regions
- **ultralytics**: YOLOv8 object detection for scoreboard tracking
- **pandas**: CSV data processing for OCR results

## Critical Configuration Understanding

### Audio Sensitivity Tuning
The system's effectiveness depends on properly tuned audio parameters in `config.py`:

- `PEAK_PROMINENCE = 0.10`: Lower values = higher sensitivity (detect more subtle moments)
- `PEAK_MIN_DIST_SEC = 2`: Minimum gap between detected peaks (prevents duplicate detections)  
- `BANDPASS = (1000, 3000)`: Crowd excitement frequency range (Hz)
- `FALLBACK_INTERVAL_SEC = 7`: When no audio peaks found, sample every N seconds

### Gemini API Integration
- `GEMINI_API_KEY`: **Required** environment variable for frame validation
- `GEMINI_POS_THRESHOLD = 0.4`: Lower threshold = more frames accepted as exciting
- `GEMINI_MAX_CONCURRENT = 8`: Concurrent API calls for performance vs rate limiting
- `GEMINI_CACHE_DB`: SQLite database prevents re-analyzing identical frames

### Video Processing Behavior  
- `PRE_SEC/POST_SEC = 12`: Clip padding around detected moments
- `DYNAMIC_CLIP_EXTENSION = True`: Automatically merge nearby exciting moments
- `MERGE_WINDOW_SEC = 4`: Maximum gap for merging separate clips into sequences

### xG Model Configuration
- **Training data**: StatsBomb open data (FIFA World Cup, La Liga, etc.)
- **Model type**: Logistic regression with shot features (distance, angle, body part, context)
- **Coordinate system**: StatsBomb 120x80 pitch, goal at (120, 40)
- **Runtime**: No dataset dependency - coefficients stored in `data/xg_coeffs.json`
- **Integration**: Optional via `--use-xg` flag, configurable weight with `--xg-weight`

## Development Workflow Patterns

### Debugging Failed Analysis
When highlights aren't generated properly, follow this diagnostic sequence:

1. **Check audio analysis**: `python audio_analysis.py video.mp4`
   - Verifies audio extraction and peak detection  
   - Should show detected peaks with timestamps and confidence scores

2. **Test Gemini connectivity**: `python gemini_filter.py video.mp4`  
   - Validates API key and frame analysis
   - Shows sample frame classification results

3. **Generate preview**: `python tools/preview_clips.py --video video.mp4`
   - Creates HTML preview showing detected vs validated frames
   - Helps visualize why certain moments were accepted/rejected

4. **Run with debug**: `python gen_highlights.py --video video.mp4 --debug`
   - Detailed logging of each processing stage
   - Shows clip merging and concatenation steps

5. **Test xG integration**: `python gen_highlights.py --video video.mp4 --use-xg --debug`
   - Shows xG calculations and coordinate estimation
   - Displays score blending: `Gemini=0.8 + xG=0.25 → Blended=0.635`
   - Verifies shot detection and context inference

### Performance Optimization Patterns
- **Cache utilization**: Gemini responses cached in `.cache/gemini.db` - reprocessing same video is much faster
- **API cost control**: `FALLBACK_MAX_SAMPLES = 20` limits fallback sampling to prevent excessive API calls
- **Concurrent processing**: `GEMINI_MAX_CONCURRENT = 8` balances speed vs rate limiting

### Testing Workflow
```bash
# Test core functionality
pytest tests/test_audio_pipeline.py::TestAudioPeak -v

# Test Gemini integration (requires API key)
pytest tests/test_audio_pipeline.py::TestGeminiFilter -v  

# Integration tests (comprehensive but slower)
pytest tests/test_audio_pipeline.py::TestIntegration -v

# Test xG model functionality
python -c "from models.xg_model import predict_xg; print(f'xG test: {predict_xg(x=108, y=40, body_part=\"right_foot\", is_set_piece=True):.3f}')"

# Verify xG training data exists
ls -la data/xg_coeffs.json

# Test xG integration components
python -c "from gemini_filter import _placeholder_pitch_coordinates; print('Coordinate mapping:', _placeholder_pitch_coordinates((720, 1280)))"
```

## Video Format Support

### Universal Athlete Support
This system is designed to work with **ANY sports video**, including:
- 📱 **Phone recordings** of training sessions or amateur matches
- 🎥 **Action camera footage** from players or coaches  
- 📺 **Broadcast/professional footage** with or without visible scoreboards
- 💻 **Screen recordings** of video analysis or game film
- 🏟️ **Stadium security camera** or fixed-angle footage

### Supported Formats
- **Video**: MP4, MOV, AVI, MKV, WEBM (any format supported by OpenCV/librosa)
- **Audio**: Automatic extraction from video container
- **Resolution**: Any resolution (frames are resized for AI analysis)
- **Duration**: Optimized for matches/sessions up to 2 hours

## How It Works for Different Video Types

### Phone Recordings & Amateur Video
- **Audio peaks**: Detects crowd reactions, player shouts, ball impacts
- **Fallback sampling**: Even silent videos get systematic frame analysis
- **Dynamic sensitivity**: Lower thresholds catch subtle excitement moments
- **Smart merging**: Combines nearby exciting moments into extended highlights

### Broadcast/Professional Footage  
- **Enhanced audio**: Professional audio provides clearer crowd noise signals
- **Scoreboard available**: Legacy scoreboard analysis available if needed (deprecated)
- **High-quality frames**: Better AI vision analysis from clearer imagery

### Training Videos & Drills
- **Coach instructions**: Audio peaks from coaching cues and feedback
- **Skill moments**: AI identifies successful moves, goals, saves
- **Player reactions**: Celebration detection works even in training contexts

## Troubleshooting

### Common Issues
1. **"No audio peaks detected"**
   - Solution: Fallback sampling automatically activates
   - Check: Audio track exists in video file
   - Try: Increase `FALLBACK_MAX_SAMPLES` for longer videos

2. **"GEMINI_API_KEY required"**  
   - Get API key from: https://aistudio.google.com/app/apikey
   - Set environment variable: `export GEMINI_API_KEY="your_key"`
   - Or create `.env` file with `GEMINI_API_KEY=your_key`

3. **"0-duration video output"**
   - Check: Input video is not corrupted
   - Try: Debug mode to see clip selection process
   - Use: Preview tool to verify frame analysis

4. **"No exciting frames validated"**
   - Lower `GEMINI_POS_THRESHOLD` to 0.3 for higher sensitivity
   - Use preview tool to see what Gemini is analyzing
   - Check video contains actual sports action (not just static shots)

5. **"xG model not working"**
   - Verify coefficients exist: `ls -la data/xg_coeffs.json`
   - Train model if missing: `python scripts/train_xg.py`
   - Test prediction: `from models.xg_model import predict_xg; predict_xg(x=108, y=40)`
   - Check for coordinate mapping issues in debug logs

### Performance Optimization
- **Cache usage**: Gemini responses are cached in SQLite to reduce API costs
- **Concurrent processing**: 8 concurrent API calls for faster analysis  
- **Fallback limits**: Maximum 20 fallback samples prevents excessive API usage
- **Quality settings**: JPEG compression optimized for AI analysis vs file size

## Testing
Run test suite: `pytest tests/ -v --asyncio-mode=auto`

## Recent Enhancements

### Universal Athlete Update
- ✅ **Universal video support**: No longer requires broadcast footage with scoreboards
- ✅ **Enhanced sensitivity**: Lower thresholds catch more subtle exciting moments
- ✅ **Fallback sampling**: Works even on silent or low-audio videos
- ✅ **Dynamic clipping**: Smart merging of nearby moments for extended highlights
- ✅ **Sports-specific AI prompts**: Gemini trained to recognize football/soccer actions
- ✅ **Comprehensive debugging tools**: HTML preview, JSON summaries, direct testing
- ✅ **0-duration video bug fix**: Enhanced validation and error handling
- ✅ **Simplified CLI**: Audio mode is default, scoreboard mode deprecated

### xG Integration Update  
- ✅ **Expected Goals model**: Trained on StatsBomb open data with logistic regression
- ✅ **Shot detection**: Identifies shot events during Gemini validation
- ✅ **Coordinate mapping**: Maps video positions to StatsBomb pitch coordinates
- ✅ **Smart ranking**: Blends Gemini confidence with xG for enhanced highlight selection
- ✅ **CLI enhancement**: `--use-xg` and `--xg-weight` flags for configurable integration
- ✅ **Debug visibility**: Real-time xG calculations and score blending in logs
- ✅ **JSON metadata**: xG values included in run summaries and clip data
- ✅ **Backward compatibility**: Works with or without xG, graceful fallback

### Web Application Architecture (Docker Deployment)
- ✅ **Containerized deployment**: Docker + nginx setup for production
- ✅ **Health monitoring**: Health check endpoints and container management
- ✅ **Environment configuration**: Configurable via environment variables
- ✅ **Scalable architecture**: Multi-service deployment with volume management