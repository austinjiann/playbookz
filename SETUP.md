# Quick Setup Guide

## 1. Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## 2. Set Up Environment

**Option 1: Using .env file (Recommended)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env file and replace with your actual key
# GEMINI_API_KEY=your_actual_key_here
```

**Option 2: Export environment variable**
```bash
export GEMINI_API_KEY="your_actual_key_here"
```

## 3. Install Dependencies

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

## 4. Test the Setup

```bash
# Test configuration
python config.py

# Should output: "Configuration validation passed!"
```

## 5. Ready to Use!

```bash
# Process a video file (replace with your actual video path)
python gen_highlights.py --video "/path/to/your/football_video.mp4" --mode hybrid
```

## Video File Requirements

- **Local files only** - no URLs or links needed
- **Supported formats**: MP4, AVI, MOV, MKV, FLV, WMV
- **Any resolution** - the system will handle it
- **Football content** - works best with actual football/soccer videos

## Troubleshooting

**"GEMINI_API_KEY environment variable is required"**
- Make sure your .env file exists and contains the API key
- Or export the environment variable before running

**"No module named 'xyz'"**
- Run `pip install -r requirements.txt` in your virtual environment

**"Video file not found"**
- Use the full absolute path to your video file
- Make sure the file exists and is accessible