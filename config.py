"""
Configuration parameters for the Football Highlights Generator.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / ".cache"
TOOLS_DIR = PROJECT_ROOT / "tools"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

# --- SYSTEM FEATURES ---
ENABLE_SCOREBOARD = False            # DEPRECATED: Disable scoreboard analysis for universal athlete support

# --- AUDIO ANALYSIS ---
AUDIO_SR = 22_050                    # Target sample rate for resampling
RMS_HOP = 512                        # Librosa RMS hop length
PEAK_PROMINENCE = 0.10               # Fraction of max RMS that counts as a peak (lowered for sensitivity)
PEAK_MIN_DIST_SEC = 2                # Minimum distance between peaks (seconds, reduced for rapid sequences)
BANDPASS = (1000, 3000)              # Cheering frequency band (Hz)
BANDPASS_THRESHOLD = 0.2             # Minimum bandpass energy ratio

# --- FALLBACK SAMPLING ---
FALLBACK_SAMPLING_ENABLED = True     # Enable fallback when no audio peaks found
FALLBACK_INTERVAL_SEC = 7            # Sample every N seconds as fallback
FALLBACK_MAX_SAMPLES = 20            # Maximum fallback samples to prevent excessive API calls

# --- GEMINI API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"    # Model for vision analysis
GEMINI_POS_THRESHOLD = 0.4           # Confidence threshold for positive detection (lowered for sensitivity)
GEMINI_MAX_RETRIES = 3               # Maximum retry attempts
GEMINI_BACKOFF = 2                   # Exponential backoff base (seconds)
GEMINI_MAX_CONCURRENT = 8            # Maximum concurrent API calls
GEMINI_CACHE_DB = CACHE_DIR / "gemini.db"  # SQLite cache file

# --- GEMINI PROMPTS ---
GEMINI_PROMPT_TEMPLATE = """
You are analyzing a single frame from a sports video to identify exciting or significant moments.

Look for these specific observable actions:
- GOALS: Ball crossing goal line, net movement, goalkeeper reactions
- SHOTS: Players shooting, ball towards goal, shooting stance
- SAVES: Goalkeeper diving, blocking, catching ball
- SKILLS: Dribbling, skillful moves, ball control in tight spaces
- CELEBRATIONS: Players with arms raised, jumping, hugging teammates
- TACKLES: Defensive challenges, slide tackles, contested possession
- FAST BREAKS: Players running at speed, counter-attacks

Do NOT consider as exciting:
- Normal passing or walking
- Static shots of field/crowd
- Referees or coaches without player action
- Players in neutral standing positions

Respond with ONLY "YES" if this frame shows any of the specific actions listed above.
Respond with ONLY "NO" if this appears to be normal gameplay without exciting action.

Your response must be exactly one word: YES or NO.
"""

# --- VIDEO CLIPPING ---
PRE_SEC = 12                         # Seconds before peak to include
POST_SEC = 12                        # Seconds after peak to include
CLIP_OVERLAP_THRESHOLD = 10          # Merge clips if closer than this (seconds)

# --- DYNAMIC CLIPPING ---
DYNAMIC_CLIP_EXTENSION = True        # Enable dynamic clip extension for nearby moments
MERGE_WINDOW_SEC = 4                 # Merge clips if within this window (seconds)

# --- SCOREBOARD ANALYSIS (existing) ---
FRAME_SKIP = 200                     # Process every Nth frame
OCR_CONFIDENCE = 0.90                # Minimum OCR confidence
SCOREBOARD_CROP = (40, 65, 110, 313)  # Y1, Y2, X1, X2 coordinates

# --- VIDEO PROCESSING ---
FRAME_RATE_FALLBACK = 30             # Default FPS if unable to detect

# --- LOGGING ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_JSON_SUMMARY = True           # Generate JSON run summaries

# --- VALIDATION ---
def validate_config():
    """Validate configuration parameters."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    if PEAK_PROMINENCE <= 0 or PEAK_PROMINENCE >= 1:
        raise ValueError("PEAK_PROMINENCE must be between 0 and 1")
    
    if PRE_SEC < 0 or POST_SEC < 0:
        raise ValueError("PRE_SEC and POST_SEC must be positive")
    
    if BANDPASS[0] >= BANDPASS[1]:
        raise ValueError("BANDPASS frequencies must be in ascending order")

if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed!")