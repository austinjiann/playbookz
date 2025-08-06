"""
Gemini Vision API integration for validating football excitement moments.
Includes caching, retry logic, and async processing.
"""
import asyncio
import sqlite3
import hashlib
import base64
import json
from typing import List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
import google.generativeai as genai
from loguru import logger

import config


class GeminiCache:
    """SQLite-based cache for Gemini API responses to avoid duplicate calls."""
    
    def __init__(self, db_path: Path = config.GEMINI_CACHE_DB):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with cache table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gemini_cache (
                    frame_hash TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get(self, frame_hash: str) -> Optional[Tuple[bool, float]]:
        """Get cached response for frame hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response, confidence FROM gemini_cache WHERE frame_hash = ?",
                (frame_hash,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0]), row[1]
        return None
    
    def set(self, frame_hash: str, response: bool, confidence: float):
        """Cache response for frame hash."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO gemini_cache (frame_hash, response, confidence) VALUES (?, ?, ?)",
                (frame_hash, json.dumps(response), confidence)
            )
            conn.commit()
    
    def clear_old_entries(self, days: int = 30):
        """Clear cache entries older than specified days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM gemini_cache WHERE timestamp < datetime('now', '-{} days')".format(days)
            )
            conn.commit()


class GeminiFilter:
    """Async Gemini Vision API client with caching and retry logic."""
    
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        self.cache = GeminiCache()
        self.semaphore = asyncio.Semaphore(config.GEMINI_MAX_CONCURRENT)
        
        # Prompt for football moment detection
        self.prompt = """
        You are analyzing a single frame from a football (soccer) video to determine if it shows an exciting moment.
        
        Look for these indicators of exciting football moments:
        - Players celebrating (arms raised, hugging, running with joy)
        - Goal celebrations or reactions
        - Crowd celebrations visible
        - Players in emotional states (joy, excitement)
        - Multiple players converging or reacting to an event
        
        Do NOT consider these as exciting:
        - Normal gameplay without celebration
        - Players walking or in neutral poses
        - Static shots of the field
        - Referee or coaching staff without player excitement
        
        Respond with ONLY "YES" if this appears to be an exciting football moment with visible celebration or strong emotional reaction.
        Respond with ONLY "NO" if this appears to be normal gameplay or non-exciting content.
        
        Your response must be exactly one word: YES or NO.
        """
    
    def _hash_frame(self, frame: np.ndarray) -> str:
        """Generate SHA-1 hash of frame for caching."""
        # Convert frame to bytes and hash
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        return hashlib.sha1(frame_bytes).hexdigest()
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG for API."""
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame as JPEG")
        return base64.b64encode(buffer).decode('utf-8')
    
    async def _call_gemini_with_retry(self, frame_b64: str) -> Tuple[bool, float]:
        """Call Gemini API with exponential backoff retry."""
        for attempt in range(config.GEMINI_MAX_RETRIES):
            try:
                # Prepare the image data
                image_data = {
                    'mime_type': 'image/jpeg',
                    'data': frame_b64
                }
                
                # Make API call
                response = await asyncio.to_thread(
                    self.model.generate_content, 
                    [self.prompt, image_data]
                )
                
                # Parse response
                text = response.text.strip().upper()
                confidence = 0.8  # Default confidence for valid responses
                
                if text == "YES":
                    return True, confidence
                elif text == "NO":
                    return False, confidence
                else:
                    logger.warning(f"Unexpected Gemini response: {text}")
                    return False, 0.1
                    
            except Exception as e:
                wait_time = (config.GEMINI_BACKOFF ** attempt)
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                
                if attempt < config.GEMINI_MAX_RETRIES - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All Gemini API retry attempts failed")
                    raise
    
    async def is_exciting_frame(self, frame: np.ndarray) -> bool:
        """
        Determine if a frame contains an exciting football moment.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            True if frame appears to show excitement/celebration
        """
        async with self.semaphore:  # Limit concurrent calls
            # Generate frame hash for caching
            frame_hash = self._hash_frame(frame)
            
            # Check cache first
            cached_result = self.cache.get(frame_hash)
            if cached_result is not None:
                is_exciting, confidence = cached_result
                logger.debug(f"Cache hit for frame {frame_hash[:8]}: {is_exciting} (conf: {confidence:.2f})")
                return is_exciting and confidence >= config.GEMINI_POS_THRESHOLD
            
            # Encode frame for API
            frame_b64 = self._encode_frame(frame)
            
            # Call Gemini API
            try:
                is_exciting, confidence = await self._call_gemini_with_retry(frame_b64)
                
                # Cache the result
                self.cache.set(frame_hash, is_exciting, confidence)
                
                logger.debug(f"Gemini response for frame {frame_hash[:8]}: {is_exciting} (conf: {confidence:.2f})")
                return is_exciting and confidence >= config.GEMINI_POS_THRESHOLD
                
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                return False
    
    async def filter_frames_batch(self, frames_with_timestamps: List[Tuple[float, np.ndarray]]) -> List[float]:
        """
        Filter a batch of frames and return timestamps of exciting ones.
        
        Args:
            frames_with_timestamps: List of (timestamp, frame) tuples
            
        Returns:
            List of timestamps for frames classified as exciting
        """
        logger.info(f"Filtering batch of {len(frames_with_timestamps)} frames")
        
        tasks = []
        for timestamp, frame in frames_with_timestamps:
            task = asyncio.create_task(self._process_frame_with_timestamp(timestamp, frame))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract successful results
        exciting_timestamps = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Frame processing failed: {result}")
            elif result is not None:
                exciting_timestamps.append(result)
        
        logger.info(f"Found {len(exciting_timestamps)} exciting frames out of {len(frames_with_timestamps)}")
        return exciting_timestamps
    
    async def _process_frame_with_timestamp(self, timestamp: float, frame: np.ndarray) -> Optional[float]:
        """Process a single frame and return timestamp if exciting."""
        is_exciting = await self.is_exciting_frame(frame)
        return timestamp if is_exciting else None


# Global filter instance
_gemini_filter = None

def get_gemini_filter() -> GeminiFilter:
    """Get singleton GeminiFilter instance."""
    global _gemini_filter
    if _gemini_filter is None:
        _gemini_filter = GeminiFilter()
    return _gemini_filter


async def validate_audio_peaks(video_path: str, peak_timestamps: List[float]) -> List[float]:
    """
    Validate audio peaks by checking if corresponding video frames show excitement.
    
    Args:
        video_path: Path to video file
        peak_timestamps: List of timestamp seconds to check
        
    Returns:
        List of timestamps that passed Gemini validation
    """
    if not peak_timestamps:
        return []
    
    logger.info(f"Validating {len(peak_timestamps)} audio peaks with Gemini Vision")
    
    # Extract frames at peak timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FALLBACK
    
    frames_with_timestamps = []
    for timestamp in peak_timestamps:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            frames_with_timestamps.append((timestamp, frame))
        else:
            logger.warning(f"Could not extract frame at {timestamp}s")
    
    cap.release()
    
    # Filter frames with Gemini
    filter_instance = get_gemini_filter()
    validated_timestamps = await filter_instance.filter_frames_batch(frames_with_timestamps)
    
    logger.info(f"Gemini validated {len(validated_timestamps)} out of {len(peak_timestamps)} peaks")
    return sorted(validated_timestamps)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python gemini_filter.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Test with a single frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video frame")
        sys.exit(1)
    
    async def test():
        filter_instance = get_gemini_filter()
        result = await filter_instance.is_exciting_frame(frame)
        print(f"Frame is exciting: {result}")
    
    asyncio.run(test())