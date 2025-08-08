"""
Gemini Vision API integration for validating football excitement moments.
Includes caching, retry logic, and async processing.
"""
import asyncio
import sqlite3
import hashlib
import base64
import json
import math
import itertools
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import cv2
import numpy as np
import google.generativeai as genai
from loguru import logger

import config
from utils.frames import get_duration_sec, read_frame_at_ts, iter_timestamps

# xG model integration with safe fallback
try:
    from models.xg_model import predict_xg
except Exception:
    predict_xg = None


# xG helper functions for shot context inference
def _infer_body_part_from_labels(labels: Dict[str, Any]) -> str:
    """Infer body part from Gemini classification keywords."""
    txt = ' '.join([str(v).lower() for v in labels.values()]) if isinstance(labels, dict) else str(labels).lower()
    if 'header' in txt or 'head' in txt:
        return 'head'
    if 'left' in txt:
        return 'left_foot'
    if 'right' in txt:
        return 'right_foot'
    return 'other'


def _infer_phase_is_set_piece(labels: Dict[str, Any]) -> bool:
    """Infer if shot is from set piece based on context."""
    txt = ' '.join([str(v).lower() for v in labels.values()]) if isinstance(labels, dict) else str(labels).lower()
    return any(k in txt for k in ['corner', 'free kick', 'penalty', 'set piece'])


def _infer_open_play(is_set_piece: bool) -> bool:
    """Determine if shot is from open play."""
    return not bool(is_set_piece)


def _placeholder_pitch_coordinates(frame_shape: Tuple[int, int], detection_box=None) -> Optional[Tuple[float, float]]:
    """
    Placeholder coordinate mapping for immediate testing.
    Returns approximate penalty area coordinates.
    TODO: Replace with homography or CV-based pitch detection.
    """
    # For now, assume shots are from penalty area (realistic for most highlights)
    x = 108.0  # 12 yards from goal (penalty spot area)
    y = 40.0   # Center of goal (StatsBomb 120x80 pitch)
    return x, y


def _estimate_xy_from_frame_meta(meta: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Try to get approximate (x,y) on a 120x80 StatsBomb grid.
    Prefer real coordinates if available, fallback to placeholder.
    """
    # Check if real coordinates are already available
    for key in ('pitch_x', 'pitch_y', 'sb_x', 'sb_y'):
        if key in meta:
            pass
    x = meta.get('pitch_x') or meta.get('sb_x')
    y = meta.get('pitch_y') or meta.get('sb_y')
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x), float(y)
    
    # Use placeholder mapping
    frame_shape = meta.get('frame_shape', (720, 1280))
    return _placeholder_pitch_coordinates(frame_shape)


# --- Vision-first helper functions ---
def _json_or_none(text: str):
    """Try to parse JSON, return None if failed."""
    try:
        return json.loads(text)
    except Exception:
        return None


def _bbox_center_xy(bbox):
    """Get center coordinates from bounding box [x1,y1,x2,y2]."""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
    return None


def _estimate_sb_xy_from_bbox_center(cx: float, cy: float, W: float, H: float, pitch_hint=None):
    """Map bbox center to StatsBomb coordinates with pitch hints."""
    # Naive normalization to StatsBomb 120x80
    x = max(0.0, min(120.0, 120.0 * (cx / max(1.0, W))))
    y = max(0.0, min(80.0, 80.0 * (cy / max(1.0, H))))
    
    if isinstance(pitch_hint, dict):
        side = (pitch_hint.get("goal_side") or "unknown").lower()
        if side == "right":
            x = min(120.0, x + 5.0)  # Attacking right goal
        elif side == "left":
            x = max(0.0, x - 5.0)   # Attacking left goal
    
    return x, y


def _maybe_compute_xg_from_entities(entities: Dict[str, Any]) -> Optional[float]:
    """Compute xG from JSON entities if bbox and frame dimensions available."""
    if predict_xg is None:
        return None
        
    ball_bbox = entities.get("ball_bbox")
    body_part = entities.get("body_part") or "other"
    phase = entities.get("phase") or "open_play"
    pitch_hint = entities.get("pitch_hint") or {}
    frame_w = entities.get("frame_w")
    frame_h = entities.get("frame_h")
    
    # Need bbox and frame dimensions for coordinate mapping
    if not isinstance(ball_bbox, (list, tuple)) or len(ball_bbox) != 4:
        return None
    if not isinstance(frame_w, (int, float)) or not isinstance(frame_h, (int, float)):
        return None
        
    # Get bbox center and map to StatsBomb coordinates
    center = _bbox_center_xy(ball_bbox)
    if center is None:
        return None
        
    cx, cy = center
    x, y = _estimate_sb_xy_from_bbox_center(cx, cy, float(frame_w), float(frame_h), pitch_hint)
    
    is_set_piece = (phase == "set_piece")
    
    try:
        return float(predict_xg(
            x=x, y=y, 
            body_part=body_part, 
            is_set_piece=is_set_piece, 
            is_open_play=not is_set_piece
        ))
    except Exception as e:
        logger.debug(f"xG calculation failed: {e}")
        return None


def _is_shot_like(label: str) -> bool:
    """Check if label indicates a shot event."""
    l = (label or '').lower()
    return any(k in l for k in ['shot', 'shoot', 'attempt', 'chance', 'strike'])


def _is_goal_like(label: str) -> bool:
    """Check if label indicates a goal event."""
    l = (label or '').lower()
    return any(k in l for k in ['goal', 'scores', 'ball crosses the line', 'in the net'])


def _is_celebration_like(label: str) -> bool:
    """Check if label indicates a celebration event."""
    l = (label or '').lower()
    return any(k in l for k in ['celebration', 'players celebrating', 'fans celebrating'])


def _dedup_events(events: List[Dict[str, Any]], tol_sec: float = 1.0) -> List[Dict[str, Any]]:
    """
    Deduplicate events by (timestamp, label, phase) to avoid merging shots→goals.
    """
    events = sorted(events, key=lambda e: e.get('timestamp', 0.0))
    merged = []
    
    for e in events:
        ts_key = round(e['timestamp'], 1)
        label_key = (e.get('label') or '').lower()
        phase_key = e.get('entities', {}).get('phase', 'unknown')
        event_key = (ts_key, label_key, phase_key)
        
        # Check if we have a similar event (same key within tolerance)
        merged_similar = False
        for i, prev in enumerate(merged):
            prev_ts_key = round(prev['timestamp'], 1) 
            prev_label_key = (prev.get('label') or '').lower()
            prev_phase_key = prev.get('entities', {}).get('phase', 'unknown')
            prev_key = (prev_ts_key, prev_label_key, prev_phase_key)
            
            if (abs(event_key[0] - prev_key[0]) <= tol_sec and 
                event_key[1] == prev_key[1] and 
                event_key[2] == prev_key[2]):
                # Merge: take max confidence and xG
                merged[i]['confidence'] = max(merged[i].get('confidence', 0), e.get('confidence', 0))
                if e.get('xg') is not None:
                    merged[i]['xg'] = max(merged[i].get('xg') or 0.0, e['xg'])
                merged_similar = True
                break
        
        if not merged_similar:
            merged.append(e)
    
    return merged


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
        
        # Use sports-specific prompt from config
        self.prompt = config.GEMINI_PROMPT_TEMPLATE
    
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


async def validate_audio_peaks_with_xg(video_path: str, peak_timestamps: List[float]) -> List[Dict[str, Any]]:
    """
    Validate audio peaks and compute xG for shot events.
    
    Args:
        video_path: Path to video file
        peak_timestamps: List of timestamp seconds to check
        
    Returns:
        List of event dictionaries with timestamp, validation result, and xG data
    """
    if not peak_timestamps:
        return []
    
    logger.info(f"Validating {len(peak_timestamps)} audio peaks with Gemini Vision + xG")
    
    # Extract frames at peak timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FALLBACK
    
    frames_with_timestamps = []
    for timestamp in peak_timestamps:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            frame_meta = {
                'frame_shape': frame.shape[:2],
                'timestamp': timestamp,
                'frame_number': frame_number
            }
            frames_with_timestamps.append((timestamp, frame, frame_meta))
        else:
            logger.warning(f"Could not extract frame at {timestamp}s")
    
    cap.release()
    
    # Process frames with Gemini + xG
    filter_instance = get_gemini_filter()
    events = []
    
    for timestamp, frame, meta in frames_with_timestamps:
        try:
            is_exciting = await filter_instance.is_exciting_frame(frame)
            
            event = {
                'timestamp': timestamp,
                'is_exciting': is_exciting,
                'confidence': 0.8 if is_exciting else 0.2,
                'xg': None,
                'meta': meta
            }
            
            # Compute xG if this appears to be a shot event
            if is_exciting and predict_xg is not None:
                xg_value = None
                try:
                    # Infer shot context - simplified for now
                    labels = {'action': 'shot'}  # Placeholder - could be enhanced with better analysis
                    xy = _estimate_xy_from_frame_meta(meta)
                    body_part = _infer_body_part_from_labels(labels)
                    is_set_piece = _infer_phase_is_set_piece(labels)
                    is_open_play = _infer_open_play(is_set_piece)
                    
                    if xy is not None:
                        x, y = xy
                        # Clamp coordinates to StatsBomb pitch
                        x = max(0.0, min(120.0, float(x)))
                        y = max(0.0, min(80.0, float(y)))
                        xg_value = float(predict_xg(
                            x=x,
                            y=y,
                            body_part=body_part,
                            is_set_piece=is_set_piece,
                            is_open_play=is_open_play
                        ))
                        logger.debug(f"Shot at {timestamp:.1f}s: xG={xg_value:.3f} (coords: {x:.1f}, {y:.1f})")
                except Exception as e:
                    logger.debug(f"xG calculation failed for {timestamp:.1f}s: {e}")
                
                event['xg'] = xg_value
            
            events.append(event)
            
        except Exception as e:
            logger.error(f"Failed to process frame at {timestamp:.1f}s: {e}")
    
    validated_events = [e for e in events if e['is_exciting']]
    logger.info(f"Validated {len(validated_events)} exciting events out of {len(peak_timestamps)} peaks")
    
    return validated_events


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


def _classify_frame_with_gemini(video_path: str, ts: float, prompt: str) -> Dict[str, Any]:
    """
    Classify a single frame using Gemini with JSON-first response parsing.
    Returns event dictionary with entities and xG calculation if applicable.
    """
    # Read frame at timestamp
    frame = read_frame_at_ts(video_path, ts)
    if frame is None:
        return {}
    
    # Get frame dimensions for coordinate mapping
    frame_h, frame_w = frame.shape[:2]
    
    # Encode frame as JPEG for Gemini
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return {}
    
    # Create cache key with prompt name and stride info
    jpg_bytes = buffer.tobytes()
    cache_key = hashlib.sha1(jpg_bytes + prompt.encode() + f"vision_v1".encode()).hexdigest()
    
    # Get Gemini filter instance and check cache
    filter_instance = get_gemini_filter()
    cached_result = filter_instance.cache.get(cache_key)
    if cached_result is not None:
        response_data, _ = cached_result  # cached_result is (response, confidence)
        if isinstance(response_data, dict):
            return response_data
    
    # Prepare image data for Gemini
    image_data = {
        'mime_type': 'image/jpeg',
        'data': base64.b64encode(jpg_bytes).decode('utf-8')
    }
    
    try:
        # Call Gemini API
        response = filter_instance.model.generate_content([prompt, image_data])
        response_text = response.text.strip()
        
        # Try JSON-first parsing
        json_data = _json_or_none(response_text)
        if json_data is not None:
            # Enhance entities with frame dimensions
            if 'entities' not in json_data:
                json_data['entities'] = {}
            json_data['entities']['frame_w'] = frame_w
            json_data['entities']['frame_h'] = frame_h
            
            # Cache the successful JSON result
            filter_instance.cache.set(cache_key, json_data, json_data.get('confidence', 0.0))
            return json_data
        
        # Fallback to keyword parsing if JSON failed
        else:
            fallback_data = {
                'label': response_text.lower(),
                'confidence': 0.6,  # Lower confidence for non-JSON responses
                'entities': {
                    'frame_w': frame_w,
                    'frame_h': frame_h
                },
                'schema': 'fallback'
            }
            filter_instance.cache.set(cache_key, fallback_data, 0.6)
            return fallback_data
            
    except Exception as e:
        logger.error(f"Gemini classification failed for timestamp {ts:.2f}s: {e}")
        return {}


def analyze_video_vision_first(
    video_path: str,
    frame_stride_sec: float,
    micro_window_sec: float, 
    micro_step_sec: float,
    min_event_conf: float,
    goal_backtrack_sec: float,
    max_events: int,
    max_frames: int
) -> List[Dict[str, Any]]:
    """
    Vision-first video analysis with systematic scanning and goal backtracking.
    
    Returns:
        List of event dicts: [{'timestamp': float, 'label': str, 'confidence': float, 'xg': float, 'entities': {}, ...}]
    """
    duration = get_duration_sec(video_path)
    if duration == 0:
        logger.error(f"Could not get duration for video: {video_path}")
        return []
    
    logger.info(f"Vision-first analysis: {duration:.1f}s video, {frame_stride_sec:.2f}s stride, max {max_frames} frames")
    
    events = []
    frames_processed = 0
    
    # Pass 1: Base stride scanning
    logger.info("Pass 1: Base stride scanning")
    base_timestamps = list(iter_timestamps(duration, frame_stride_sec))
    
    for ts in base_timestamps:
        if frames_processed >= max_frames:
            logger.warning(f"Reached max frames limit ({max_frames}), stopping base scan")
            break
            
        result = _classify_frame_with_gemini(video_path, ts, config.GEMINI_PROMPT_SCAN)
        frames_processed += 1
        
        if result and result.get('confidence', 0.0) >= min_event_conf:
            label = result.get('label', '').lower()
            if any(pattern in label for pattern in ['shot', 'goal', 'celebration', 'save']):
                event = {
                    'timestamp': float(ts),
                    'label': label,
                    'confidence': min(1.0, max(0.0, float(result.get('confidence', 0.0)))),  # Clamp [0,1]
                    'entities': result.get('entities', {}),
                    'meta': {'pass': 'base_scan'}
                }
                
                # Compute xG for shot-like events
                if _is_shot_like(label) or _is_goal_like(label):
                    event['xg'] = _maybe_compute_xg_from_entities(result.get('entities', {}))
                else:
                    event['xg'] = None
                    
                events.append(event)
                logger.debug(f"Base scan event at {ts:.2f}s: {label} (conf={event['confidence']:.3f}, xg={event.get('xg', 'N/A')})")
        
        if len(events) >= max_events:
            logger.warning(f"Reached max events limit ({max_events}), stopping base scan")
            break
    
    # Pass 2: Micro-burst sampling around detections
    if events and frames_processed < max_frames:
        logger.info(f"Pass 2: Micro-burst sampling around {len(events)} detections")
        seed_timestamps = [e['timestamp'] for e in events]
        
        # Generate micro-burst timestamps
        micro_timestamps = []
        for seed_ts in seed_timestamps:
            start_ts = max(0.0, seed_ts - micro_window_sec)
            end_ts = min(duration, seed_ts + micro_window_sec) 
            burst_timestamps = list(iter_timestamps(end_ts - start_ts, micro_step_sec))
            micro_timestamps.extend([start_ts + t for t in burst_timestamps])
        
        # Remove duplicates and existing timestamps
        existing_ts = {round(e['timestamp'], 1) for e in events}
        micro_timestamps = [ts for ts in set(micro_timestamps) if round(ts, 1) not in existing_ts]
        micro_timestamps.sort()
        
        for ts in micro_timestamps:
            if frames_processed >= max_frames or len(events) >= max_events:
                break
                
            result = _classify_frame_with_gemini(video_path, ts, config.GEMINI_PROMPT_SCAN)
            frames_processed += 1
            
            if result and result.get('confidence', 0.0) >= min_event_conf:
                label = result.get('label', '').lower()
                if any(pattern in label for pattern in ['shot', 'goal', 'celebration', 'save']):
                    event = {
                        'timestamp': float(ts),
                        'label': label,
                        'confidence': min(1.0, max(0.0, float(result.get('confidence', 0.0)))),
                        'entities': result.get('entities', {}),
                        'meta': {'pass': 'micro_burst'}
                    }
                    
                    if _is_shot_like(label) or _is_goal_like(label):
                        event['xg'] = _maybe_compute_xg_from_entities(result.get('entities', {}))
                    else:
                        event['xg'] = None
                        
                    events.append(event)
    
    # Pass 3: Goal backtracking
    if frames_processed < max_frames and len(events) < max_events:
        celebrations_and_goals = [e for e in events if _is_goal_like(e['label']) or _is_celebration_like(e['label'])]
        
        if celebrations_and_goals:
            logger.info(f"Pass 3: Goal backtracking from {len(celebrations_and_goals)} celebrations/goals")
            
            for celebration in celebrations_and_goals:
                if frames_processed >= max_frames or len(events) >= max_events:
                    break
                    
                backtrack_start = max(0.0, celebration['timestamp'] - goal_backtrack_sec)
                backtrack_timestamps = list(iter_timestamps(
                    celebration['timestamp'] - backtrack_start, 
                    micro_step_sec
                ))
                backtrack_timestamps = [backtrack_start + t for t in backtrack_timestamps]
                backtrack_timestamps.reverse()  # Search backwards
                
                # Stop backtracking once we find a higher-confidence goal/shot
                for ts in backtrack_timestamps:
                    if frames_processed >= max_frames:
                        break
                        
                    result = _classify_frame_with_gemini(video_path, ts, config.GEMINI_PROMPT_PRECISION)
                    frames_processed += 1
                    
                    if result and result.get('confidence', 0.0) >= min_event_conf:
                        label = result.get('label', '').lower() 
                        confidence = min(1.0, max(0.0, float(result.get('confidence', 0.0))))
                        
                        if (_is_goal_like(label) or _is_shot_like(label)) and confidence > celebration['confidence']:
                            event = {
                                'timestamp': float(ts),
                                'label': label,
                                'confidence': confidence,
                                'entities': result.get('entities', {}),
                                'meta': {'pass': 'backtrack', 'from_celebration': celebration['timestamp']}
                            }
                            
                            event['xg'] = _maybe_compute_xg_from_entities(result.get('entities', {}))
                            events.append(event)
                            logger.debug(f"Backtracked goal/shot at {ts:.2f}s (conf={confidence:.3f}) from celebration at {celebration['timestamp']:.2f}s")
                            break  # Stop backtracking for this celebration
    
    # Pass 4: Deduplication and final processing
    logger.info(f"Pass 4: Deduplicating {len(events)} events")
    events = _dedup_events(events, tol_sec=1.0)
    
    # Sort by timestamp for final output
    events.sort(key=lambda e: e['timestamp'])
    
    logger.info(f"Vision-first analysis complete: {len(events)} events from {frames_processed} frames")
    return events