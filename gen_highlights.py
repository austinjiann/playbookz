"""
Enhanced football highlights generator with multiple detection modes:
- Scoreboard-based (original OCR method)
- Audio-based (excitement peak detection + Gemini validation)
- Hybrid (combines both methods)
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips
from loguru import logger

import config
from get_dataframe import get_dataframe
from audio_analysis import get_audio_peaks
from gemini_filter import validate_audio_peaks, validate_audio_peaks_with_xg, analyze_video_vision_first

# xG model integration with safe fallback
try:
    from models.xg_model import predict_xg
except Exception:
    predict_xg = None


class HighlightClip:
    """Represents a highlight clip with metadata."""
    
    def __init__(self, start_time: float, end_time: float, source: str, confidence: float = 1.0, xg: Optional[float] = None):
        self.start_time = max(0, start_time)  # Ensure non-negative
        self.end_time = end_time
        self.source = source  # 'scoreboard', 'audio', 'hybrid'
        self.confidence = confidence
        self.xg = xg  # Expected Goals value if available
        self.duration = self.end_time - self.start_time
    
    def overlaps_with(self, other: 'HighlightClip', threshold: float = config.CLIP_OVERLAP_THRESHOLD) -> bool:
        """Check if this clip overlaps with another within threshold."""
        return abs(self.start_time - other.start_time) < threshold or abs(self.end_time - other.end_time) < threshold
    
    def is_within_merge_window(self, other: 'HighlightClip') -> bool:
        """Check if clips are within dynamic merge window for extension."""
        if not config.DYNAMIC_CLIP_EXTENSION:
            return False
        
        # Check if clips are close enough to merge
        time_gap = min(
            abs(self.end_time - other.start_time),    # Gap between end of self and start of other
            abs(other.end_time - self.start_time)     # Gap between end of other and start of self
        )
        return time_gap <= config.MERGE_WINDOW_SEC
    
    def merge_with(self, other: 'HighlightClip') -> 'HighlightClip':
        """Merge this clip with another overlapping clip with dynamic extension."""
        start_time = min(self.start_time, other.start_time)
        end_time = max(self.end_time, other.end_time)
        
        # If dynamic extension is enabled and clips are close, extend further
        if config.DYNAMIC_CLIP_EXTENSION and self.is_within_merge_window(other):
            # Extend the merged clip slightly to capture more context
            extension = config.MERGE_WINDOW_SEC / 2  # Extend by half the merge window
            start_time = max(0, start_time - extension)
            end_time = end_time + extension
        
        # Combine sources and take max confidence
        sources = sorted(set([self.source, other.source]))
        source = '+'.join(sources)
        confidence = max(self.confidence, other.confidence)
        
        # Take max xG if available
        merged_xg = None
        if self.xg is not None and other.xg is not None:
            merged_xg = max(self.xg, other.xg)
        elif self.xg is not None:
            merged_xg = self.xg
        elif other.xg is not None:
            merged_xg = other.xg
        
        return HighlightClip(start_time, end_time, source, confidence, xg=merged_xg)
    
    def __repr__(self):
        xg_str = f", xG={self.xg:.3f}" if self.xg is not None else ""
        return f"HighlightClip({self.start_time:.1f}-{self.end_time:.1f}s, {self.source}, conf={self.confidence:.2f}{xg_str})"


def _blend_conf_with_xg(base_conf: float, xg: Optional[float], use_xg: bool, xg_weight: float) -> float:
    """Blend Gemini confidence with xG score for enhanced ranking."""
    if not use_xg or xg is None:
        return base_conf
    w = max(0.0, min(1.0, xg_weight))
    # Blend: both values expected in [0,1] range
    return (1.0 - w) * base_conf + w * xg


def extract_scoreboard_clips(video_path: str) -> List[HighlightClip]:
    """
    DEPRECATED: Extract highlight clips using scoreboard OCR method.
    This method is preserved for future broadcast video support but disabled by default.
    Use audio mode for universal athlete video analysis.
    """
    if not config.ENABLE_SCOREBOARD:
        logger.warning("Scoreboard analysis is disabled - skipping OCR extraction")
        return []
    
    logger.info("Extracting highlights using scoreboard method (DEPRECATED)")
    
    try:
        # Use existing OCR-based analysis
        csv_path = get_dataframe(video_path)
        df = pd.read_csv(csv_path)
        
        if df.empty:
            logger.warning("No scoreboard data found")
            return []
        
        # Clean OCR results
        df["Score1"] = df["Score1"].replace(["O"], "0")
        df["Score2"] = df["Score2"].replace(["O"], "0")
        
        # Detect score changes
        df["Score1_change"] = df["Score1"].shift(1, fill_value=df["Score1"].iloc[0]) != df["Score1"]
        df["Score2_change"] = df["Score2"].shift(1, fill_value=df["Score2"].iloc[0]) != df["Score2"]
        
        # Get timestamps for score changes
        score1_changes = df.query('Score1_change == True')['Timestamp'].tolist()
        score2_changes = df.query('Score2_change == True')['Timestamp'].tolist()
        
        # Combine all goal timestamps
        goal_timestamps = sorted(set(score1_changes + score2_changes))
        
        # Create clips around goal events
        clips = []
        for timestamp in goal_timestamps:
            start_time = timestamp - config.PRE_SEC
            end_time = timestamp + config.POST_SEC
            clip = HighlightClip(start_time, end_time, 'scoreboard', confidence=0.9)
            clips.append(clip)
        
        logger.info(f"Found {len(clips)} scoreboard-based clips")
        return clips
        
    except Exception as e:
        logger.error(f"Scoreboard extraction failed: {e}")
        return []


async def extract_vision_first_clips(video_path: str, use_xg: bool = False, 
                                     vision_stride: float = None, 
                                     vision_max_frames: int = None,
                                     vision_min_conf: float = None) -> List[HighlightClip]:
    """Extract highlight clips using vision-first systematic scanning."""
    logger.info("Extracting highlights using vision-first method")
    
    try:
        # Use CLI args or config defaults
        stride = vision_stride if vision_stride is not None else config.VISION_FRAME_STRIDE_SEC
        max_frames = vision_max_frames if vision_max_frames is not None else config.VISION_MAX_FRAMES
        min_conf = vision_min_conf if vision_min_conf is not None else config.VISION_MIN_EVENT_CONF
        
        # Run vision-first analysis
        events = analyze_video_vision_first(
            video_path=video_path,
            frame_stride_sec=stride,
            micro_window_sec=config.VISION_MICRO_WINDOW_SEC,
            micro_step_sec=config.VISION_MICRO_STEP_SEC,
            min_event_conf=min_conf,
            goal_backtrack_sec=config.VISION_GOAL_BACKTRACK_SEC,
            max_events=config.VISION_MAX_EVENTS,
            max_frames=max_frames
        )
        
        if not events:
            logger.warning("No events detected in vision-first analysis")
            return []
        
        # Convert events to HighlightClip objects with event-aware padding
        clips = []
        for event in events:
            timestamp = event['timestamp']
            label = event.get('label', '').lower()
            confidence = event.get('confidence', 0.7)
            xg_value = event.get('xg')
            
            # Event-aware clip padding
            if any(pattern in label for pattern in ['goal', 'scores']):
                pre_sec = config.CLIP_PAD_GOAL_PRE_SEC
                post_sec = config.CLIP_PAD_GOAL_POST_SEC
            elif any(pattern in label for pattern in ['shot', 'attempt']):
                pre_sec = config.CLIP_PAD_SHOT_PRE_SEC  
                post_sec = config.CLIP_PAD_SHOT_POST_SEC
            else:
                # Default padding for celebrations, saves, etc.
                pre_sec = config.PRE_SEC
                post_sec = config.POST_SEC
            
            start_time = timestamp - pre_sec
            end_time = timestamp + post_sec
            
            clip = HighlightClip(start_time, end_time, 'vision-first', confidence, xg=xg_value)
            clips.append(clip)
        
        logger.info(f"Found {len(clips)} vision-first clips")
        return clips
        
    except Exception as e:
        logger.error(f"Vision-first extraction failed: {e}")
        return []


async def extract_audio_clips(video_path: str, use_xg: bool = False) -> List[HighlightClip]:
    """Extract highlight clips using audio peak detection + Gemini validation."""
    logger.info("Extracting highlights using audio method")
    
    try:
        # Detect audio peaks
        audio_peaks = get_audio_peaks(video_path, use_bandpass=True)
        
        if not audio_peaks:
            logger.warning("No audio peaks detected")
            return []
        
        # Extract timestamps from top peaks
        peak_timestamps = [peak.timestamp for peak in audio_peaks[:20]]  # Top 20 peaks
        
        # Validate with Gemini Vision (with or without xG)
        if use_xg and predict_xg is not None:
            validated_events = await validate_audio_peaks_with_xg(video_path, peak_timestamps)
        else:
            validated_timestamps = await validate_audio_peaks(video_path, peak_timestamps)
            validated_events = [{'timestamp': ts, 'confidence': 0.8, 'xg': None} for ts in validated_timestamps]
        
        # Create clips around validated events
        clips = []
        for event in validated_events:
            timestamp = event['timestamp']
            start_time = timestamp - config.PRE_SEC
            end_time = timestamp + config.POST_SEC
            
            # Find corresponding peak for base confidence
            confidence = event.get('confidence', 0.7)
            for peak in audio_peaks:
                if abs(peak.timestamp - timestamp) < 1.0:  # Within 1 second
                    confidence = peak.confidence
                    break
            
            clip = HighlightClip(start_time, end_time, 'audio', confidence, xg=event.get('xg'))
            clips.append(clip)
        
        logger.info(f"Found {len(clips)} audio-based clips")
        return clips
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return []


def merge_overlapping_clips(clips: List[HighlightClip]) -> List[HighlightClip]:
    """Merge overlapping clips with dynamic extension support."""
    if not clips:
        return []
    
    logger.info(f"Merging {len(clips)} clips (dynamic extension: {config.DYNAMIC_CLIP_EXTENSION})")
    
    # Sort clips by start time
    clips.sort(key=lambda c: c.start_time)
    
    merged = [clips[0]]
    for current in clips[1:]:
        last_merged = merged[-1]
        
        # Check for traditional overlap or dynamic merge window
        should_merge = (
            current.overlaps_with(last_merged) or 
            (config.DYNAMIC_CLIP_EXTENSION and current.is_within_merge_window(last_merged))
        )
        
        if should_merge:
            # Merge with the last clip
            merged_clip = last_merged.merge_with(current)
            merged[-1] = merged_clip
            logger.debug(f"Merged clips: {last_merged.start_time:.1f}-{last_merged.end_time:.1f}s + {current.start_time:.1f}-{current.end_time:.1f}s → {merged_clip.start_time:.1f}-{merged_clip.end_time:.1f}s")
        else:
            # No overlap or merge condition, add as new clip
            merged.append(current)
    
    logger.info(f"Merged to {len(merged)} clips with total duration: {sum(c.duration for c in merged):.1f}s")
    return merged


def fast_concat(clips: List[HighlightClip], video_path: str, output_path: str) -> Optional[str]:
    """Fast concatenation of video clips using MoviePy with enhanced debugging."""
    if not clips:
        logger.warning("No clips to concatenate - returning None")
        return None
    
    logger.info(f"Concatenating {len(clips)} clips to {output_path}")
    logger.info(f"Clips to process: {[f'{c.start_time:.1f}-{c.end_time:.1f}s ({c.source})' for c in clips]}")
    
    try:
        # Load video
        video = VideoFileClip(video_path)
        video_duration = video.duration
        logger.info(f"Source video duration: {video_duration:.1f}s")
        
        # Create subclips with validation
        subclips = []
        valid_clips_found = 0
        
        for i, clip in enumerate(clips):
            # Ensure clip is within video bounds
            start = max(0, clip.start_time)
            end = min(video_duration, clip.end_time)
            
            # Validate clip duration
            clip_duration = end - start
            if clip_duration <= 0:
                logger.warning(f"Clip {i+1} has invalid duration: {start:.1f}-{end:.1f}s (duration: {clip_duration:.1f}s)")
                continue
            
            if clip_duration < 1.0:  # Skip clips shorter than 1 second
                logger.warning(f"Clip {i+1} too short: {clip_duration:.1f}s, skipping")
                continue
            
            try:
                subclip = video.subclip(start, end)
                # Validate subclip has content
                if subclip.duration > 0:
                    subclips.append(subclip)
                    valid_clips_found += 1
                    logger.info(f"✅ Added clip {i+1}: {start:.1f}-{end:.1f}s (duration: {clip_duration:.1f}s) from {clip.source}")
                else:
                    logger.warning(f"❌ Clip {i+1} has 0 duration after subclip creation")
            except Exception as e:
                logger.error(f"❌ Failed to create subclip {i+1}: {e}")
                continue
        
        if not subclips or valid_clips_found == 0:
            logger.error(f"No valid clips found! Processed {len(clips)} input clips, got {len(subclips)} subclips")
            video.close()
            return None
        
        logger.info(f"Successfully created {len(subclips)} valid subclips")
        
        # Concatenate clips
        try:
            final_video = concatenate_videoclips(subclips)
            logger.info(f"Concatenation successful, final video duration: {final_video.duration:.1f}s")
        except Exception as e:
            logger.error(f"Failed to concatenate clips: {e}")
            video.close()
            for subclip in subclips:
                subclip.close()
            return None
        
        # Write output with better error handling
        try:
            logger.info(f"Writing video to {output_path}...")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            logger.info(f"✅ Video written successfully to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write video file: {e}")
            return None
        finally:
            # Cleanup
            video.close()
            final_video.close()
            for subclip in subclips:
                subclip.close()
        
        # Verify output file
        try:
            from pathlib import Path
            output_file = Path(output_path)
            if output_file.exists() and output_file.stat().st_size > 1000:  # At least 1KB
                logger.info(f"✅ Output file verified: {output_path} ({output_file.stat().st_size} bytes)")
                return output_path
            else:
                logger.error(f"❌ Output file invalid or too small: {output_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to verify output file: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Video concatenation failed with exception: {e}")
        return None


async def generate_highlights(
    video_path: str, 
    mode: str = 'hybrid', 
    output_path: str = 'highlights.mp4',
    use_xg: bool = False,
    xg_weight: float = 0.3,
    args=None  # Pass CLI args for vision-first tunability
) -> Optional[str]:
    """
    Generate highlights using specified mode.
    
    Args:
        video_path: Path to input video
        mode: 'scoreboard', 'audio', or 'hybrid'
        output_path: Output video path
        
    Returns:
        Path to generated highlights video or None if failed
    """
    logger.info(f"Generating highlights from {video_path} using {mode} mode")
    
    # Validate inputs
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return None
    
    clips = []
    
    try:
        # Extract clips based on mode
        if mode in ['scoreboard', 'hybrid']:
            scoreboard_clips = extract_scoreboard_clips(video_path)
            clips.extend(scoreboard_clips)
        
        if mode in ['audio', 'hybrid']:
            audio_clips = await extract_audio_clips(video_path, use_xg=use_xg)
            clips.extend(audio_clips)
        
        if getattr(args, 'experimental_vision_first', False):
            # Extract vision-first parameters from args if available
            vision_stride = getattr(args, 'vision_stride', None) if args else None
            vision_max_frames = getattr(args, 'vision_max_frames', None) if args else None
            vision_min_conf = getattr(args, 'vision_min_conf', None) if args else None
            
            vision_clips = await extract_vision_first_clips(
                video_path, use_xg=use_xg,
                vision_stride=vision_stride,
                vision_max_frames=vision_max_frames, 
                vision_min_conf=vision_min_conf
            )
            clips.extend(vision_clips)
        
        # Merge overlapping clips
        merged_clips = merge_overlapping_clips(clips)
        
        if not merged_clips:
            logger.warning("No highlight clips found")
            return None, []
        
        # Apply xG-enhanced ranking if enabled
        for clip in merged_clips:
            clip.blended_score = _blend_conf_with_xg(clip.confidence, clip.xg, use_xg, xg_weight)
            if use_xg and clip.xg is not None:
                logger.debug(f"Clip {clip.start_time:.1f}s: Gemini={clip.confidence:.3f} + xG={clip.xg:.3f} → Blended={clip.blended_score:.3f}")
        
        # Sort by blended score and take top clips if too many
        merged_clips.sort(key=lambda c: getattr(c, 'blended_score', c.confidence), reverse=True)
        if len(merged_clips) > 10:  # Limit to top 10 clips
            merged_clips = merged_clips[:10]
            logger.info(f"Limited to top {len(merged_clips)} clips")
        
        # Sort by time for final video
        merged_clips.sort(key=lambda c: c.start_time)
        
        # Generate final video
        result_path = fast_concat(merged_clips, video_path, output_path)
        
        # Log summary
        total_duration = sum(clip.duration for clip in merged_clips)
        logger.info(f"Generated {len(merged_clips)} clips, total duration: {total_duration:.1f}s")
        
        return result_path, merged_clips
        
    except Exception as e:
        logger.error(f"Highlight generation failed: {e}")
        return None, []


def create_run_summary(
    video_path: str, 
    mode: str, 
    output_path: str, 
    clips: List[HighlightClip],
    success: bool
) -> str:
    """Create JSON summary of the run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    summary_path = f"run-summary-{timestamp}.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_video": video_path,
        "mode": mode,
        "output_video": output_path,
        "success": success,
        "clips_found": len(clips),
        "clips": [
            {
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "source": clip.source,
                "confidence": clip.confidence,
                "xg": None if clip.xg is None else float(clip.xg),
                "blended_score": getattr(clip, 'blended_score', clip.confidence)
            } for clip in clips
        ] if clips else [],
        "total_duration": sum(clip.duration for clip in clips) if clips else 0
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate football highlights for athletes")
    parser.add_argument('--video', required=True, help='Path to input video')
    
    # Determine available modes based on configuration
    available_modes = ['audio']
    if config.ENABLE_SCOREBOARD:
        available_modes.extend(['scoreboard', 'hybrid'])
        default_mode = 'hybrid'
        mode_help = 'Detection mode: audio (universal), scoreboard (broadcast only), hybrid (combines both)'
    else:
        default_mode = 'audio'
        mode_help = 'Detection mode: audio (universal sports video analysis with crowd excitement detection)'
    
    parser.add_argument('--mode', choices=available_modes, 
                       default=default_mode, help=mode_help)
    parser.add_argument('--output', default='highlights.mp4', help='Output video path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--use-xg', action='store_true', help='Use xG to influence highlight ranking when available')
    parser.add_argument('--xg-weight', type=float, default=0.3, help='Weight (0..1) to blend xG with model confidence')
    
    # Experimental vision-first mode (hidden from main help)
    parser.add_argument('--experimental-vision-first', action='store_true', 
                       help=argparse.SUPPRESS)  # Hidden experimental feature
    parser.add_argument('--vision-stride', type=float, default=config.VISION_FRAME_STRIDE_SEC, 
                       help=argparse.SUPPRESS)
    parser.add_argument('--vision-max-frames', type=int, default=config.VISION_MAX_FRAMES,
                       help=argparse.SUPPRESS)
    parser.add_argument('--vision-min-conf', type=float, default=config.VISION_MIN_EVENT_CONF,
                       help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Validate scoreboard mode usage
    if args.mode in ['scoreboard', 'hybrid'] and not config.ENABLE_SCOREBOARD:
        logger.error(f"Scoreboard mode is disabled. Use --mode audio for universal athlete support.")
        return 1
    
    # Validate xG usage
    if args.use_xg and predict_xg is None:
        logger.warning("xG model not available - continuing without xG integration")
        args.use_xg = False
    
    # Configure logging
    log_level = "DEBUG" if args.debug else config.LOG_LEVEL
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )
    
    # Show mode info
    if args.mode == 'audio':
        logger.info("🎵 Audio mode: Universal analysis using crowd excitement detection + AI validation")
    elif args.mode == 'scoreboard':
        logger.info("📊 Scoreboard mode: Broadcast video with visible scoreboard")
    elif args.mode == 'hybrid':
        logger.info("🔄 Hybrid mode: Combines audio analysis + scoreboard detection")
    
    if getattr(args, 'experimental_vision_first', False):
        logger.info(f"👁️ [EXPERIMENTAL] Vision-first mode enabled ({args.vision_stride:.2f}s stride, max {args.vision_max_frames} frames)")
    
    # Validate configuration
    try:
        if args.mode in ['audio', 'hybrid'] or getattr(args, 'experimental_vision_first', False):
            config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Generate highlights
    clips = []
    try:
        result_data = await generate_highlights(
            args.video, 
            args.mode, 
            args.output,
            use_xg=args.use_xg,
            xg_weight=args.xg_weight,
            args=args  # Pass args for vision-first CLI tunability
        )
        
        if isinstance(result_data, tuple):
            result_path, clips = result_data
        else:
            result_path = result_data
            clips = []
            
        success = result_path is not None
        
        if success:
            logger.success(f"Highlights generated successfully: {result_path}")
        else:
            logger.error("Failed to generate highlights")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        success = False
    
    # Create summary
    if config.ENABLE_JSON_SUMMARY:
        summary_path = create_run_summary(args.video, args.mode, args.output, clips, success)
        logger.info(f"Run summary saved to {summary_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))