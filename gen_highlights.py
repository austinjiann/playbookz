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
from gemini_filter import validate_audio_peaks


class HighlightClip:
    """Represents a highlight clip with metadata."""
    
    def __init__(self, start_time: float, end_time: float, source: str, confidence: float = 1.0):
        self.start_time = max(0, start_time)  # Ensure non-negative
        self.end_time = end_time
        self.source = source  # 'scoreboard', 'audio', 'hybrid'
        self.confidence = confidence
        self.duration = self.end_time - self.start_time
    
    def overlaps_with(self, other: 'HighlightClip', threshold: float = config.CLIP_OVERLAP_THRESHOLD) -> bool:
        """Check if this clip overlaps with another within threshold."""
        return abs(self.start_time - other.start_time) < threshold or abs(self.end_time - other.end_time) < threshold
    
    def merge_with(self, other: 'HighlightClip') -> 'HighlightClip':
        """Merge this clip with another overlapping clip."""
        start_time = min(self.start_time, other.start_time)
        end_time = max(self.end_time, other.end_time)
        
        # Combine sources and take max confidence
        sources = sorted(set([self.source, other.source]))
        source = '+'.join(sources)
        confidence = max(self.confidence, other.confidence)
        
        return HighlightClip(start_time, end_time, source, confidence)
    
    def __repr__(self):
        return f"HighlightClip({self.start_time:.1f}-{self.end_time:.1f}s, {self.source}, conf={self.confidence:.2f})"


def extract_scoreboard_clips(video_path: str) -> List[HighlightClip]:
    """Extract highlight clips using scoreboard OCR method."""
    logger.info("Extracting highlights using scoreboard method")
    
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


async def extract_audio_clips(video_path: str) -> List[HighlightClip]:
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
        
        # Validate with Gemini Vision
        validated_timestamps = await validate_audio_peaks(video_path, peak_timestamps)
        
        # Create clips around validated peaks
        clips = []
        for timestamp in validated_timestamps:
            start_time = timestamp - config.PRE_SEC
            end_time = timestamp + config.POST_SEC
            
            # Find corresponding peak for confidence
            confidence = 0.7  # Default
            for peak in audio_peaks:
                if abs(peak.timestamp - timestamp) < 1.0:  # Within 1 second
                    confidence = peak.confidence
                    break
            
            clip = HighlightClip(start_time, end_time, 'audio', confidence)
            clips.append(clip)
        
        logger.info(f"Found {len(clips)} audio-based clips")
        return clips
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return []


def merge_overlapping_clips(clips: List[HighlightClip]) -> List[HighlightClip]:
    """Merge overlapping clips to avoid duplication."""
    if not clips:
        return []
    
    logger.info(f"Merging {len(clips)} clips")
    
    # Sort clips by start time
    clips.sort(key=lambda c: c.start_time)
    
    merged = [clips[0]]
    for current in clips[1:]:
        last_merged = merged[-1]
        
        if current.overlaps_with(last_merged):
            # Merge with the last clip
            merged[-1] = last_merged.merge_with(current)
        else:
            # No overlap, add as new clip
            merged.append(current)
    
    logger.info(f"Merged to {len(merged)} clips")
    return merged


def fast_concat(clips: List[HighlightClip], video_path: str, output_path: str) -> Optional[str]:
    """Fast concatenation of video clips using MoviePy."""
    if not clips:
        logger.warning("No clips to concatenate")
        return None
    
    logger.info(f"Concatenating {len(clips)} clips to {output_path}")
    
    try:
        # Load video
        video = VideoFileClip(video_path)
        video_duration = video.duration
        
        # Create subclips
        subclips = []
        for clip in clips:
            # Ensure clip is within video bounds
            start = max(0, clip.start_time)
            end = min(video_duration, clip.end_time)
            
            if end > start:
                subclip = video.subclip(start, end)
                subclips.append(subclip)
                logger.debug(f"Added clip: {start:.1f}-{end:.1f}s ({clip.source})")
        
        if not subclips:
            logger.error("No valid clips to concatenate")
            return None
        
        # Concatenate clips
        final_video = concatenate_videoclips(subclips)
        
        # Write output
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Cleanup
        video.close()
        final_video.close()
        for subclip in subclips:
            subclip.close()
        
        logger.info(f"Highlights saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Video concatenation failed: {e}")
        return None


async def generate_highlights(
    video_path: str, 
    mode: str = 'hybrid', 
    output_path: str = 'highlights.mp4'
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
            audio_clips = await extract_audio_clips(video_path)
            clips.extend(audio_clips)
        
        # Merge overlapping clips
        merged_clips = merge_overlapping_clips(clips)
        
        if not merged_clips:
            logger.warning("No highlight clips found")
            return None
        
        # Sort by confidence and take top clips if too many
        merged_clips.sort(key=lambda c: c.confidence, reverse=True)
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
        
        return result_path
        
    except Exception as e:
        logger.error(f"Highlight generation failed: {e}")
        return None


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
                "confidence": clip.confidence
            } for clip in clips
        ] if clips else [],
        "total_duration": sum(clip.duration for clip in clips) if clips else 0
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate football highlights")
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--mode', choices=['scoreboard', 'audio', 'hybrid'], 
                       default='hybrid', help='Detection mode')
    parser.add_argument('--output', default='highlights.mp4', help='Output video path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.debug else config.LOG_LEVEL
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )
    
    # Validate configuration
    try:
        if args.mode in ['audio', 'hybrid']:
            config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Generate highlights
    clips = []
    try:
        result_path = await generate_highlights(args.video, args.mode, args.output)
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