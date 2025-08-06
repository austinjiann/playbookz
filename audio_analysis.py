"""
Audio analysis module for detecting excitement peaks in football videos.
Uses PyAV for fast audio demuxing and librosa for signal processing.
"""
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
import av
import librosa
import scipy.signal
from loguru import logger

import config


class AudioPeak:
    """Represents a detected audio peak with metadata."""
    
    def __init__(self, timestamp: float, amplitude: float, bandpass_ratio: float = 0.0):
        self.timestamp = timestamp
        self.amplitude = amplitude
        self.bandpass_ratio = bandpass_ratio
        self.confidence = self._calculate_confidence()
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence score based on amplitude and bandpass ratio."""
        amplitude_score = min(self.amplitude, 1.0)
        bandpass_score = min(self.bandpass_ratio / config.BANDPASS_THRESHOLD, 1.0)
        return (amplitude_score * 0.7) + (bandpass_score * 0.3)
    
    def __repr__(self):
        return f"AudioPeak(t={self.timestamp:.1f}s, amp={self.amplitude:.3f}, conf={self.confidence:.3f})"


def extract_audio_fast(video_path: str) -> str:
    """
    Extract audio from video using PyAV and save as temporary WAV file.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Path to extracted WAV file
    """
    logger.info(f"Extracting audio from {video_path}")
    
    # Create temporary file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    
    try:
        # Open video container
        container = av.open(video_path)
        audio_stream = container.streams.audio[0]
        
        # Create output container
        output = av.open(temp_wav.name, 'w')
        out_stream = output.add_stream('pcm_s16le', rate=config.AUDIO_SR, channels=1)
        
        # Process audio frames
        frame_count = 0
        for frame in container.decode(audio_stream):
            # Resample and convert to mono
            frame = frame.resample(format=av.AudioFormat('s16').packed, rate=config.AUDIO_SR, channels=1)
            
            for packet in out_stream.encode(frame):
                output.mux(packet)
            frame_count += 1
        
        # Flush encoder
        for packet in out_stream.encode():
            output.mux(packet)
            
        output.close()
        container.close()
        
        logger.info(f"Audio extracted successfully: {frame_count} frames processed")
        return temp_wav.name
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        Path(temp_wav.name).unlink(missing_ok=True)
        raise


def compute_rms_envelope(audio_path: str) -> Tuple[np.ndarray, float]:
    """
    Compute RMS envelope of audio signal.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (rms_envelope, duration_seconds)
    """
    logger.info(f"Computing RMS envelope for {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.AUDIO_SR, mono=True)
    duration = len(y) / sr
    
    # Compute RMS envelope
    rms = librosa.feature.rms(y=y, hop_length=config.RMS_HOP)[0]
    
    logger.info(f"RMS envelope computed: {len(rms)} frames, {duration:.1f}s duration")
    return rms, duration


def apply_bandpass_filter(audio_path: str) -> np.ndarray:
    """
    Apply bandpass filter to emphasize crowd cheering frequencies.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Bandpass energy envelope
    """
    logger.info(f"Applying bandpass filter {config.BANDPASS[0]}-{config.BANDPASS[1]} Hz")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.AUDIO_SR, mono=True)
    
    # Design bandpass filter
    nyquist = sr / 2
    low = config.BANDPASS[0] / nyquist
    high = config.BANDPASS[1] / nyquist
    
    # Apply bandpass filter
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    y_filtered = scipy.signal.filtfilt(b, a, y)
    
    # Compute energy envelope
    frame_length = config.RMS_HOP * 2  # Match RMS frame size
    bandpass_energy = []
    
    for i in range(0, len(y_filtered) - frame_length, config.RMS_HOP):
        frame = y_filtered[i:i + frame_length]
        energy = np.mean(frame ** 2)
        bandpass_energy.append(energy)
    
    return np.array(bandpass_energy)


def detect_peaks(rms_envelope: np.ndarray, duration: float, 
                bandpass_energy: np.ndarray = None) -> List[AudioPeak]:
    """
    Detect audio peaks that exceed prominence threshold.
    
    Args:
        rms_envelope: RMS amplitude envelope
        duration: Total audio duration in seconds
        bandpass_energy: Optional bandpass energy envelope
        
    Returns:
        List of detected AudioPeak objects
    """
    logger.info("Detecting audio peaks")
    
    # Normalize RMS envelope
    rms_max = np.max(rms_envelope)
    if rms_max == 0:
        logger.warning("RMS envelope is empty or all zeros")
        return []
    
    rms_norm = rms_envelope / rms_max
    
    # Find peaks above prominence threshold
    prominence_threshold = config.PEAK_PROMINENCE
    min_distance_frames = int(config.PEAK_MIN_DIST_SEC * config.AUDIO_SR / config.RMS_HOP)
    
    peak_indices, properties = scipy.signal.find_peaks(
        rms_norm, 
        prominence=prominence_threshold,
        distance=min_distance_frames
    )
    
    logger.info(f"Found {len(peak_indices)} candidate peaks")
    
    # Convert to AudioPeak objects
    peaks = []
    time_per_frame = duration / len(rms_envelope)
    
    for i, idx in enumerate(peak_indices):
        timestamp = idx * time_per_frame
        amplitude = properties['prominences'][i]
        
        # Calculate bandpass ratio if available
        bandpass_ratio = 0.0
        if bandpass_energy is not None and idx < len(bandpass_energy):
            total_energy = rms_envelope[idx] ** 2
            bp_energy = bandpass_energy[idx]
            if total_energy > 0:
                bandpass_ratio = bp_energy / total_energy
        
        peak = AudioPeak(timestamp, amplitude, bandpass_ratio)
        peaks.append(peak)
    
    # Sort by confidence score
    peaks.sort(key=lambda p: p.confidence, reverse=True)
    
    logger.info(f"Detected {len(peaks)} audio peaks")
    for peak in peaks[:5]:  # Log top 5 peaks
        logger.debug(f"Peak: {peak}")
    
    return peaks


def get_audio_peaks(video_path: str, use_bandpass: bool = True) -> List[AudioPeak]:
    """
    Main function to extract audio peaks from video.
    
    Args:
        video_path: Path to input video file
        use_bandpass: Whether to apply bandpass filtering
        
    Returns:
        List of detected AudioPeak objects sorted by confidence
    """
    logger.info(f"Analyzing audio from {video_path}")
    
    temp_audio = None
    try:
        # Extract audio
        temp_audio = extract_audio_fast(video_path)
        
        # Compute RMS envelope
        rms_envelope, duration = compute_rms_envelope(temp_audio)
        
        # Optionally apply bandpass filter
        bandpass_energy = None
        if use_bandpass:
            bandpass_energy = apply_bandpass_filter(temp_audio)
        
        # Detect peaks
        peaks = detect_peaks(rms_envelope, duration, bandpass_energy)
        
        logger.info(f"Audio analysis complete: {len(peaks)} peaks detected")
        return peaks
        
    finally:
        # Clean up temporary file
        if temp_audio:
            Path(temp_audio).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python audio_analysis.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    peaks = get_audio_peaks(video_path)
    
    print(f"\nDetected {len(peaks)} audio peaks:")
    for i, peak in enumerate(peaks[:10]):  # Show top 10
        print(f"{i+1:2d}. {peak}")