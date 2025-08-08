"""
Test suite for the audio + Gemini pipeline.
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import cv2

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from audio_analysis import AudioPeak, detect_peaks, get_audio_peaks
from gemini_filter import GeminiFilter, validate_audio_peaks
from gen_highlights import HighlightClip, merge_overlapping_clips, generate_highlights


class TestAudioPeak:
    """Test AudioPeak class functionality."""
    
    def test_peak_creation(self):
        """Test basic AudioPeak creation and properties."""
        peak = AudioPeak(timestamp=10.5, amplitude=0.8, bandpass_ratio=0.3)
        
        assert peak.timestamp == 10.5
        assert peak.amplitude == 0.8
        assert peak.bandpass_ratio == 0.3
        assert 0 <= peak.confidence <= 1
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # High amplitude, high bandpass ratio
        peak1 = AudioPeak(10.0, 0.9, 0.5)
        
        # Low amplitude, low bandpass ratio  
        peak2 = AudioPeak(20.0, 0.1, 0.05)
        
        assert peak1.confidence > peak2.confidence
        assert peak1.confidence > 0.5
    
    def test_peak_repr(self):
        """Test string representation."""
        peak = AudioPeak(10.5, 0.8, 0.3)
        repr_str = repr(peak)
        
        assert "10.5s" in repr_str
        assert "0.800" in repr_str
        assert "AudioPeak" in repr_str


class TestDetectPeaks:
    """Test peak detection functionality."""
    
    def test_detect_peaks_empty_envelope(self):
        """Test with empty RMS envelope."""
        empty_rms = np.array([])
        peaks = detect_peaks(empty_rms, 10.0)
        assert peaks == []
    
    def test_detect_peaks_zeros(self):
        """Test with all-zero RMS envelope."""
        zero_rms = np.zeros(100)
        peaks = detect_peaks(zero_rms, 10.0)
        assert peaks == []
    
    def test_detect_peaks_single_peak(self):
        """Test detection of single prominent peak."""
        # Create simple signal with one clear peak
        t = np.linspace(0, 10, 1000)
        signal = np.exp(-0.5 * ((t - 5) / 1)**2)  # Gaussian peak at t=5
        
        peaks = detect_peaks(signal, 10.0)
        
        assert len(peaks) >= 1
        # Check that detected peak is near the actual peak
        assert abs(peaks[0].timestamp - 5.0) < 1.0
    
    def test_detect_peaks_with_bandpass(self):
        """Test peak detection with bandpass energy."""
        rms = np.array([0.1, 0.5, 0.8, 0.3, 0.7, 0.2])
        bandpass = np.array([0.02, 0.15, 0.25, 0.1, 0.2, 0.05])
        duration = 6.0
        
        peaks = detect_peaks(rms, duration, bandpass)
        
        # Should find peaks with bandpass ratios
        for peak in peaks:
            assert peak.bandpass_ratio >= 0


class TestHighlightClip:
    """Test HighlightClip class functionality."""
    
    def test_clip_creation(self):
        """Test basic clip creation."""
        clip = HighlightClip(10.0, 25.0, 'audio', confidence=0.8)
        
        assert clip.start_time == 10.0
        assert clip.end_time == 25.0
        assert clip.duration == 15.0
        assert clip.source == 'audio'
        assert clip.confidence == 0.8
    
    def test_negative_start_time(self):
        """Test handling of negative start time."""
        clip = HighlightClip(-5.0, 10.0, 'audio')
        assert clip.start_time == 0.0  # Should be clamped to 0
        assert clip.end_time == 10.0
    
    def test_clip_overlap_detection(self):
        """Test overlap detection between clips."""
        clip1 = HighlightClip(10.0, 25.0, 'audio')
        clip2 = HighlightClip(20.0, 35.0, 'scoreboard')  # Overlaps
        clip3 = HighlightClip(40.0, 55.0, 'audio')      # No overlap
        
        assert clip1.overlaps_with(clip2)
        assert not clip1.overlaps_with(clip3)
        assert clip2.overlaps_with(clip1)  # Symmetric
    
    def test_clip_merging(self):
        """Test merging of overlapping clips."""
        clip1 = HighlightClip(10.0, 25.0, 'audio', 0.7)
        clip2 = HighlightClip(20.0, 35.0, 'scoreboard', 0.9)
        
        merged = clip1.merge_with(clip2)
        
        assert merged.start_time == 10.0  # Min start
        assert merged.end_time == 35.0    # Max end
        assert merged.confidence == 0.9   # Max confidence
        assert 'audio' in merged.source and 'scoreboard' in merged.source


class TestMergeOverlappingClips:
    """Test clip merging functionality."""
    
    def test_merge_empty_list(self):
        """Test merging empty clip list."""
        result = merge_overlapping_clips([])
        assert result == []
    
    def test_merge_single_clip(self):
        """Test merging single clip."""
        clip = HighlightClip(10.0, 25.0, 'audio')
        result = merge_overlapping_clips([clip])
        
        assert len(result) == 1
        assert result[0] == clip
    
    def test_merge_no_overlaps(self):
        """Test merging clips with no overlaps."""
        clips = [
            HighlightClip(10.0, 20.0, 'audio'),
            HighlightClip(30.0, 40.0, 'scoreboard'),
            HighlightClip(50.0, 60.0, 'audio')
        ]
        
        result = merge_overlapping_clips(clips)
        assert len(result) == 3
    
    def test_merge_with_overlaps(self):
        """Test merging clips with overlaps."""
        clips = [
            HighlightClip(10.0, 25.0, 'audio'),
            HighlightClip(20.0, 35.0, 'scoreboard'),  # Overlaps with first
            HighlightClip(50.0, 60.0, 'audio')        # No overlap
        ]
        
        result = merge_overlapping_clips(clips)
        
        assert len(result) == 2  # First two should be merged
        assert result[0].start_time == 10.0
        assert result[0].end_time == 35.0
        assert result[1].start_time == 50.0


class TestGeminiFilter:
    """Test Gemini Vision API integration."""
    
    @pytest.fixture
    def mock_filter(self):
        """Create mock GeminiFilter for testing."""
        with patch('config.GEMINI_API_KEY', 'test-key'):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    filter_instance = GeminiFilter()
                    return filter_instance
    
    def test_frame_hashing(self, mock_filter):
        """Test frame hashing for caching."""
        # Create test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 255  # White square
        
        hash1 = mock_filter._hash_frame(frame)
        hash2 = mock_filter._hash_frame(frame)
        
        assert hash1 == hash2  # Same frame should have same hash
        assert len(hash1) == 40  # SHA-1 hex length
    
    def test_frame_encoding(self, mock_filter):
        """Test frame encoding to base64."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        encoded = mock_filter._encode_frame(frame)
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_filter):
        """Test batch frame processing."""
        # Mock the actual Gemini call
        mock_filter.is_exciting_frame = AsyncMock(return_value=True)
        
        # Create test frames
        frames_with_timestamps = [
            (10.0, np.zeros((100, 100, 3), dtype=np.uint8)),
            (20.0, np.zeros((100, 100, 3), dtype=np.uint8)),
            (30.0, np.zeros((100, 100, 3), dtype=np.uint8))
        ]
        
        results = await mock_filter.filter_frames_batch(frames_with_timestamps)
        
        assert len(results) == 3
        assert all(isinstance(ts, float) for ts in results)


@pytest.mark.asyncio  
async def test_validate_audio_peaks():
    """Test audio peak validation with mocked Gemini."""
    with patch('gemini_filter.get_gemini_filter') as mock_get_filter:
        # Mock the filter instance
        mock_filter = AsyncMock()
        mock_filter.filter_frames_batch = AsyncMock(return_value=[10.5, 25.3])
        mock_get_filter.return_value = mock_filter
        
        # Mock video capture
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = Mock()
            mock_cap_instance.get.return_value = 30.0  # FPS
            mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
            mock_cap.return_value = mock_cap_instance
            
            # Test validation
            peak_timestamps = [10.5, 20.0, 25.3, 40.0]
            validated = await validate_audio_peaks('test_video.mp4', peak_timestamps)
            
            assert len(validated) == 2
            assert 10.5 in validated
            assert 25.3 in validated


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    async def test_generate_highlights_scoreboard_mode(self):
        """Test highlights generation in scoreboard mode."""
        with patch('gen_highlights.extract_scoreboard_clips') as mock_scoreboard:
            # Mock scoreboard clips
            mock_clips = [
                HighlightClip(10.0, 22.0, 'scoreboard', 0.9),
                HighlightClip(45.0, 57.0, 'scoreboard', 0.8)
            ]
            mock_scoreboard.return_value = mock_clips
            
            with patch('gen_highlights.fast_concat') as mock_concat:
                mock_concat.return_value = 'output.mp4'
                
                with patch('pathlib.Path.exists', return_value=True):
                    result = await generate_highlights('test_video.mp4', mode='scoreboard')
                    
                    assert result == 'output.mp4'
                    mock_scoreboard.assert_called_once()
                    mock_concat.assert_called_once()
    
    @pytest.mark.asyncio  
    async def test_generate_highlights_audio_mode(self):
        """Test highlights generation in audio mode."""
        with patch('gen_highlights.extract_audio_clips') as mock_audio:
            # Mock audio clips
            mock_clips = [
                HighlightClip(15.0, 27.0, 'audio', 0.7),
                HighlightClip(50.0, 62.0, 'audio', 0.8)
            ]
            mock_audio.return_value = mock_clips
            
            with patch('gen_highlights.fast_concat') as mock_concat:
                mock_concat.return_value = 'output.mp4'
                
                with patch('pathlib.Path.exists', return_value=True):
                    result = await generate_highlights('test_video.mp4', mode='audio')
                    
                    assert result == 'output.mp4'
                    mock_audio.assert_called_once()
                    mock_concat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_highlights_hybrid_mode(self):
        """Test highlights generation in hybrid mode."""
        with patch('gen_highlights.extract_scoreboard_clips') as mock_scoreboard:
            with patch('gen_highlights.extract_audio_clips') as mock_audio:
                # Mock clips from both sources
                mock_scoreboard.return_value = [HighlightClip(10.0, 22.0, 'scoreboard', 0.9)]
                mock_audio.return_value = [HighlightClip(15.0, 27.0, 'audio', 0.7)]  # Overlapping
                
                with patch('gen_highlights.fast_concat') as mock_concat:
                    mock_concat.return_value = 'output.mp4'
                    
                    with patch('pathlib.Path.exists', return_value=True):
                        result = await generate_highlights('test_video.mp4', mode='hybrid')
                        
                        assert result == 'output.mp4'
                        mock_scoreboard.assert_called_once()
                        mock_audio.assert_called_once()
                        mock_concat.assert_called_once()


# Fixtures and test data
@pytest.fixture
def sample_audio_peaks():
    """Sample audio peaks for testing."""
    return [
        AudioPeak(10.5, 0.8, 0.3),
        AudioPeak(25.3, 0.9, 0.4), 
        AudioPeak(41.7, 0.6, 0.2),
        AudioPeak(58.2, 0.7, 0.35)
    ]


@pytest.fixture  
def sample_clips():
    """Sample highlight clips for testing."""
    return [
        HighlightClip(10.0, 22.0, 'scoreboard', 0.9),
        HighlightClip(15.0, 27.0, 'audio', 0.7),      # Overlaps with first
        HighlightClip(45.0, 57.0, 'scoreboard', 0.8),
        HighlightClip(70.0, 82.0, 'audio', 0.6)
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])