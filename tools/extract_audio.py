#!/usr/bin/env python3
"""
Debug utility for audio peak detection visualization.
Extracts audio, computes RMS envelope, and generates plots with peak markers.
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_analysis import get_audio_peaks, compute_rms_envelope, extract_audio_fast
import config


def plot_audio_analysis(video_path: str, output_dir: str = "debug_output"):
    """
    Create visualization plots for audio analysis debugging.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save plots
    """
    print(f"Analyzing audio from {video_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract audio and compute RMS
    temp_audio = extract_audio_fast(video_path)
    try:
        rms_envelope, duration = compute_rms_envelope(temp_audio)
        
        # Get audio peaks
        peaks = get_audio_peaks(video_path, use_bandpass=True)
        
        # Create time axis
        time_axis = np.linspace(0, duration, len(rms_envelope))
        
        # Plot RMS envelope with peaks
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: RMS Envelope
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, rms_envelope, 'b-', alpha=0.7, label='RMS Envelope')
        
        # Mark peaks
        peak_times = [peak.timestamp for peak in peaks[:10]]  # Top 10 peaks
        peak_amplitudes = []
        
        for peak_time in peak_times:
            # Find closest RMS sample
            idx = int(peak_time * len(rms_envelope) / duration)
            if 0 <= idx < len(rms_envelope):
                peak_amplitudes.append(rms_envelope[idx])
            else:
                peak_amplitudes.append(0)
        
        plt.scatter(peak_times, peak_amplitudes, c='red', s=100, 
                   label=f'Top {len(peak_times)} Peaks', zorder=5)
        
        # Add threshold line
        max_rms = np.max(rms_envelope)
        threshold = max_rms * config.PEAK_PROMINENCE
        plt.axhline(y=threshold, color='orange', linestyle='--', 
                   label=f'Threshold ({config.PEAK_PROMINENCE:.2f})')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('RMS Amplitude')
        plt.title(f'Audio Analysis: {Path(video_path).name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Peak Details
        plt.subplot(2, 1, 2)
        if peaks:
            confidences = [peak.confidence for peak in peaks[:20]]
            timestamps = [peak.timestamp for peak in peaks[:20]]
            
            bars = plt.bar(range(len(confidences)), confidences, 
                          color=['red' if c >= 0.7 else 'orange' if c >= 0.5 else 'gray' 
                                for c in confidences])
            
            plt.xlabel('Peak Index (sorted by confidence)')
            plt.ylabel('Confidence Score')
            plt.title('Peak Confidence Scores')
            plt.grid(True, alpha=0.3)
            
            # Add timestamps as labels
            for i, (bar, timestamp) in enumerate(zip(bars, timestamps)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{timestamp:.1f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"audio_analysis_{Path(video_path).stem}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        
        # Save peak data
        data_path = output_path / f"peaks_{Path(video_path).stem}.txt"
        with open(data_path, 'w') as f:
            f.write("Audio Peak Analysis Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Total peaks found: {len(peaks)}\n")
            f.write(f"Peak prominence threshold: {config.PEAK_PROMINENCE}\n")
            f.write(f"Minimum peak distance: {config.PEAK_MIN_DIST_SEC}s\n")
            f.write("\nTop 20 peaks:\n")
            f.write("-" * 60 + "\n")
            f.write("Rank | Time (s) | Confidence | Amplitude | Bandpass\n")
            f.write("-" * 60 + "\n")
            
            for i, peak in enumerate(peaks[:20]):
                f.write(f"{i+1:4d} | {peak.timestamp:8.1f} | {peak.confidence:10.3f} | "
                       f"{peak.amplitude:9.3f} | {peak.bandpass_ratio:8.3f}\n")
        
        print(f"Peak data saved to {data_path}")
        
        # Display summary
        print(f"\nAnalysis Summary:")
        print(f"- Video duration: {duration:.1f}s")
        print(f"- Total peaks detected: {len(peaks)}")
        print(f"- High-confidence peaks (>0.7): {sum(1 for p in peaks if p.confidence > 0.7)}")
        print(f"- Medium-confidence peaks (0.5-0.7): {sum(1 for p in peaks if 0.5 < p.confidence <= 0.7)}")
        
        if peaks:
            print(f"- Best peak: {peaks[0].timestamp:.1f}s (conf: {peaks[0].confidence:.3f})")
        
        plt.show()
        
    finally:
        # Cleanup
        Path(temp_audio).unlink(missing_ok=True)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Debug audio peak detection")
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('--output', default='debug_output', 
                       help='Output directory for plots and data')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Skip showing plot (just save to file)')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    try:
        plot_audio_analysis(args.video, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())