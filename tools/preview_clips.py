#!/usr/bin/env python3
"""
Preview tool for debugging clip selection before rendering.
Generates HTML preview with frame thumbnails and timestamps.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import base64

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_analysis import get_audio_peaks
from gemini_filter import validate_audio_peaks
import config


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> Tuple[bool, bytes]:
    """Extract a single frame at given timestamp as JPEG bytes."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FALLBACK
        
        # Seek to the desired timestamp
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, b""
        
        # Resize frame for preview (max 400px width)
        height, width = frame.shape[:2]
        if width > 400:
            scale = 400 / width
            new_width = 400
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            return True, buffer.tobytes()
        
        return False, b""
        
    except Exception as e:
        print(f"Error extracting frame at {timestamp}s: {e}")
        return False, b""


def generate_html_preview(video_path: str, audio_peaks: List, validated_timestamps: List[float], 
                         output_html: str) -> str:
    """Generate HTML preview with frame thumbnails and analysis data."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Football Highlights Preview</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
                background-color: #f5f5f7;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .header {{
                border-bottom: 2px solid #007AFF;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #007AFF;
            }}
            .frames-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }}
            .frame-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background: white;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .frame-card.validated {{
                border-color: #34C759;
                box-shadow: 0 2px 8px rgba(52, 199, 89, 0.3);
            }}
            .frame-card.rejected {{
                border-color: #FF3B30;
                box-shadow: 0 2px 8px rgba(255, 59, 48, 0.2);
            }}
            .frame-image {{
                width: 100%;
                height: auto;
                border-radius: 6px;
                margin-bottom: 10px;
            }}
            .frame-info {{
                font-size: 0.9em;
            }}
            .timestamp {{
                font-weight: bold;
                color: #007AFF;
                font-size: 1.1em;
            }}
            .confidence {{
                color: #666;
            }}
            .status {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 0.8em;
                margin-top: 8px;
            }}
            .validated {{
                background-color: #34C759;
                color: white;
            }}
            .rejected {{
                background-color: #FF3B30;
                color: white;
            }}
            .fallback {{
                background-color: #FF9500;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎬 Football Highlights Preview</h1>
                <p><strong>Video:</strong> {Path(video_path).name}</p>
                <p><strong>Analysis Date:</strong> {config.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if hasattr(config, 'datetime') else 'N/A'}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(audio_peaks)}</div>
                    <div>Audio Peaks Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(validated_timestamps)}</div>
                    <div>Gemini Validated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(validated_timestamps) / max(len(audio_peaks), 1) * 100:.1f}%</div>
                    <div>Validation Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(validated_timestamps) * (config.PRE_SEC + config.POST_SEC):.0f}s</div>
                    <div>Total Highlight Duration</div>
                </div>
            </div>
            
            <h2>🔍 Frame Analysis</h2>
            <div class="frames-grid">
    """
    
    # Process each audio peak
    for i, peak in enumerate(audio_peaks[:30]):  # Limit to 30 for performance
        timestamp = peak.timestamp
        is_validated = timestamp in validated_timestamps
        
        # Extract frame
        success, frame_bytes = extract_frame_at_timestamp(video_path, timestamp)
        
        if success:
            # Convert to base64 for HTML embedding
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Determine peak type
            peak_type = "Fallback Sample" if peak.confidence < 0.2 else "Audio Peak"
            
            html_content += f"""
                <div class="frame-card {'validated' if is_validated else 'rejected'}">
                    <img src="data:image/jpeg;base64,{frame_b64}" class="frame-image" alt="Frame at {timestamp:.1f}s">
                    <div class="frame-info">
                        <div class="timestamp">⏰ {timestamp:.1f}s</div>
                        <div class="confidence">📊 Confidence: {peak.confidence:.3f}</div>
                        <div>🎵 Amplitude: {peak.amplitude:.3f}</div>
                        <div>📻 Bandpass: {peak.bandpass_ratio:.3f}</div>
                        <div>🎯 Type: {peak_type}</div>
                        <span class="status {'validated' if is_validated else 'rejected'}">
                            {'✅ VALIDATED' if is_validated else '❌ REJECTED'}
                        </span>
                    </div>
                </div>
            """
    
    html_content += """
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em;">
                <p><strong>How to interpret this preview:</strong></p>
                <ul>
                    <li>🟢 <strong>Green borders:</strong> Frames validated by Gemini Vision as exciting moments</li>
                    <li>🔴 <strong>Red borders:</strong> Frames rejected by Gemini Vision as non-exciting</li>
                    <li>🟠 <strong>Fallback samples:</strong> Low-confidence frames sampled when no audio peaks found</li>
                    <li>Only validated frames will be included in the final highlights video</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    return output_html


async def preview_clips(video_path: str, output_dir: str = "debug_output") -> str:
    """
    Generate preview of selected clips with frame thumbnails.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save preview files
        
    Returns:
        Path to generated HTML preview
    """
    print(f"🎬 Generating clip preview for {Path(video_path).name}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Analyze audio peaks
    print("📊 Analyzing audio peaks...")
    audio_peaks = get_audio_peaks(video_path, use_bandpass=True)
    
    if not audio_peaks:
        print("⚠️ No audio peaks detected!")
        return ""
    
    print(f"🎵 Found {len(audio_peaks)} audio peaks")
    
    # Validate with Gemini
    print("🤖 Validating frames with Gemini Vision...")
    peak_timestamps = [peak.timestamp for peak in audio_peaks]
    validated_timestamps = await validate_audio_peaks(video_path, peak_timestamps)
    
    print(f"✅ Validated {len(validated_timestamps)} out of {len(peak_timestamps)} frames")
    
    # Generate HTML preview
    video_name = Path(video_path).stem
    html_path = output_path / f"preview_{video_name}.html"
    json_path = output_path / f"timestamps_{video_name}.json"
    
    print("🌐 Generating HTML preview...")
    generate_html_preview(video_path, audio_peaks, validated_timestamps, str(html_path))
    
    # Save timestamps to JSON
    timestamps_data = {
        "video_path": video_path,
        "analysis_timestamp": "2025-08-06T15:00:00",  # Would use datetime.now().isoformat()
        "audio_peaks": [
            {
                "timestamp": peak.timestamp,
                "confidence": peak.confidence,
                "amplitude": peak.amplitude,
                "bandpass_ratio": peak.bandpass_ratio
            } for peak in audio_peaks
        ],
        "validated_timestamps": validated_timestamps,
        "config": {
            "PEAK_PROMINENCE": config.PEAK_PROMINENCE,
            "GEMINI_POS_THRESHOLD": config.GEMINI_POS_THRESHOLD,
            "PRE_SEC": config.PRE_SEC,
            "POST_SEC": config.POST_SEC,
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(timestamps_data, f, indent=2)
    
    print(f"📁 Preview saved to: {html_path}")
    print(f"📄 Timestamps saved to: {json_path}")
    
    return str(html_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Preview football highlight clips")
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='debug_output', help='Output directory')
    parser.add_argument('--open', action='store_true', help='Open preview in browser')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    try:
        # Validate configuration
        config.validate_config()
        
        # Generate preview
        import asyncio
        html_path = asyncio.run(preview_clips(args.video, args.output))
        
        if html_path:
            print(f"\n🎉 Preview generated successfully!")
            print(f"📂 Output: {html_path}")
            
            if args.open:
                import webbrowser
                webbrowser.open(f"file://{Path(html_path).absolute()}")
                print("🌐 Preview opened in browser")
        else:
            print("❌ Failed to generate preview")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())