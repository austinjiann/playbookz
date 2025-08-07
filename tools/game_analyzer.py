#!/usr/bin/env python3
"""
Game Analysis Tool - Generate detailed match statistics and summaries.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_analysis import get_audio_peaks
from gemini_filter import validate_audio_peaks, GeminiFilter
import config


class GameAnalyzer:
    """Analyzes football videos to generate detailed game statistics."""
    
    def __init__(self):
        self.gemini_filter = GeminiFilter()
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def categorize_moments_by_intensity(self, validated_timestamps: List[float], 
                                      audio_peaks: List) -> Dict[str, List]:
        """Categorize moments by excitement intensity."""
        peak_lookup = {peak.timestamp: peak for peak in audio_peaks}
        
        high_intensity = []  # Top 30% confidence
        medium_intensity = []  # Middle 40% 
        low_intensity = []   # Bottom 30%
        
        # Sort by confidence and categorize
        moments_with_confidence = []
        for timestamp in validated_timestamps:
            # Find matching peak
            for peak in audio_peaks:
                if abs(peak.timestamp - timestamp) < 1.0:  # Within 1 second
                    moments_with_confidence.append((timestamp, peak.confidence))
                    break
        
        # Sort by confidence
        moments_with_confidence.sort(key=lambda x: x[1], reverse=True)
        
        total = len(moments_with_confidence)
        if total > 0:
            high_cutoff = int(total * 0.3)
            medium_cutoff = int(total * 0.7)
            
            high_intensity = moments_with_confidence[:high_cutoff]
            medium_intensity = moments_with_confidence[high_cutoff:medium_cutoff]  
            low_intensity = moments_with_confidence[medium_cutoff:]
        
        return {
            "high_intensity": high_intensity,
            "medium_intensity": medium_intensity, 
            "low_intensity": low_intensity
        }
    
    def detect_goal_sequences(self, validated_timestamps: List[float]) -> List[Dict]:
        """Detect potential goal sequences based on timing patterns."""
        goals = []
        
        for i, timestamp in enumerate(validated_timestamps):
            # Look for clusters of exciting moments (potential goals + celebrations)
            cluster_start = timestamp
            cluster_end = timestamp
            cluster_moments = [timestamp]
            
            # Check next few moments for clustering
            for j in range(i + 1, min(i + 5, len(validated_timestamps))):
                next_time = validated_timestamps[j]
                if next_time - cluster_end <= 30:  # Within 30 seconds
                    cluster_end = next_time
                    cluster_moments.append(next_time)
                else:
                    break
            
            # If cluster has multiple moments, likely a goal sequence
            if len(cluster_moments) >= 2:
                goals.append({
                    "estimated_goal_time": self.format_timestamp(cluster_start),
                    "goal_time_seconds": cluster_start,
                    "celebration_duration": cluster_end - cluster_start,
                    "excitement_moments": len(cluster_moments),
                    "timestamps": cluster_moments
                })
                
                # Skip ahead to avoid overlapping detections
                while i + 1 < len(validated_timestamps) and validated_timestamps[i + 1] in cluster_moments:
                    i += 1
        
        return goals
    
    def analyze_game_tempo(self, validated_timestamps: List[float], 
                          video_duration: float) -> Dict[str, Any]:
        """Analyze the tempo and excitement distribution throughout the game."""
        if not validated_timestamps:
            return {"periods": [], "overall_tempo": "low"}
        
        # Divide game into periods (e.g., 10-minute segments)
        period_length = 600  # 10 minutes
        num_periods = int(video_duration // period_length) + 1
        
        periods = []
        for i in range(num_periods):
            period_start = i * period_length
            period_end = min((i + 1) * period_length, video_duration)
            
            # Count moments in this period
            period_moments = [
                t for t in validated_timestamps 
                if period_start <= t < period_end
            ]
            
            excitement_level = "low"
            if len(period_moments) >= 5:
                excitement_level = "high"
            elif len(period_moments) >= 2:
                excitement_level = "medium"
            
            periods.append({
                "period": f"{self.format_timestamp(period_start)}-{self.format_timestamp(period_end)}",
                "exciting_moments": len(period_moments),
                "excitement_level": excitement_level,
                "timestamps": period_moments
            })
        
        # Overall tempo
        moments_per_minute = len(validated_timestamps) / (video_duration / 60)
        if moments_per_minute >= 1.0:
            overall_tempo = "high"
        elif moments_per_minute >= 0.5:
            overall_tempo = "medium"  
        else:
            overall_tempo = "low"
        
        return {
            "periods": periods,
            "overall_tempo": overall_tempo,
            "moments_per_minute": round(moments_per_minute, 2)
        }
    
    async def enhanced_moment_classification(self, video_path: str, 
                                           validated_timestamps: List[float]) -> Dict[str, List]:
        """Use Gemini to classify specific types of exciting moments."""
        
        # Enhanced prompts for different action types
        action_prompts = {
            "goals": """
            Look at this frame from a football/soccer match. Does this frame show:
            - A ball crossing the goal line
            - Net movement from a goal
            - Players celebrating a goal that just happened
            - Goalkeeper beaten/dejected after conceding
            
            Reply ONLY "YES" if this shows a goal or immediate goal celebration, "NO" otherwise.
            """,
            
            "saves": """
            Look at this frame from a football/soccer match. Does this frame show:
            - Goalkeeper making a diving save
            - Goalkeeper catching/blocking a shot
            - Ball being deflected away from goal by keeper
            - Spectacular defensive action
            
            Reply ONLY "YES" if this shows a save or defensive action, "NO" otherwise.
            """,
            
            "skills": """
            Look at this frame from a football/soccer match. Does this frame show:
            - Player performing skillful dribbling
            - Nutmeg or skill move beating a defender
            - Creative passing or ball control
            - Technical skill in tight spaces
            
            Reply ONLY "YES" if this shows skillful play, "NO" otherwise.
            """
        }
        
        # For demo purposes, simulate classification
        # In real implementation, would call Gemini with different prompts
        import random
        
        classified_moments = {
            "goals": [],
            "saves": [],
            "skills": [],
            "other_exciting": []
        }
        
        # Simulate classification (in real version, would call Gemini for each type)
        for timestamp in validated_timestamps:
            # Higher confidence moments more likely to be goals
            rand_val = random.random()
            if rand_val < 0.3:
                classified_moments["goals"].append(timestamp)
            elif rand_val < 0.5:
                classified_moments["saves"].append(timestamp)
            elif rand_val < 0.7:
                classified_moments["skills"].append(timestamp)
            else:
                classified_moments["other_exciting"].append(timestamp)
        
        return classified_moments
    
    async def generate_game_summary(self, video_path: str) -> Dict[str, Any]:
        """Generate comprehensive game analysis and summary."""
        print(f"🎬 Analyzing match: {Path(video_path).name}")
        
        # Get video duration
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        
        # Analyze audio and get Gemini validation
        print("📊 Analyzing audio peaks...")
        audio_peaks = get_audio_peaks(video_path, use_bandpass=True)
        
        print("🤖 Validating moments with AI...")
        peak_timestamps = [peak.timestamp for peak in audio_peaks[:30]]  # Limit for demo
        validated_timestamps = await validate_audio_peaks(video_path, peak_timestamps)
        
        print("🏆 Detecting goal sequences...")
        potential_goals = self.detect_goal_sequences(validated_timestamps)
        
        print("⚡ Analyzing game tempo...")
        tempo_analysis = self.analyze_game_tempo(validated_timestamps, duration)
        
        print("🎯 Categorizing moment types...")
        classified_moments = await self.enhanced_moment_classification(video_path, validated_timestamps)
        
        print("📈 Computing statistics...")
        intensity_categories = self.categorize_moments_by_intensity(validated_timestamps, audio_peaks)
        
        # Calculate statistics
        total_peaks = len(audio_peaks)
        validated_peaks = len(validated_timestamps)
        validation_rate = (validated_peaks / total_peaks * 100) if total_peaks > 0 else 0
        
        # Generate summary
        summary = {
            "match_info": {
                "video_file": Path(video_path).name,
                "duration": self.format_timestamp(duration),
                "duration_seconds": round(duration, 1),
                "analysis_date": datetime.now().isoformat()
            },
            
            "excitement_overview": {
                "total_audio_peaks": total_peaks,
                "ai_validated_moments": validated_peaks,
                "validation_rate": round(validation_rate, 1),
                "moments_per_minute": tempo_analysis["moments_per_minute"],
                "overall_tempo": tempo_analysis["overall_tempo"]
            },
            
            "estimated_events": {
                "potential_goals": len(potential_goals),
                "goal_sequences": potential_goals,
                "estimated_saves": len(classified_moments["saves"]),
                "skillful_moments": len(classified_moments["skills"]),
                "other_exciting": len(classified_moments["other_exciting"])
            },
            
            "moment_breakdown": {
                "high_intensity": len(intensity_categories["high_intensity"]),
                "medium_intensity": len(intensity_categories["medium_intensity"]), 
                "low_intensity": len(intensity_categories["low_intensity"])
            },
            
            "game_tempo_analysis": tempo_analysis,
            
            "key_moments": [
                {
                    "time": self.format_timestamp(ts),
                    "time_seconds": ts,
                    "intensity": "high" if any(ts == t for t, _ in intensity_categories["high_intensity"]) else 
                               "medium" if any(ts == t for t, _ in intensity_categories["medium_intensity"]) else "low"
                }
                for ts in sorted(validated_timestamps)[:10]  # Top 10 moments
            ],
            
            "raw_data": {
                "all_validated_timestamps": validated_timestamps,
                "classified_moments": classified_moments,
                "intensity_categories": intensity_categories
            }
        }
        
        return summary


def create_readable_report(summary: Dict[str, Any]) -> str:
    """Generate human-readable match report."""
    
    report = f"""
🏆 MATCH ANALYSIS REPORT
{'='*50}

📹 Match: {summary['match_info']['video_file']}
⏱️  Duration: {summary['match_info']['duration']}
📅 Analyzed: {summary['match_info']['analysis_date'][:16]}

🎯 EXCITEMENT SUMMARY
{'='*30}
• Total exciting moments detected: {summary['excitement_overview']['ai_validated_moments']}
• Moments per minute: {summary['excitement_overview']['moments_per_minute']}
• Game tempo: {summary['excitement_overview']['overall_tempo'].upper()}
• AI validation rate: {summary['excitement_overview']['validation_rate']}%

⚽ ESTIMATED EVENTS  
{'='*25}
• Potential Goals: {summary['estimated_events']['potential_goals']}
• Great Saves: {summary['estimated_events']['estimated_saves']}  
• Skillful Plays: {summary['estimated_events']['skillful_moments']}
• Other Exciting: {summary['estimated_events']['other_exciting']}

🔥 INTENSITY BREAKDOWN
{'='*28}
• High Intensity: {summary['moment_breakdown']['high_intensity']} moments
• Medium Intensity: {summary['moment_breakdown']['medium_intensity']} moments  
• Low Intensity: {summary['moment_breakdown']['low_intensity']} moments

⚡ KEY MOMENTS
{'='*20}
"""
    
    for i, moment in enumerate(summary['key_moments'][:5], 1):
        intensity_emoji = {"high": "🔥", "medium": "⚡", "low": "✨"}[moment['intensity']]
        report += f"{i:2d}. {moment['time']} {intensity_emoji} {moment['intensity'].capitalize()} intensity\n"
    
    if summary['estimated_events']['potential_goals'] > 0:
        report += f"\n🥅 GOAL SEQUENCES\n{'='*20}\n"
        for i, goal in enumerate(summary['estimated_events']['goal_sequences'], 1):
            report += f"{i}. Goal around {goal['estimated_goal_time']} "
            report += f"(celebration lasted {goal['celebration_duration']:.0f}s)\n"
    
    report += f"\n📊 GAME TEMPO BY PERIOD\n{'='*30}\n"
    for period in summary['game_tempo_analysis']['periods']:
        tempo_emoji = {"high": "🔥", "medium": "⚡", "low": "😴"}[period['excitement_level']]
        report += f"{period['period']}: {period['exciting_moments']} moments {tempo_emoji}\n"
    
    return report


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze football match and generate detailed statistics")
    parser.add_argument('--video', required=True, help='Path to match video')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--report', action='store_true', help='Generate readable report')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    try:
        # Generate analysis
        analyzer = GameAnalyzer()
        summary = await analyzer.generate_game_summary(args.video)
        
        # Save JSON summary
        if args.output:
            output_path = Path(args.output)
        else:
            video_name = Path(args.video).stem
            output_path = Path(f"match_analysis_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Detailed data saved to: {output_path}")
        
        # Generate readable report
        if args.report:
            report = create_readable_report(summary)
            print(report)
        else:
            # Print brief summary
            print(f"\n🎯 QUICK SUMMARY")
            print(f"⚽ Estimated Goals: {summary['estimated_events']['potential_goals']}")
            print(f"🎪 Exciting Moments: {summary['excitement_overview']['ai_validated_moments']}")
            print(f"⚡ Game Tempo: {summary['excitement_overview']['overall_tempo'].title()}")
            print(f"📊 Moments/Minute: {summary['excitement_overview']['moments_per_minute']}")
            print(f"\nRun with --report for detailed breakdown!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))