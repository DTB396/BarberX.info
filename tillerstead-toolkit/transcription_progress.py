#!/usr/bin/env python3
"""
REAL-TIME BWC TRANSCRIPTION PROGRESS MONITOR
Displays live progress bar and statistics
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys


class ProgressMonitor:
    def __init__(self, bwc_folder: str, output_folder: str):
        self.bwc_folder = Path(bwc_folder)
        self.output_folder = Path(output_folder)
        self.total_videos = 0
        self.start_time = None
        
    def count_total_videos(self):
        """Count total BWC videos to process"""
        videos = list(self.bwc_folder.glob("**/*.mp4"))
        self.total_videos = len(videos)
        return self.total_videos
    
    def count_completed_transcripts(self):
        """Count completed individual transcripts"""
        transcript_folder = self.output_folder / "individual_transcripts"
        if not transcript_folder.exists():
            return 0
        
        # Count .txt files (one per completed video)
        completed = len(list(transcript_folder.glob("*_transcript.txt")))
        return completed
    
    def get_latest_transcript(self):
        """Get the most recently created transcript"""
        transcript_folder = self.output_folder / "individual_transcripts"
        if not transcript_folder.exists():
            return None
        
        transcripts = list(transcript_folder.glob("*_transcript.txt"))
        if not transcripts:
            return None
        
        # Get most recent by modification time
        latest = max(transcripts, key=lambda p: p.stat().st_mtime)
        return latest.stem.replace('_transcript', '')
    
    def estimate_time_remaining(self, completed, total, elapsed_seconds):
        """Estimate time remaining based on current progress"""
        if completed == 0:
            return None
        
        avg_time_per_video = elapsed_seconds / completed
        remaining_videos = total - completed
        remaining_seconds = avg_time_per_video * remaining_videos
        
        return timedelta(seconds=int(remaining_seconds))
    
    def draw_progress_bar(self, completed, total, width=50):
        """Draw a visual progress bar"""
        if total == 0:
            return "[" + " " * width + "] 0%"
        
        percent = (completed / total) * 100
        filled = int((completed / total) * width)
        empty = width - filled
        
        bar = "[" + "█" * filled + "░" * empty + "]"
        return f"{bar} {percent:.1f}%"
    
    def format_time(self, td):
        """Format timedelta as HH:MM:SS"""
        if td is None:
            return "calculating..."
        
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_progress(self):
        """Display current progress"""
        completed = self.count_completed_transcripts()
        total = self.total_videos
        
        if self.start_time is None:
            self.start_time = datetime.now()
        
        elapsed = datetime.now() - self.start_time
        eta = self.estimate_time_remaining(completed, total, elapsed.total_seconds())
        latest_file = self.get_latest_transcript()
        
        # Clear screen for clean display
        self.clear_screen()
        
        # Header
        print("=" * 80)
        print("BWC TRANSCRIPTION PROGRESS MONITOR".center(80))
        print("=" * 80)
        print()
        
        # Progress bar
        print(f"  {self.draw_progress_bar(completed, total, 60)}")
        print()
        
        # Statistics
        print(f"  Videos Completed:  {completed} / {total}")
        print(f"  Videos Remaining:  {total - completed}")
        print()
        
        # Timing
        print(f"  Elapsed Time:      {self.format_time(elapsed)}")
        print(f"  Est. Remaining:    {self.format_time(eta)}")
        if eta:
            estimated_completion = datetime.now() + eta
            print(f"  Est. Completion:   {estimated_completion.strftime('%I:%M:%S %p')}")
        print()
        
        # Current file
        if latest_file:
            print(f"  Latest Completed:  {latest_file}")
        print()
        
        # Status
        if completed == total:
            print("  STATUS: ✅ TRANSCRIPTION COMPLETE!")
            print()
            print("  Next Steps:")
            print("    - Generating synchronized multi-POV transcript...")
            print("    - Creating certified legal transcript...")
            print("    - Finalizing court-ready documents...")
        elif completed == 0:
            print("  STATUS: 🔄 INITIALIZING WHISPER MODEL...")
            print()
            print("  This may take a few minutes on first run.")
        else:
            avg_time = elapsed.total_seconds() / completed
            print(f"  STATUS: 🎤 TRANSCRIBING... (~{int(avg_time)}s per video)")
            print()
            print(f"  Processing video {completed + 1}/{total}...")
        
        print()
        print("=" * 80)
        print("  Press Ctrl+C to exit monitor (transcription continues in background)")
        print("=" * 80)
    
    def monitor(self, refresh_interval=5):
        """Monitor transcription progress with auto-refresh"""
        print("Starting progress monitor...")
        print(f"Counting total videos in {self.bwc_folder}...")
        
        total = self.count_total_videos()
        print(f"Found {total} BWC videos to transcribe")
        print()
        print("Monitoring progress... (updating every {refresh_interval} seconds)")
        time.sleep(2)
        
        try:
            last_completed = -1
            while True:
                completed = self.count_completed_transcripts()
                
                # Only refresh display if progress changed or first time
                if completed != last_completed or last_completed == -1:
                    self.display_progress()
                    last_completed = completed
                
                # Check if done
                if completed >= total:
                    print("\n✅ All transcripts completed!")
                    print("\nChecking for synchronized and certified transcripts...")
                    time.sleep(2)
                    
                    # Check for final outputs
                    sync_file = self.output_folder / "synchronized" / "synchronized_transcript.txt"
                    cert_file = self.output_folder / "certified" / "CERTIFIED_TRANSCRIPT.txt"
                    
                    if sync_file.exists() and cert_file.exists():
                        print("\n🎊 COMPLETE BWC TRANSCRIPTION PACKAGE READY!")
                        print(f"\n📁 Location: {self.output_folder}")
                        print("\n✅ Individual transcripts: {completed}")
                        print(f"✅ Synchronized transcript: {sync_file}")
                        print(f"✅ Certified transcript: {cert_file}")
                        print("\n⚖️  READY FOR COURT FILING!")
                    else:
                        print("\n🔄 Waiting for synchronized and certified transcripts...")
                        print("    (This should complete in ~30 seconds)")
                    
                    break
                
                # Wait before next check
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("Progress monitor stopped by user")
            print("=" * 80)
            print(f"\nTranscription is still running in background!")
            print(f"Progress so far: {completed}/{total} videos completed")
            print(f"\nTo check status again, run:")
            print(f"  python transcription_progress.py")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor BWC transcription progress')
    parser.add_argument('--bwc-folder', 
                       default='C:/web-dev/github-repos/BarberX.info/tillerstead-toolkit/private-core-barber-cam',
                       help='Folder containing BWC videos')
    parser.add_argument('--output', 
                       default='./bwc_transcripts_certified',
                       help='Output folder for transcripts')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    monitor = ProgressMonitor(args.bwc_folder, args.output)
    monitor.monitor(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
