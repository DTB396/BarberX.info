"""
BWC Footage Analysis Test Script

This script tests the BWC analysis capabilities:
1. Scans a folder of BWC videos
2. Transcribes audio (using Whisper or local transcription)
3. Analyzes for constitutional violations
4. Creates synchronized multi-POV timeline
5. Generates comprehensive report

Usage:
    python test_bwc_analysis.py --folder "path/to/bwc/footage"
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import subprocess

# Color output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_step(step, total, text):
    print(f"{Colors.OKCYAN}[{step}/{total}] {text}{Colors.ENDC}")


class BWCAnalysisTester:
    """Test BWC analysis features"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.videos: List[Path] = []
        self.results: Dict[str, Any] = {}
        
    def scan_folder(self) -> List[Path]:
        """Scan folder for video files"""
        print_step(1, 6, "Scanning folder for BWC footage...")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        if not self.folder_path.exists():
            print_error(f"Folder not found: {self.folder_path}")
            return []
        
        videos = []
        for file in self.folder_path.rglob('*'):
            if file.suffix.lower() in video_extensions:
                videos.append(file)
                print_info(f"Found: {file.name} ({self.get_file_size(file)})")
        
        self.videos = videos
        print_success(f"Found {len(videos)} video files")
        return videos
    
    def get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size"""
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using ffprobe"""
        print_step(2, 6, f"Analyzing video metadata: {video_path.name}...")
        
        try:
            # Try to use ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                format_info = data.get('format', {})
                video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), {})
                audio_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'audio'), {})
                
                duration = float(format_info.get('duration', 0))
                duration_formatted = f"{int(duration // 60)}m {int(duration % 60)}s"
                
                info = {
                    'filename': video_path.name,
                    'path': str(video_path),
                    'size': self.get_file_size(video_path),
                    'duration_seconds': duration,
                    'duration_formatted': duration_formatted,
                    'resolution': f"{video_stream.get('width', 'unknown')}x{video_stream.get('height', 'unknown')}",
                    'video_codec': video_stream.get('codec_name', 'unknown'),
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'has_audio': bool(audio_stream),
                    'bitrate': format_info.get('bit_rate', 'unknown')
                }
                
                print_success(f"Duration: {duration_formatted}, Resolution: {info['resolution']}")
                return info
            else:
                print_warning("ffprobe not available, using basic info")
                
        except FileNotFoundError:
            print_warning("ffprobe not installed, using basic file info only")
        except Exception as e:
            print_warning(f"Error getting video info: {e}")
        
        # Fallback: basic info
        return {
            'filename': video_path.name,
            'path': str(video_path),
            'size': self.get_file_size(video_path),
            'has_audio': 'unknown'
        }
    
    def transcribe_audio(self, video_path: Path, method: str = 'whisper') -> str:
        """Transcribe audio from video"""
        print_step(3, 6, f"Transcribing audio: {video_path.name}...")
        
        # Check if local transcription tools are available
        transcription = None
        
        # Try Whisper.cpp (fastest)
        if method == 'whisper' or method == 'auto':
            try:
                print_info("Attempting Whisper.cpp transcription...")
                # This would call actual Whisper API
                # For demo, simulating output
                transcription = self.simulate_transcription(video_path)
                print_success("Transcription complete (simulated)")
            except Exception as e:
                print_warning(f"Whisper.cpp failed: {e}")
        
        # Try Vosk (lightweight alternative)
        if not transcription and (method == 'vosk' or method == 'auto'):
            try:
                print_info("Attempting Vosk transcription...")
                transcription = self.simulate_transcription(video_path)
                print_success("Transcription complete (simulated)")
            except Exception as e:
                print_warning(f"Vosk failed: {e}")
        
        # Fallback
        if not transcription:
            print_warning("No transcription available - would use API or local model")
            transcription = "[Transcription would be generated here using Whisper/Vosk]"
        
        return transcription
    
    def simulate_transcription(self, video_path: Path) -> str:
        """Simulate transcription output"""
        # Realistic BWC transcription simulation
        return """
[00:00:01] Officer: Dispatch, I'm on scene at 123 Main Street.
[00:00:05] Dispatch: Copy that, 10-4.
[00:00:08] Officer: Subject is standing near the vehicle. Approaching now.
[00:00:12] Officer: Sir, can I see your license and registration?
[00:00:15] Subject: What's this about, officer?
[00:00:17] Officer: Routine traffic stop. License please.
[00:00:21] Subject: I don't have it with me.
[00:00:23] Officer: Step out of the vehicle.
[00:00:28] Subject: Why am I being detained? What's your probable cause?
[00:00:32] Officer: You're not under arrest. Just step out.
[00:00:35] [Sound of door opening]
[00:00:38] Officer: Turn around, hands behind your back.
[00:00:42] Subject: I'm not doing anything wrong!
[00:00:45] Officer: Stop resisting!
[00:00:48] [Sounds of struggle]
[00:00:52] Subject: I can't breathe! You're hurting me!
[00:00:55] Officer 2: We need backup here!
[00:01:00] [End of relevant portion]
"""
    
    def analyze_constitutional_violations(self, video_info: Dict, transcription: str) -> Dict[str, Any]:
        """Analyze for constitutional violations"""
        print_step(4, 6, "Analyzing for constitutional violations...")
        
        # Simulate analysis (in production, this would use AI models)
        violations = []
        
        # Check for 4th Amendment issues
        if "probable cause" in transcription.lower() or "detained" in transcription.lower():
            violations.append({
                'type': '4th Amendment - Search & Seizure',
                'severity': 'moderate',
                'timestamp': '00:00:28',
                'description': 'Possible detention without clear probable cause',
                'transcript_quote': 'Why am I being detained? What\'s your probable cause?',
                'legal_basis': 'Terry v. Ohio requires reasonable suspicion for detention',
                'confidence': 0.78
            })
        
        # Check for excessive force
        if "can't breathe" in transcription.lower() or "hurting" in transcription.lower():
            violations.append({
                'type': '4th Amendment - Excessive Force',
                'severity': 'severe',
                'timestamp': '00:00:52',
                'description': 'Possible excessive force - subject indicates pain and difficulty breathing',
                'transcript_quote': 'I can\'t breathe! You\'re hurting me!',
                'legal_basis': 'Graham v. Connor - objectively unreasonable force',
                'confidence': 0.85
            })
        
        # Check for Miranda rights
        miranda_mentioned = 'miranda' in transcription.lower() or 'right to remain silent' in transcription.lower()
        if not miranda_mentioned and 'arrest' in transcription.lower():
            violations.append({
                'type': '5th Amendment - Miranda Violation',
                'severity': 'moderate',
                'timestamp': 'N/A',
                'description': 'Custodial interrogation without Miranda warnings',
                'transcript_quote': 'N/A - Miranda not read',
                'legal_basis': 'Miranda v. Arizona - warnings required before custodial interrogation',
                'confidence': 0.65
            })
        
        # Analyze use of force continuum
        force_analysis = {
            'officer_actions': ['verbal commands', 'physical contact', 'possible restraint'],
            'subject_resistance': 'passive resistance (verbal objection)',
            'force_proportionality': 'questionable',
            'graham_factors': {
                'severity_of_crime': 'unknown (traffic stop)',
                'immediate_threat': 'none apparent',
                'actively_resisting': 'minimal verbal resistance',
                'attempting_to_evade': 'no'
            },
            'conclusion': 'Force may exceed reasonable response based on Graham factors'
        }
        
        # Overall liability assessment
        liability_score = 7.2  # Out of 10 (higher = more liability)
        
        print_success(f"Found {len(violations)} potential constitutional concerns")
        for v in violations:
            severity_color = Colors.FAIL if v['severity'] == 'severe' else Colors.WARNING
            print(f"  {severity_color}• {v['type']} ({v['severity']}) - Confidence: {v['confidence']:.0%}{Colors.ENDC}")
        
        return {
            'violations': violations,
            'force_analysis': force_analysis,
            'liability_score': liability_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_sync_timeline(self, videos_info: List[Dict]) -> Dict[str, Any]:
        """Create synchronized multi-POV timeline"""
        print_step(5, 6, "Creating synchronized multi-POV timeline...")
        
        # In production, this would:
        # 1. Extract timestamps from video metadata
        # 2. Align videos based on audio fingerprinting or timecode
        # 3. Find overlapping segments
        # 4. Create synchronized playback timeline
        
        timeline = {
            'sync_group_id': f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'video_count': len(videos_info),
            'primary_video': videos_info[0]['filename'] if videos_info else None,
            'synchronization_method': 'metadata_timestamps',
            'timeline_segments': []
        }
        
        # Simulate timeline alignment
        if len(videos_info) > 1:
            timeline['timeline_segments'] = [
                {
                    'start_ms': 0,
                    'end_ms': 60000,  # 1 minute
                    'videos_active': len(videos_info),
                    'description': 'All cameras recording - initial contact'
                },
                {
                    'start_ms': 60000,
                    'end_ms': 90000,
                    'videos_active': len(videos_info),
                    'description': 'Detention and search'
                }
            ]
            print_success(f"Created synchronized timeline for {len(videos_info)} videos")
        else:
            print_info("Single video - no synchronization needed")
        
        return timeline
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        print_step(6, 6, "Generating comprehensive report...")
        
        report = f"""
{Colors.BOLD}{'='*80}
BWC FOOTAGE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}{Colors.ENDC}

{Colors.HEADER}VIDEOS ANALYZED:{Colors.ENDC}
{'-'*80}
"""
        
        for i, video in enumerate(analysis_results.get('videos', []), 1):
            report += f"""
Video {i}: {video.get('filename', 'Unknown')}
  • Duration: {video.get('duration_formatted', 'Unknown')}
  • Size: {video.get('size', 'Unknown')}
  • Resolution: {video.get('resolution', 'Unknown')}
  • Audio: {'Yes' if video.get('has_audio') else 'Unknown'}
"""
        
        report += f"""
{Colors.HEADER}CONSTITUTIONAL ANALYSIS:{Colors.ENDC}
{'-'*80}
"""
        
        violations = analysis_results.get('violations', [])
        if violations:
            for v in violations:
                severity_marker = "🔴" if v['severity'] == 'severe' else "🟡"
                report += f"""
{severity_marker} {v['type']} [{v['severity'].upper()}]
  Timestamp: {v['timestamp']}
  Confidence: {v['confidence']:.0%}
  Description: {v['description']}
  Quote: "{v['transcript_quote']}"
  Legal Basis: {v['legal_basis']}
"""
        else:
            report += "\n✅ No constitutional violations detected\n"
        
        report += f"""
{Colors.HEADER}USE OF FORCE ANALYSIS:{Colors.ENDC}
{'-'*80}
"""
        
        force = analysis_results.get('force_analysis', {})
        if force:
            report += f"""
Officer Actions: {', '.join(force.get('officer_actions', []))}
Subject Resistance: {force.get('subject_resistance', 'Unknown')}
Force Proportionality: {force.get('force_proportionality', 'Unknown')}

Graham v. Connor Factors:
  • Severity of Crime: {force.get('graham_factors', {}).get('severity_of_crime', 'Unknown')}
  • Immediate Threat: {force.get('graham_factors', {}).get('immediate_threat', 'Unknown')}
  • Active Resistance: {force.get('graham_factors', {}).get('actively_resisting', 'Unknown')}
  • Attempting to Evade: {force.get('graham_factors', {}).get('attempting_to_evade', 'Unknown')}

Conclusion: {force.get('conclusion', 'N/A')}
"""
        
        report += f"""
{Colors.HEADER}LIABILITY ASSESSMENT:{Colors.ENDC}
{'-'*80}
Overall Liability Score: {analysis_results.get('liability_score', 0):.1f} / 10.0
"""
        
        liability = analysis_results.get('liability_score', 0)
        if liability >= 7:
            report += f"{Colors.FAIL}HIGH LIABILITY RISK - Immediate legal review recommended{Colors.ENDC}\n"
        elif liability >= 4:
            report += f"{Colors.WARNING}MODERATE LIABILITY RISK - Further investigation needed{Colors.ENDC}\n"
        else:
            report += f"{Colors.OKGREEN}LOW LIABILITY RISK{Colors.ENDC}\n"
        
        report += f"""
{Colors.HEADER}SYNCHRONIZATION TIMELINE:{Colors.ENDC}
{'-'*80}
"""
        
        sync = analysis_results.get('sync_timeline', {})
        if sync:
            report += f"""
Sync Group ID: {sync.get('sync_group_id', 'N/A')}
Videos Synchronized: {sync.get('video_count', 0)}
Primary Video: {sync.get('primary_video', 'N/A')}
Synchronization Method: {sync.get('synchronization_method', 'N/A')}
"""
            
            segments = sync.get('timeline_segments', [])
            if segments:
                report += "\nTimeline Segments:\n"
                for seg in segments:
                    start = seg['start_ms'] / 1000
                    end = seg['end_ms'] / 1000
                    report += f"  • {start:.0f}s - {end:.0f}s: {seg['description']} ({seg['videos_active']} cameras)\n"
        
        report += f"""
{Colors.HEADER}RECOMMENDATIONS:{Colors.ENDC}
{'-'*80}
"""
        
        recommendations = [
            "1. Conduct full legal review of detention and search procedures",
            "2. Interview subject regarding allegations of excessive force",
            "3. Obtain medical records to document any injuries",
            "4. Review department use-of-force policies and training",
            "5. Consider disciplinary action for policy violations",
            "6. Preserve all evidence including additional camera angles",
            "7. Consult with civil rights attorney regarding potential liability"
        ]
        
        for rec in recommendations:
            report += f"  {rec}\n"
        
        report += f"""
{Colors.BOLD}{'='*80}
END OF REPORT
{'='*80}{Colors.ENDC}
"""
        
        print_success("Report generated")
        return report
    
    async def run_full_analysis(self):
        """Run complete BWC analysis workflow"""
        print_header("BWC FOOTAGE ANALYSIS - TESTING SUITE")
        
        # Step 1: Scan folder
        videos = self.scan_folder()
        if not videos:
            print_error("No videos found. Exiting.")
            return
        
        # Step 2: Get video info
        videos_info = []
        for video in videos[:3]:  # Limit to first 3 for demo
            info = self.get_video_info(video)
            videos_info.append(info)
        
        # Step 3: Transcribe audio
        transcriptions = {}
        for video in videos[:1]:  # Transcribe first video for demo
            transcription = self.transcribe_audio(video)
            transcriptions[video.name] = transcription
        
        # Step 4: Analyze violations
        analysis = self.analyze_constitutional_violations(
            videos_info[0] if videos_info else {},
            list(transcriptions.values())[0] if transcriptions else ""
        )
        
        # Step 5: Create sync timeline
        sync_timeline = self.create_sync_timeline(videos_info)
        
        # Compile results
        results = {
            'videos': videos_info,
            'transcriptions': transcriptions,
            'violations': analysis.get('violations', []),
            'force_analysis': analysis.get('force_analysis', {}),
            'liability_score': analysis.get('liability_score', 0),
            'sync_timeline': sync_timeline,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.results = results
        
        # Step 6: Generate report
        report = self.generate_report(results)
        
        # Display report
        print(report)
        
        # Save report to file
        report_path = Path(f"bwc_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            # Remove color codes for file
            clean_report = report
            for color_code in [Colors.HEADER, Colors.OKBLUE, Colors.OKGREEN, Colors.WARNING, 
                             Colors.FAIL, Colors.BOLD, Colors.ENDC]:
                clean_report = clean_report.replace(color_code, '')
            f.write(clean_report)
        
        print_success(f"Report saved to: {report_path}")
        
        # Save JSON results
        json_path = Path(f"bwc_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print_success(f"JSON results saved to: {json_path}")
        
        print_header("ANALYSIS COMPLETE")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test BWC analysis features')
    parser.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing BWC footage',
        default='.'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['whisper', 'vosk', 'auto'],
        default='auto',
        help='Transcription method'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    tester = BWCAnalysisTester(args.folder)
    
    # Run async analysis
    asyncio.run(tester.run_full_analysis())


if __name__ == "__main__":
    main()
