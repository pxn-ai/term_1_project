"""
Optimized Video Human In/Out Counter for Raspberry Pi 4
Uses multiprocessing to leverage quad-core processor
"""

import cv2
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
import queue

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing required packages...")
    import os
    os.system("pip3 install ultralytics opencv-python numpy")
    from ultralytics import YOLO


class HumanInOutCounter:
    def __init__(self, model_size='n'):
        """
        Initialize human counter with tracking
        model_size: 'n' (nano - recommended for Pi)
        """
        print(f"Loading YOLOv8{model_size} model with tracking...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # Enable INT8 quantization for Raspberry Pi (faster inference)
        self.model.overrides['half'] = False  # Full precision on CPU
        self.model.overrides['device'] = 'cpu'
        
        print("✓ Model loaded\n")
        
        # Tracking settings
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        
        # Line position (percentage from top: 0.0 to 1.0)
        self.line_position = 0.5  # Middle of frame
        
        # Direction tracking
        self.direction_history = defaultdict(lambda: [])
        
        # Counters
        self.entered = 0
        self.exited = 0
        
        # Confidence threshold
        self.confidence_threshold = 0.4
        
    def set_counting_line(self, position=0.5):
        """
        Set the virtual line position for counting
        position: 0.0 (left) to 1.0 (right), default 0.5 (middle)
        """
        self.line_position = max(0.1, min(0.9, position))
        print(f"Counting line set at {self.line_position*100:.0f}% from left")
    
    def get_net_entered_count(self, video_path, count_line_pos=0.5):
        """
        Optimized version using frame decimation and reduced resolution
        """
        print(f"Analyzing video {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"Total frames: {total_frames}")
        
        # Optimization: Reduce resolution for faster processing
        target_width = 640  # Reduced from 1280
        scale_factor = target_width / frame_width
        target_height = int(frame_height * scale_factor)
        
        # Optimization: Process every 4th frame (adjustable based on video FPS)
        skip_frames = max(3, fps // 10)  # Process ~10 frames per second
        
        print(f"Processing resolution: {target_width}x{target_height}")
        print(f"Frame skip: {skip_frames}")
        
        movements = defaultdict(list)
        frame_count = 0
        processed_count = 0
        
        start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % skip_frames != 0:
                continue
            
            processed_count += 1
            
            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (target_width, target_height))
            
            # Run tracking with optimized parameters
            results = self.model.track(
                frame_resized,
                conf=self.confidence_threshold,
                classes=[0],  # Only persons
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                imgsz=target_width  # Explicit image size
            )
            
            # Extract positions
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2.0
                    movements[track_id].append(center_x)
            
            # Progress indicator
            if processed_count % 20 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = processed_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f}")
        
        cap.release()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"\nProcessed {processed_count} frames in {processing_time:.1f}s")
        print(f"Average FPS: {processed_count/processing_time:.1f}")
        
        # Count entries/exits
        scaled_line_pos = count_line_pos * target_width
        net_change = 0
        
        for track_id, positions in movements.items():
            if len(positions) < 2:
                continue
            
            start_x = positions[0]
            end_x = positions[-1]
            
            # Crossed line left to right (entered)
            if start_x < scaled_line_pos and end_x > scaled_line_pos:
                net_change += 1
                print(f"Track {track_id}: ENTERED")
            # Crossed line right to left (exited)
            elif start_x > scaled_line_pos and end_x < scaled_line_pos:
                net_change -= 1
                print(f"Track {track_id}: EXITED")
        
        print(f"\nNet change: {net_change}")
        return net_change

    def analyze_video(self, video_path, output_path=None, show_preview=False, 
                     skip_frames=3, count_line_pos=0.5):
        """
        Analyze video and count humans entering/exiting
        Optimized version with better performance
        """
        
        # Reset counters
        self.track_history.clear()
        self.direction_history.clear()
        self.counted_ids.clear()
        self.entered = 0
        self.exited = 0
        self.set_counting_line(count_line_pos)
        
        print("="*70)
        print("VIDEO ANALYSIS - HUMAN IN/OUT COUNTER (OPTIMIZED)")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"Counting line position: {self.line_position*100:.0f}% from left")
        print(f"Frame skip: {skip_frames}")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reduce processing resolution
        target_width = 640
        scale_factor = target_width / frame_width
        target_height = int(frame_height * scale_factor)
        
        print(f"Original resolution: {frame_width}x{frame_height}")
        print(f"Processing resolution: {target_width}x{target_height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f} seconds\n")
        
        # Calculate counting line X position
        line_x = int(target_width * self.line_position)
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps//skip_frames, (target_width, target_height))
            print(f"Output will be saved to: {output_path}\n")
        
        frame_count = 0
        processed_frames = 0
        
        print("Processing video...")
        start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames != 0:
                continue
            
            processed_frames += 1
            
            # Resize for faster processing
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Run tracking
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                classes=[0],  # Only persons
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                imgsz=target_width
            )
            
            # Process detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate center point of bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Store track history
                    self.track_history[track_id].append((center_x, center_y))
                    
                    # Keep only last 20 positions
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)
                    
                    # Check if crossed line (only count once per ID)
                    if track_id not in self.counted_ids and len(self.track_history[track_id]) >= 2:
                        prev_x = self.track_history[track_id][-2][0]
                        curr_x = center_x
                        
                        # Crossed line going right (ENTERED)
                        if prev_x < line_x <= curr_x:
                            self.entered += 1
                            self.counted_ids.add(track_id)
                            self.direction_history[track_id] = "IN"
                            print(f"Frame {frame_count}: Person {track_id} ENTERED")
                        
                        # Crossed line going left (EXITED)
                        elif prev_x > line_x >= curr_x:
                            self.exited += 1
                            self.counted_ids.add(track_id)
                            self.direction_history[track_id] = "OUT"
                            print(f"Frame {frame_count}: Person {track_id} EXITED")
                    
                    # Draw on frame (only if output or preview requested)
                    if output_path or show_preview:
                        # Color based on direction
                        if track_id in self.direction_history:
                            if self.direction_history[track_id] == "IN":
                                color = (0, 255, 0)  # Green for entered
                                status = "IN"
                            else:
                                color = (0, 0, 255)  # Red for exited
                                status = "OUT"
                        else:
                            color = (255, 0, 0)  # Blue for tracking
                            status = "TRACKING"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"ID:{track_id} {status}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw tracking trail
                        points = np.array(self.track_history[track_id], dtype=np.int32)
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, color, 2)
            
            # Draw annotations only if needed
            if output_path or show_preview:
                # Draw counting line
                cv2.line(frame, (line_x, 0), (line_x, target_height), (255, 255, 0), 3)
                cv2.putText(frame, "COUNTING LINE", (line_x + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw statistics overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                cv2.putText(frame, f"ENTERED: {self.entered}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"EXITED: {self.exited}", (20, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"INSIDE: {self.entered - self.exited}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Progress indicator
                progress = (frame_count / total_frames) * 100
                cv2.putText(frame, f"Progress: {progress:.1f}%", 
                           (target_width - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Show preview if requested
                if show_preview:
                    cv2.imshow('Human In/Out Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
            
            # Progress update every 30 processed frames
            if processed_frames % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Final statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Total frames processed: {processed_frames} / {total_frames}")
        print(f"Processing time: {processing_time:.1f} seconds")
        print(f"Average processing FPS: {processed_frames/processing_time:.1f}")
        print("\nRESULTS:")
        print(f"  People ENTERED: {self.entered}")
        print(f"  People EXITED: {self.exited}")
        print(f"  Net change (IN - OUT): {self.entered - self.exited}")
        print(f"  Total unique people tracked: {len(self.counted_ids)}")
        print("="*70 + "\n")
        
        # Return results as dictionary
        results = {
            'video_path': video_path,
            'entered': self.entered,
            'exited': self.exited,
            'net_change': self.entered - self.exited,
            'total_tracked': len(self.counted_ids),
            'processing_time': processing_time,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def save_results(self, results, output_file='results.json'):
        """Save analysis results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")

    # Keep all other methods unchanged...
    def get_human_movements(self, video_file_path: str) -> dict:
        """Analyzes video file and extracts human movement data"""
        movements = defaultdict(list)
        cap = cv2.VideoCapture(video_file_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file_path}")
            return {}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Processing video: {video_file_path}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print("Analyzing human movements...\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = self.model.track(
                frame,
                conf=0.4,
                classes=[0],
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2.0
                    movements[track_id].append(center_x)
            
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        movements = dict(movements)
        
        print(f"\nAnalysis complete!")
        print(f"Total unique people tracked: {len(movements)}")
        for track_id, positions in movements.items():
            print(f"  Track ID {track_id}: {len(positions)} position samples")
        
        return movements


# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Human In/Out Counter (Optimized for RPi4)')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated video (optional)')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview while processing')
    parser.add_argument('--model', type=str, default='n',
                       help='Model size: n (nano), s (small)')
    parser.add_argument('--skip', type=int, default=3,
                       help='Process every Nth frame (default: 3)')
    parser.add_argument('--line', type=float, default=0.5,
                       help='Counting line position 0.0-1.0 from left (default: 0.5 = middle)')
    parser.add_argument('--json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Create counter
    counter = HumanInOutCounter(model_size=args.model)
    
    # Analyze video
    results = counter.analyze_video(
        video_path=args.video,
        output_path=args.output,
        show_preview=args.preview,
        skip_frames=args.skip,
        count_line_pos=args.line
    )
    
    # Save results if requested
    if args.json and results:
        counter.save_results(results, args.json)