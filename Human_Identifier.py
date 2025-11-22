"""
Video Human In/Out Counter for Raspberry Pi
Analyzes video clips to count people entering and exiting
Uses tracking to follow individuals across frames
"""

import cv2
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

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
    
    def analyze_video(self, video_path, output_path=None, show_preview=False, 
                     skip_frames=1, count_line_pos=0.5):
        """
        Analyze video and count humans entering/exiting
        
        video_path: Path to video file
        output_path: Optional path to save annotated video
        show_preview: Show video while processing (requires display)
        skip_frames: Process every Nth frame for speed
        count_line_pos: Position of counting line (0.0-1.0)
        """
        
        # Reset counters
        self.track_history.clear()
        self.direction_history.clear()
        self.counted_ids.clear()
        self.entered = 0
        self.exited = 0
        self.set_counting_line(count_line_pos)
        
        print("="*70)
        print("VIDEO ANALYSIS - HUMAN IN/OUT COUNTER")
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
        
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f} seconds\n")
        
        # Calculate counting line X position
        line_x = int(frame_width * self.line_position)
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
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
            
            # Run tracking
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                classes=[0],  # Only persons
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
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
                    
                    # Draw on frame
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
            
            # Draw counting line
            cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 255, 0), 3)
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
                       (frame_width - 200, 30),
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
    
    def get_human_movements(self, video_file_path: str) -> dict:
        """
        Analyzes video file and extracts human movement data
        
        Args:
            video_file_path (str): Path to the video file
        
        Returns:
            dict: Dictionary where:
                  - key: track_id (int)
                  - value: list of x_positions (center x coordinate) for each frame
                  
        Example output:
            {
                1: [320.5, 325.0, 330.2, 335.8, ...],  # Person 1's x positions
                2: [640.1, 638.5, 635.0, 632.3, ...],  # Person 2's x positions
                3: [150.0, 155.5, 160.2, ...]          # Person 3's x positions
            }
        """
        
        # Initialize tracking dictionary
        movements = defaultdict(list)
        
        # Open video
        cap = cv2.VideoCapture(video_file_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file_path}")
            return {}
        
        # Get video properties
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
            
            # Run tracking on frame
            results = self.model.track(
                frame,
                conf=0.4,           # Confidence threshold
                classes=[0],        # Only detect persons (class 0)
                persist=True,       # Keep track IDs consistent
                verbose=False,      # Suppress output
                tracker="bytetrack.yaml"
            )
            
            # Extract tracking information
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # Calculate center x position
                    center_x = (x1 + x2) / 2.0
                    
                    # Store x position for this track_id
                    movements[track_id].append(center_x)
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        
        # Convert defaultdict to regular dict
        movements = dict(movements)
        
        print(f"\nAnalysis complete!")
        print(f"Total unique people tracked: {len(movements)}")
        for track_id, positions in movements.items():
            print(f"  Track ID {track_id}: {len(positions)} position samples")
        
        return movements

    def get_human_movements_with_y(self, video_file_path: str) -> dict:
        """
        Extended version that returns both x and y positions
        
        Returns:
            dict: {
                track_id: {
                    'x': [x1, x2, x3, ...],
                    'y': [y1, y2, y3, ...]
                }
            }
        """
        movements = defaultdict(lambda: {'x': [], 'y': []})
        
        cap = cv2.VideoCapture(video_file_path)
        
        if not cap.isOpened():
            return {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
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
                    center_y = (y1 + y2) / 2.0
                    
                    movements[track_id]['x'].append(center_x)
                    movements[track_id]['y'].append(center_y)
        
        cap.release()
        return dict(movements)

    def get_human_movements_detailed(self, video_file_path: str) -> dict:
        """
        Detailed version with complete bounding box and metadata
        
        Returns:
            dict: {
                track_id: {
                    'frames': [frame_numbers],
                    'x': [center_x positions],
                    'y': [center_y positions],
                    'bbox': [(x1, y1, x2, y2), ...],
                    'confidence': [conf1, conf2, ...]
                }
            }
        """
        movements = defaultdict(lambda: {
            'frames': [],
            'x': [],
            'y': [],
            'bbox': [],
            'confidence': []
        })
        
        cap = cv2.VideoCapture(video_file_path)
        
        if not cap.isOpened():
            return {}
        
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
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    
                    movements[track_id]['frames'].append(frame_count)
                    movements[track_id]['x'].append(center_x)
                    movements[track_id]['y'].append(center_y)
                    movements[track_id]['bbox'].append((float(x1), float(y1), float(x2), float(y2)))
                    movements[track_id]['confidence'].append(float(conf))
        
        cap.release()
        return dict(movements)

    def analyze_movement_statistics(self, movements: dict) -> dict:
        """
        Calculate statistics from movement data
        
        Args:
            movements: Output from get_human_movements()
        
        Returns:
            dict: Statistics for each track_id
        """
        stats = {}
        
        for track_id, x_positions in movements.items():
            if len(x_positions) < 2:
                continue
            
            x_array = np.array(x_positions)
            
            stats[track_id] = {
                'total_frames': len(x_positions),
                'start_x': x_positions[0],
                'end_x': x_positions[-1],
                'min_x': float(np.min(x_array)),
                'max_x': float(np.max(x_array)),
                'mean_x': float(np.mean(x_array)),
                'total_displacement': float(x_positions[-1] - x_positions[0]),
                'total_distance': float(np.sum(np.abs(np.diff(x_array)))),
                'direction': 'left-to-right' if x_positions[-1] > x_positions[0] else 'right-to-left'
            }
        
        return stats
    
    def get_net_entered_count(self, video_path, count_line_pos= 0.5 ) :
        """Return current counting results as integer ( net change ) """
        results = self.get_human_movements(video_path)
        if results is None:
            return 0
        
        net_change = 0
        for positions in results.values():
            if positions[0] < positions[-1] and positions[-1] > count_line_pos :
                net_change += 1
            elif positions[0] > positions[-1] and positions[-1] < count_line_pos :
                net_change -= 1

        return net_change


# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Human In/Out Counter')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated video (optional)')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview while processing')
    parser.add_argument('--model', type=str, default='n',
                       help='Model size: n (nano), s (small)')
    parser.add_argument('--skip', type=int, default=2,
                       help='Process every Nth frame (default: 2)')
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