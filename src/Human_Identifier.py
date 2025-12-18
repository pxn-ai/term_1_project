"""
Video Human In/Out Counter for Raspberry Pi
Uses YOLO object detection with ByteTrack to track and count people entering/exiting.
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
    """Tracks humans in video and counts entries/exits using a virtual counting line."""
    
    def __init__(self, model_size='n'):
        """Initialize the counter with YOLOv8 model. Use 'n' (nano) for Raspberry Pi."""
        print(f"Loading YOLOv8{model_size} model with tracking...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        self.model.overrides['half'] = False
        self.model.overrides['device'] = 'cpu'
        print("✓ Model loaded\n")
        
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        self.line_position = 0.5
        self.direction_history = defaultdict(lambda: [])
        self.entered = 0
        self.exited = 0
        self.confidence_threshold = 0.4
        
    def set_counting_line(self, position=0.5):
        """Set virtual counting line position (0.0=left to 1.0=right)."""
        self.line_position = max(0.1, min(0.9, position))
        print(f"Counting line set at {self.line_position*100:.0f}% from left")
    
    def analyze_video(self, video_path, output_path=None, show_preview=False, 
                     skip_frames=3, count_line_pos=0.5):
        """Analyze video to count people crossing the counting line."""
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
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Scale down for faster processing
        target_width = 640
        scale_factor = target_width / frame_width
        target_height = int(frame_height * scale_factor)
        
        print(f"Original resolution: {frame_width}x{frame_height}")
        print(f"Processing resolution: {target_width}x{target_height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f} seconds\n")
        
        line_x = int(target_width * self.line_position)
        
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
            
            if frame_count % skip_frames != 0:
                continue
            
            processed_frames += 1
            
            frame = cv2.resize(frame, (target_width, target_height))
            
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                classes=[0],  # Only persons
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                imgsz=target_width
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    self.track_history[track_id].append((center_x, center_y))
                    
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)
                    
                    # Check if person crossed the counting line
                    if track_id not in self.counted_ids and len(self.track_history[track_id]) >= 2:
                        prev_x = self.track_history[track_id][-2][0]
                        curr_x = center_x
                        
                        if prev_x < line_x <= curr_x:
                            self.entered += 1
                            self.counted_ids.add(track_id)
                            self.direction_history[track_id] = "IN"
                            print(f"Frame {frame_count}: Person {track_id} ENTERED")
                        
                        elif prev_x > line_x >= curr_x:
                            self.exited += 1
                            self.counted_ids.add(track_id)
                            self.direction_history[track_id] = "OUT"
                            print(f"Frame {frame_count}: Person {track_id} EXITED")
                    
                    if output_path or show_preview:
                        # Color code: green=entered, red=exited, blue=tracking
                        if track_id in self.direction_history:
                            if self.direction_history[track_id] == "IN":
                                color = (0, 255, 0)
                                status = "IN"
                            else:
                                color = (0, 0, 255)
                                status = "OUT"
                        else:
                            color = (255, 0, 0)
                            status = "TRACKING"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track_id} {status}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        points = np.array(self.track_history[track_id], dtype=np.int32)
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, color, 2)
            
            if output_path or show_preview:
                cv2.line(frame, (line_x, 0), (line_x, target_height), (255, 255, 0), 3)
                cv2.putText(frame, "COUNTING LINE", (line_x + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                cv2.putText(frame, f"ENTERED: {self.entered}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"EXITED: {self.exited}", (20, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"INSIDE: {self.entered - self.exited}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                progress = (frame_count / total_frames) * 100
                cv2.putText(frame, f"Progress: {progress:.1f}%", 
                           (target_width - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if out:
                    out.write(frame)
                
                if show_preview:
                    cv2.imshow('Human In/Out Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
            
            if processed_frames % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f}")
        
        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
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
        """Save analysis results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    
    def get_human_movements(self, video_file_path: str) -> dict:
        """
        Extract x-position history for each tracked person.
        Returns: {track_id: [x_positions]}
        """
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

    def get_human_movements_with_y(self, video_file_path: str) -> dict:
        """Extract both x and y positions. Returns: {track_id: {'x': [], 'y': []}}"""
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
        """Extract detailed tracking data including bounding boxes and confidence scores."""
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
        """Calculate movement statistics (displacement, direction) for each tracked person."""
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
    
    def get_net_entered_count(self, video_path, count_line_pos=0.5):
        """Return net count of people who crossed the line (entered - exited)."""
        results = self.get_human_movements(video_path)
        if results is None:
            return 0
        
        net_change = 0
        for positions in results.values():
            if positions[0] < positions[-1] and positions[-1] > count_line_pos:
                net_change += 1
            elif positions[0] > positions[-1] and positions[-1] < count_line_pos:
                net_change -= 1

        return net_change
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