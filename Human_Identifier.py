"""
OPTIMIZED Video Human In/Out Counter for Raspberry Pi 4
Uses multiprocessing to leverage all 4 cores for maximum speed
"""

import cv2
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing required packages...")
    import os
    os.system("pip3 install ultralytics opencv-python numpy")
    from ultralytics import YOLO


import math

def process_video_segment(segment_info, model_path, conf_threshold):
    """
    Process a segment of video frames in a separate process
    Returns movements dictionary and track metadata for stitching
    """
    video_path, start_frame, end_frame, segment_id = segment_info
    
    # Load model in this process
    model = YOLO(model_path)
    
    movements = defaultdict(list)
    track_meta = {}  # Store start/end info for stitching
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}, {}
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    frames_batch = []
    batch_size = 4  # Process 4 frames at once
    
    print(f"[Worker {segment_id}] Processing frames {start_frame}-{end_frame}")
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames (process every 3rd frame)
        if frame_count % 3 != 0:
            continue
        
        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 360))
        frames_batch.append(frame_resized)
        
        # Process batch when full
        if len(frames_batch) >= batch_size:
            results = model.track(
                frames_batch,
                conf=conf_threshold,
                classes=[0],
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            
            # Extract positions from batch results
            for i, result in enumerate(results):
                current_frame_idx = frame_count - len(frames_batch) + i + 1
                
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, track_ids):
                        center_x = (box[0] + box[2]) / 2.0
                        center_y = (box[1] + box[3]) / 2.0
                        
                        # Store movement (X only for counting logic)
                        movements[track_id].append(center_x)
                        
                        # Update metadata for stitching
                        if track_id not in track_meta:
                            track_meta[track_id] = {
                                'start_frame': current_frame_idx,
                                'end_frame': current_frame_idx,
                                'start_pos': (center_x, center_y),
                                'end_pos': (center_x, center_y)
                            }
                        else:
                            track_meta[track_id]['end_frame'] = current_frame_idx
                            track_meta[track_id]['end_pos'] = (center_x, center_y)
            
            frames_batch = []
    
    # Process remaining frames
    if frames_batch:
        results = model.track(
            frames_batch,
            conf=conf_threshold,
            classes=[0],
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        for i, result in enumerate(results):
            current_frame_idx = frame_count - len(frames_batch) + i + 1
            
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    center_x = (box[0] + box[2]) / 2.0
                    center_y = (box[1] + box[3]) / 2.0
                    
                    movements[track_id].append(center_x)
                    
                    if track_id not in track_meta:
                        track_meta[track_id] = {
                            'start_frame': current_frame_idx,
                            'end_frame': current_frame_idx,
                            'start_pos': (center_x, center_y),
                            'end_pos': (center_x, center_y)
                        }
                    else:
                        track_meta[track_id]['end_frame'] = current_frame_idx
                        track_meta[track_id]['end_pos'] = (center_x, center_y)
    
    cap.release()
    
    print(f"[Worker {segment_id}] Completed: {len(movements)} tracks found")
    return dict(movements), track_meta


class HumanInOutCounter:
    def __init__(self, model_size='n'):
        """
        Initialize human counter with tracking
        model_size: 'n' (nano - recommended for Pi)
        """
        print(f"Loading YOLOv8{model_size} model with tracking...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.model_path = f'yolov8{model_size}.pt'
        print("✓ Model loaded\n")
        
        # Tracking settings
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        
        # Line position
        self.line_position = 0.5
        
        # Direction tracking
        self.direction_history = defaultdict(lambda: [])
        
        # Counters
        self.entered = 0
        self.exited = 0
        
        # Confidence threshold
        self.confidence_threshold = 0.3  # Lowered to detect more people
        
    def get_net_entered_count_multicore(self, video_path, count_line_pos=0.5, num_workers=3):
        """
        OPTIMIZED: Use multiprocessing to analyze video on multiple cores
        
        Args:
            video_path: Path to video file
            count_line_pos: Position of counting line (pixels from left)
            num_workers: Number of parallel workers (default: 3, leave 1 core free)
        
        Returns:
            int: Net count of people entered (positive) or exited (negative)
        """
        print(f"\n{'='*70}")
        print(f"MULTICORE VIDEO ANALYSIS ({num_workers} workers)")
        print(f"{'='*70}")
        print(f"Video: {video_path}")
        print(f"Counting Line Position: {count_line_pos:.1f}")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Workers: {num_workers}")
        print(f"{'='*70}\n")
        
        # Split video into segments for parallel processing
        frames_per_segment = total_frames // num_workers
        segments = []
        
        for i in range(num_workers):
            start_frame = i * frames_per_segment
            end_frame = (i + 1) * frames_per_segment if i < num_workers - 1 else total_frames
            segments.append((video_path, start_frame, end_frame, i))
        
        # Process segments in parallel
        start_time = datetime.now()
        
        with mp.Pool(processes=num_workers) as pool:
            process_func = partial(
                process_video_segment,
                model_path=self.model_path,
                conf_threshold=self.confidence_threshold
            )
            segment_results = pool.map(process_func, segments)
        
        # Merge results from all segments with ID stitching
        all_movements = {}
        global_max_id = 0
        
        # Store tracks from previous segment that ended near the boundary
        prev_segment_tracks = []
        
        for i, (seg_movements, seg_meta) in enumerate(segment_results):
            current_segment_map = {}  # Map local_id -> global_id
            
            # Get segment boundary info
            seg_start_frame = segments[i][1]
            
            # Process each track in current segment
            for local_id, positions in seg_movements.items():
                meta = seg_meta[local_id]
                
                # Try to match with previous segment
                matched_id = None
                
                if i > 0:
                    # Check if this track started near the beginning of this segment
                    if meta['start_frame'] - seg_start_frame < 10:  # Within 10 frames of start
                        best_dist = float('inf')
                        
                        for prev_id, prev_meta in prev_segment_tracks:
                            # Check if previous track ended near the end of previous segment
                            # (We don't have prev_seg_end_frame easily available here, but 
                            # we know it's the same as current seg_start_frame)
                            if seg_start_frame - prev_meta['end_frame'] < 10:
                                # Calculate distance between end of prev and start of curr
                                p1 = prev_meta['end_pos']
                                p2 = meta['start_pos']
                                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                                
                                # Threshold for matching (e.g., 100 pixels)
                                if dist < 100 and dist < best_dist:
                                    best_dist = dist
                                    matched_id = prev_id
                
                if matched_id is not None:
                    global_id = matched_id
                    print(f"Stitched track: Seg {i} ID {local_id} -> Global ID {global_id}")
                else:
                    # Create new global ID
                    global_max_id += 1
                    global_id = global_max_id
                
                current_segment_map[local_id] = global_id
                
                # Add to all_movements
                if global_id not in all_movements:
                    all_movements[global_id] = []
                all_movements[global_id].extend(positions)
            
            # Prepare for next segment
            prev_segment_tracks = []
            for local_id, meta in seg_meta.items():
                global_id = current_segment_map[local_id]
                prev_segment_tracks.append((global_id, meta))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"Processing completed in {processing_time:.1f} seconds")
        print(f"Total tracks found: {len(all_movements)}")
        print(f"{'='*70}\n")
        
        # Calculate net change
        net_change = 0
        print("\nTrack Analysis:")
        for track_id, positions in all_movements.items():
            if len(positions) < 2:
                continue
            
            start_x = positions[0]
            end_x = positions[-1]
            min_x = min(positions)
            max_x = max(positions)
            
            print(f"  Track {track_id}: Start={start_x:.0f}, End={end_x:.0f}, Min={min_x:.0f}, Max={max_x:.0f}, Points={len(positions)}")
            
            # Check for crossing events (more robust than just start/end)
            crossed_in = False
            crossed_out = False
            
            # Method 1: Direct start/end check (good for complete tracks)
            if start_x < count_line_pos and end_x > count_line_pos:
                crossed_in = True
            elif start_x > count_line_pos and end_x < count_line_pos:
                crossed_out = True
                
            # Method 2: Check if track covers the line with significant movement
            # This helps if the track starts/ends slightly off but clearly crossed
            if not (crossed_in or crossed_out):
                if min_x < count_line_pos and max_x > count_line_pos:
                    # It crossed, but determine direction based on start/end
                    if start_x < end_x:
                        crossed_in = True
                        print(f"    -> Detected crossing IN (based on min/max)")
                    else:
                        crossed_out = True
                        print(f"    -> Detected crossing OUT (based on min/max)")

            if crossed_in:
                net_change += 1
                print(f"    -> COUNTED: ENTERED")
            elif crossed_out:
                net_change -= 1
                print(f"    -> COUNTED: EXITED")
            else:
                print(f"    -> Ignored (Did not cross line at {count_line_pos:.0f})")
        
        print(f"\nNet change: {net_change:+d}")
        return net_change
    
    def get_net_entered_count(self, video_path, count_line_pos=0.5):
        """
        OPTIMIZED: Single-threaded version with all optimizations
        (fallback if multiprocessing not available)
        """
        movements = defaultdict(list)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return 0
        
        frame_count = 0
        frames_batch = []
        batch_size = 4
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames
                if frames_batch:
                    self._process_batch(frames_batch, movements)
                break
            
            frame_count += 1
            
            # Skip frames (process every 3rd frame)
            if frame_count % 3 != 0:
                continue
            
            # Resize for faster processing
            frame_resized = cv2.resize(frame, (640, 360))
            frames_batch.append(frame_resized)
            
            # Process batch when full
            if len(frames_batch) >= batch_size:
                self._process_batch(frames_batch, movements)
                frames_batch = []
        
        cap.release()
        
        # Calculate net change
        net_change = 0
        for positions in movements.values():
            if len(positions) < 2:
                continue
            if positions[0] < count_line_pos < positions[-1]:
                net_change += 1
            elif positions[0] > count_line_pos > positions[-1]:
                net_change -= 1
        
        return net_change
    
    def _process_batch(self, frames, movements):
        """Process a batch of frames"""
        results = self.model.track(
            frames,
            conf=self.confidence_threshold,
            classes=[0],
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        for result in results:
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    center_x = (box[0] + box[2]) / 2.0
                    movements[track_id].append(center_x)
    
    def save_results(self, net_count, output_file='results.json'):
        """Save analysis results to JSON file"""
        results = {
            'net_change': net_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    
    # Keep old methods for backwards compatibility
    def get_human_movements(self, video_file_path: str) -> dict:
        """Legacy method - kept for compatibility"""
        return self._extract_movements_simple(video_file_path)
    
    def _extract_movements_simple(self, video_path):
        """Simplified movement extraction"""
        movements = defaultdict(list)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 3 != 0:
                continue
            
            frame = cv2.resize(frame, (640, 360))
            
            results = self.model.track(
                frame,
                conf=0.5,
                classes=[0],
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    center_x = (box[0] + box[2]) / 2.0
                    movements[track_id].append(center_x)
        
        cap.release()
        return dict(movements)


# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Video Human In/Out Counter')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='n',
                       help='Model size: n (nano), s (small)')
    parser.add_argument('--line', type=float, default=0.5,
                       help='Counting line position 0.0-1.0 from left (default: 0.5 = middle)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of worker processes (default: 3)')
    parser.add_argument('--json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Create counter
    counter = HumanInOutCounter(model_size=args.model)
    
    # Get video dimensions for line position
    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    # Analyze video with multiprocessing
    net_count = counter.get_net_entered_count_multicore(
        video_path=args.video,
        count_line_pos=args.line * frame_width,
        num_workers=args.workers
    )
    
    print(f"\nFinal Result: {net_count:+d} people")
    
    # Save results if requested
    if args.json:
        counter.save_results(net_count, args.json)