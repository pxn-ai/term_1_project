#!/usr/bin/env python3
"""
Function to extract human movement data from video
Returns track_id -> list of x positions across frames
"""

import cv2
import numpy as np
from collections import defaultdict


def get_human_movements(video_file_path: str, ai_model) -> dict:
    """
    Analyzes video file and extracts human movement data
    
    Args:
        video_file_path (str): Path to the video file
        ai_model: YOLOv8 model instance (already loaded)
    
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
        results = ai_model.track(
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


# Example usage and helper functions
def get_human_movements_with_y(video_file_path: str, ai_model) -> dict:
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
        
        results = ai_model.track(
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


def get_human_movements_detailed(video_file_path: str, ai_model) -> dict:
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
        
        results = ai_model.track(
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


def analyze_movement_statistics(movements: dict) -> dict:
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


# Demo usage
if __name__ == "__main__":
    from ultralytics import YOLO
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Example 1: Basic usage - just x positions
    print("="*70)
    print("Example 1: Basic movement tracking (X positions only)")
    print("="*70)
    movements = get_human_movements("test_video.mp4", model)
    print(f"\nResult: {len(movements)} people tracked")
    for track_id, x_positions in movements.items():
        print(f"Track {track_id}: {len(x_positions)} samples, range: {min(x_positions):.1f} to {max(x_positions):.1f}")
    
    # Example 2: With statistics
    print("\n" + "="*70)
    print("Example 2: Movement with statistics")
    print("="*70)
    stats = analyze_movement_statistics(movements)
    for track_id, stat in stats.items():
        print(f"\nTrack {track_id}:")
        print(f"  Direction: {stat['direction']}")
        print(f"  Total displacement: {stat['total_displacement']:.1f} pixels")
        print(f"  Total distance traveled: {stat['total_distance']:.1f} pixels")
    
    # Example 3: Detailed tracking with X and Y
    print("\n" + "="*70)
    print("Example 3: Detailed tracking (X and Y positions)")
    print("="*70)
    detailed_movements = get_human_movements_with_y("test_video.mp4", model)
    for track_id, positions in detailed_movements.items():
        print(f"Track {track_id}: {len(positions['x'])} frames")
        print(f"  X range: {min(positions['x']):.1f} to {max(positions['x']):.1f}")
        print(f"  Y range: {min(positions['y']):.1f} to {max(positions['y']):.1f}")
    
    # Example 4: Full detailed version
    print("\n" + "="*70)
    print("Example 4: Complete detailed tracking")
    print("="*70)
    full_data = get_human_movements_detailed("test_video.mp4", model)
    for track_id, data in full_data.items():
        print(f"\nTrack {track_id}:")
        print(f"  Frames: {data['frames'][0]} to {data['frames'][-1]}")
        print(f"  Average confidence: {np.mean(data['confidence']):.2%}")
        print(f"  Position range X: {min(data['x']):.1f} to {max(data['x']):.1f}")
        print(f"  Position range Y: {min(data['y']):.1f} to {max(data['y']):.1f}")