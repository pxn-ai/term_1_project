#!/usr/bin/env python3
"""
Human Counting System - Zone Based Logic (Most Efficient)
Counts people based on Zone Transitions (Left Zone <-> Right Zone)
Optimized for Raspberry Pi - Headless Mode
"""

import cv2
import numpy as np
import time
from collections import defaultdict
import torch
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import os
    os.system("pip3 install ultralytics")
    from ultralytics import YOLO

class HumanCounterZone:
    def __init__(self, model_size='n', line_position=0.5):
        """
        Initialize Human Counter
        model_size: 'n' (nano) - ONLY nano recommended for speed
        line_position: 0.0 to 1.0 (percentage of screen width for the counting line)
        """
        print(f"Loading YOLOv8{model_size} for Tracking...")
        
        # Check for GPU
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print("✓ GPU detected! Using CUDA acceleration.")
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            # Use NCNN for CPU optimization
            self.model = YOLO(f'yolov8{model_size}.pt')
            try:
                print("Exporting to NCNN format for CPU optimization...")
                self.model.export(format='ncnn', half=True)
                ncnn_model_path = f'yolov8{model_size}_ncnn_model'
                self.model = YOLO(ncnn_model_path)
                print("✓ Using NCNN optimized model (FP16)")
            except Exception as e:
                print(f"Could not export to NCNN, using standard model: {e}")
        
        # Zone Logic State
        self.person_start_zone = {} # Stores {track_id: 'left' or 'right'}
        self.counts = {'left': 0, 'right': 0}
        self.counted_ids = set()
        
        # Line configuration
        self.line_position = line_position
        self.line_x = 0
        
    def run(self, camera_id=0, resolution=(320, 240)):
        """
        OPTIMIZED: Default to 320x240 for maximum speed
        """
        print(f"Starting Camera {camera_id} at {resolution}")
        print("Format: ( In Frame | Total Went Left | Total Went Right | FPS )")
        print("-" * 60)
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Calculate line position
        self.line_x = int(resolution[0] * self.line_position)
        
        last_print_time = time.time()
        frame_count = 0
        process_count = 0
        
        # PERFORMANCE: Frame skipping
        frame_skip = 2  # Process every 2nd frame
        
        # Buffer zone to prevent jitter counting (pixels)
        buffer = 10 
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # PERFORMANCE: Skip frames
                if frame_count % frame_skip != 0:
                    continue
                
                process_count += 1
                
                # Force resize to target resolution for performance
                if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                    frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
                
                # PERFORMANCE: Lower imgsz and higher conf threshold
                results = self.model.track(frame, 
                                         persist=True, 
                                         classes=[0],
                                         conf=0.4,  # Higher threshold = fewer false positives
                                         verbose=False,
                                         device=self.device,
                                         imgsz=320,  # Reduced inference size
                                         iou=0.5,  # Faster NMS
                                         max_det=10)  # Limit max detections
                
                current_human_count = 0
                active_ids = []
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    current_human_count = len(track_ids)
                    active_ids = track_ids
                    
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        center_x = float(x)
                        
                        # Determine current zone
                        current_zone = None
                        if center_x < (self.line_x - buffer):
                            current_zone = 'left'
                        elif center_x > (self.line_x + buffer):
                            current_zone = 'right'
                        
                        # If in buffer zone, ignore for now
                        if current_zone is None:
                            continue
                            
                        # Logic:
                        # 1. If ID is new, record its starting zone
                        # 2. If ID is known, check if it moved to the opposite zone
                        
                        if track_id not in self.person_start_zone:
                            self.person_start_zone[track_id] = current_zone
                        
                        elif track_id not in self.counted_ids:
                            start_zone = self.person_start_zone[track_id]
                            
                            # Check for transition
                            if start_zone == 'left' and current_zone == 'right':
                                self.counts['right'] += 1
                                self.counted_ids.add(track_id)
                            elif start_zone == 'right' and current_zone == 'left':
                                self.counts['left'] += 1
                                self.counted_ids.add(track_id)

                # Periodic cleanup of old IDs to save memory
                if frame_count % 100 == 0:
                    # Remove IDs that are no longer active
                    active_set = set(active_ids)
                    # Keep counted_ids to prevent double counting if they re-enter? 
                    # For now, let's just clean up start_zones of people who left
                    # This is a simple cleanup strategy
                    keys_to_remove = [k for k in self.person_start_zone.keys() if k not in active_set]
                    for k in keys_to_remove:
                        del self.person_start_zone[k]
                        # Optional: remove from counted_ids if you want to allow re-counting upon re-entry
                        # if k in self.counted_ids: self.counted_ids.remove(k)

                # Print status every second
                current_time = time.time()
                elapsed_time = current_time - last_print_time
                
                if elapsed_time >= 1.0:
                    fps = process_count / elapsed_time
                    print(f"( In Frame: {current_human_count} | Left: {self.counts['left']} | Right: {self.counts['right']} | {fps:.1f} FPS )")
                    last_print_time = current_time
                    process_count = 0
                    
        except KeyboardInterrupt:
            print("\nStopping...")
            
        cap.release()
        print("\nFinal Counts:")
        print(f"Left: {self.counts['left']}")
        print(f"Right: {self.counts['right']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='n', help='Model size (n only for speed)')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='320x240', 
                       help='Resolution (e.g., 320x240, 640x480)')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    counter = HumanCounterZone(model_size=args.model)
    counter.run(camera_id=args.cam, resolution=(width, height))
