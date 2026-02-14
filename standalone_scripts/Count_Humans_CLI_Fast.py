#!/usr/bin/env python3
"""
Human Counting System with Direction Detection (CLI Version - OPTIMIZED)
Counts people moving Left <-> Right
Optimized for Raspberry Pi - Headless Mode
PERFORMANCE ENHANCEMENTS:
- Lower resolution (320x240)
- Frame skipping (process every 2nd frame)
- Reduced model input size (imgsz=320)
- Minimal tracking history
- NCNN export for CPU optimization
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

class HumanCounterCLI:
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
        
        # Tracking state
        self.track_history = defaultdict(lambda: [])
        self.counts = {'left': 0, 'right': 0}
        self.crossed_ids = set()
        
        # Line configuration
        self.line_position = line_position
        self.line_x = 0
        
        # PERFORMANCE: Minimal tracking history
        self.trace_length = 5  # Reduced from 30
        
    def run(self, camera_id=0, resolution=(320, 240)):
        """
        OPTIMIZED: Default to 320x240 for maximum speed
        """
        print(f"Starting Camera {camera_id} at {resolution}")
        print("Format: (Human count detected | total_went_left | total_went_right | FPS)")
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
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    current_human_count = len(track_ids)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        center = (float(x), float(y))
                        
                        track = self.track_history[track_id]
                        track.append(center)
                        if len(track) > self.trace_length:
                            track.pop(0)
                            
                        # Check for line crossing
                        if track_id not in self.crossed_ids and len(track) >= 2:
                            prev_x = track[-2][0]
                            curr_x = track[-1][0]
                            
                            # Moving Right
                            if prev_x < self.line_x and curr_x >= self.line_x:
                                self.counts['right'] += 1
                                self.crossed_ids.add(track_id)
                                
                            # Moving Left
                            elif prev_x > self.line_x and curr_x <= self.line_x:
                                self.counts['left'] += 1
                                self.crossed_ids.add(track_id)

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
    
    counter = HumanCounterCLI(model_size=args.model)
    counter.run(camera_id=args.cam, resolution=(width, height))
