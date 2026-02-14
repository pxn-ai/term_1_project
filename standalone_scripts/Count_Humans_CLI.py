#!/usr/bin/env python3
"""
Human Counting System with Direction Detection (CLI Version)
Counts people moving Left <-> Right
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

class HumanCounterCLI:
    def __init__(self, model_size='n', line_position=0.5):
        """
        Initialize Human Counter
        model_size: 'n' (nano) or 's' (small)
        line_position: 0.0 to 1.0 (percentage of screen width for the counting line)
        """
        print(f"Loading YOLOv8{model_size} for Tracking...")
        
        # Check for GPU
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print("✓ GPU detected! Using CUDA acceleration.")
        
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # Tracking state
        self.track_history = defaultdict(lambda: [])
        self.counts = {'left': 0, 'right': 0}
        self.crossed_ids = set() # Keep track of IDs that have already been counted
        
        # Line configuration
        self.line_position = line_position # 0.5 = center
        self.line_x = 0 # Will be calculated based on resolution
        
        # Visual settings
        self.trace_length = 30
        
    def run(self, camera_id=0, resolution=(640, 480)):
        print(f"Starting Camera {camera_id} at {resolution}")
        print("Format: (Human count detected | total_went_left | total_went_right | FPS)")
        print("-" * 60)
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Calculate line position
        self.line_x = int(resolution[0] * self.line_position)
        
        last_print_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Force resize to target resolution for performance
                if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                    frame = cv2.resize(frame, resolution)
                
                # Run YOLOv8 tracking
                # persist=True is crucial for tracking objects across frames
                results = self.model.track(frame, 
                                         persist=True, 
                                         classes=[0], # 0 = person
                                         conf=0.3,
                                         verbose=False,
                                         device=self.device)
                
                current_human_count = 0
                
                if results[0].boxes.id is not None:
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    current_human_count = len(track_ids)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        center = (float(x), float(y))
                        
                        # Store history
                        track = self.track_history[track_id]
                        track.append(center)
                        if len(track) > self.trace_length:
                            track.pop(0)
                            
                        # Check for line crossing
                        if track_id not in self.crossed_ids and len(track) > 2:
                            # Get previous and current x positions
                            prev_x = track[-2][0]
                            curr_x = track[-1][0]
                            
                            # Check if crossed the line
                            # Moving Right: prev_x < line < curr_x
                            if prev_x < self.line_x and curr_x >= self.line_x:
                                self.counts['right'] += 1
                                self.crossed_ids.add(track_id)
                                
                            # Moving Left: prev_x > line > curr_x
                            elif prev_x > self.line_x and curr_x <= self.line_x:
                                self.counts['left'] += 1
                                self.crossed_ids.add(track_id)

                # Print status every second
                current_time = time.time()
                elapsed_time = current_time - last_print_time
                
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    print(f"( In the Frame : {current_human_count} | To Left :{self.counts['left']} | To Right : {self.counts['right']} | = | {fps:.1f} FPS )")
                    last_print_time = current_time
                    frame_count = 0
                    
        except KeyboardInterrupt:
            print("\nStopping...")
            
        cap.release()
        print("\nFinal Counts:")
        print(f"Left: {self.counts['left']}")
        print(f"Right: {self.counts['right']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='n', help='Model size (n, s, m)')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    args = parser.parse_args()
    
    counter = HumanCounterCLI(model_size=args.model)
    counter.run(camera_id=args.cam)
