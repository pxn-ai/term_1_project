#!/usr/bin/env python3
"""
Human Counting System - Zone Based Logic (GUI Version)
Counts people based on Zone Transitions (Left Zone <-> Right Zone)
Optimized for Raspberry Pi with Visual Feedback and Improved Accuracy
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

class HumanCounterZoneGUI:
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
        
    def run(self, camera_id=0, resolution=(640, 480)):
        """
        GUI Version with Visual Feedback
        """
        print(f"Starting Camera {camera_id} at {resolution}")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Calculate line position
        self.line_x = int(resolution[0] * self.line_position)
        
        prev_time = time.time()
        frame_count = 0
        
        # PERFORMANCE: Frame skipping
        frame_skip = 2  # Process every 2nd frame
        
        # Buffer zone to prevent jitter counting (pixels)
        buffer = 20 
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Force resize to target resolution for performance
                if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                    frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
                
                # Draw the counting line
                cv2.line(frame, (self.line_x, 0), (self.line_x, resolution[1]), (0, 255, 255), 2)
                
                # Only run inference on skipped frames
                if frame_count % frame_skip == 0:
                    # PERFORMANCE: Higher conf threshold to reduce false positives
                    results = self.model.track(frame, 
                                             persist=True, 
                                             classes=[0],
                                             conf=0.5,  # Increased to 0.5 for better accuracy
                                             verbose=False,
                                             device=self.device,
                                             imgsz=320,  # Reduced inference size
                                             iou=0.5,
                                             max_det=10)
                    
                    active_ids = []
                    
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.cpu()
                        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        
                        active_ids = track_ids
                        
                        for box, box_coords, track_id in zip(boxes, boxes_xyxy, track_ids):
                            x, y, w, h = box
                            x1, y1, x2, y2 = box_coords
                            center_x = float(x)
                            
                            # Filter small boxes (noise)
                            # If box area is less than 1.5% of screen, ignore it
                            if w * h < (resolution[0] * resolution[1] * 0.015): 
                                continue

                            # Draw bounding box
                            color = (255, 0, 0) # Blue
                            if track_id in self.counted_ids:
                                color = (0, 255, 0) # Green if counted
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
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
                            if track_id not in self.person_start_zone:
                                self.person_start_zone[track_id] = current_zone
                            
                            elif track_id not in self.counted_ids:
                                start_zone = self.person_start_zone[track_id]
                                
                                # Check for transition
                                if start_zone == 'left' and current_zone == 'right':
                                    self.counts['right'] += 1
                                    self.counted_ids.add(track_id)
                                    # Visual flash
                                    cv2.line(frame, (self.line_x, 0), (self.line_x, resolution[1]), (0, 255, 0), 4)
                                    
                                elif start_zone == 'right' and current_zone == 'left':
                                    self.counts['left'] += 1
                                    self.counted_ids.add(track_id)
                                    # Visual flash
                                    cv2.line(frame, (self.line_x, 0), (self.line_x, resolution[1]), (0, 0, 255), 4)

                    # Periodic cleanup
                    if frame_count % 100 == 0:
                        active_set = set(active_ids)
                        keys_to_remove = [k for k in self.person_start_zone.keys() if k not in active_set]
                        for k in keys_to_remove:
                            del self.person_start_zone[k]

                # Display Stats
                # Background
                cv2.rectangle(frame, (0, 0), (resolution[0], 40), (0, 0, 0), -1)
                
                # Counts
                cv2.putText(frame, f"Left: {self.counts['left']}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Right: {self.counts['right']}", (150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 0.001)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (resolution[0] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Human Counter Zone GUI", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping...")
            
        cap.release()
        cv2.destroyAllWindows()
        print("\nFinal Counts:")
        print(f"Left: {self.counts['left']}")
        print(f"Right: {self.counts['right']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='n', help='Model size (n only for speed)')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', 
                       help='Resolution (e.g., 320x240, 640x480)')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    counter = HumanCounterZoneGUI(model_size=args.model)
    counter.run(camera_id=args.cam, resolution=(width, height))