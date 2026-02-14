#!/usr/bin/env python3
"""
Human Counting System with Direction Detection
Counts people moving Left <-> Right
Optimized for Raspberry Pi
"""

import cv2
import numpy as np
import time
from collections import defaultdict
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import os
    os.system("pip3 install ultralytics")
    from ultralytics import YOLO

class HumanCounter:
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
        
    def run(self, camera_id=0, resolution=(1280, 720)):
        print(f"Starting Camera {camera_id} at {resolution}")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Calculate line position
        self.line_x = int(resolution[0] * self.line_position)
        
        prev_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                
                # Visualize the counting line
                cv2.line(frame, (self.line_x, 0), (self.line_x, resolution[1]), (0, 255, 255), 2)
                cv2.putText(frame, "Line", (self.line_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                if results[0].boxes.id is not None:
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    for box, box_coords, track_id in zip(boxes, boxes_xyxy, track_ids):
                        x, y, w, h = box
                        x1, y1, x2, y2 = box_coords
                        
                        # Draw blue box around human
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
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
                                # Visual feedback
                                cv2.line(frame, (self.line_x, 0), (self.line_x, resolution[1]), (0, 255, 0), 4)
                                
                            # Moving Left: prev_x > line > curr_x
                            elif prev_x > self.line_x and curr_x <= self.line_x:
                                self.counts['left'] += 1
                                self.crossed_ids.add(track_id)
                                # Visual feedback
                                cv2.line(frame, (self.line_x, 0), (self.line_x, resolution[1]), (0, 0, 255), 4)

                        # Draw tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                
                # Display Counts
                # Background for text
                cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
                
                # Left Count
                cv2.putText(frame, f"<- LEFT: {self.counts['left']}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Right Count
                cv2.putText(frame, f"RIGHT ->: {self.counts['right']}", (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                # Use actual frame width for positioning
                h, w = frame.shape[:2]
                # Draw black background for FPS visibility
                cv2.rectangle(frame, (w - 130, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Human Direction Counter", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Stopping...")
            
        cap.release()
        cv2.destroyAllWindows()
        print("\nFinal Counts:")
        print(f"Left: {self.counts['left']}")
        print(f"Right: {self.counts['right']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='n', help='Model size (n, s, m)')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    args = parser.parse_args()
    
    counter = HumanCounter(model_size=args.model)
    counter.run(camera_id=args.cam)
