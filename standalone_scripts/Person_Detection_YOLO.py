#!/usr/bin/env python3
"""
Human Detection System for Raspberry Pi
Optimized for Pi 4/5 with 720p webcam
Uses YOLOv8 with performance optimizations
"""

import cv2
import numpy as np
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics (YOLOv8)...")
    import os
    os.system("pip3 install ultralytics opencv-python")
    from ultralytics import YOLO

import torch

class RaspberryPiHumanDetector:
    def __init__(self, model_size='n', use_threading=True):
        """
        Initialize human detector optimized for Raspberry Pi
        model_size: 'n' (nano - RECOMMENDED for Pi), 's' (small)
        use_threading: Enable threaded frame processing for better FPS
        """
        print(f"Loading YOLOv8{model_size} model for Raspberry Pi...")
        
        # Check for GPU availability
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print("✓ GPU (CUDA) detected! Using GPU acceleration.")
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            self.model = YOLO(f'yolov8{model_size}.pt')
            
            # Export to NCNN format for better Pi performance (optional but recommended)
            try:
                print("Exporting to NCNN format for Raspberry Pi optimization...")
                # Use half=True for FP16 optimization (faster on Pi)
                self.model.export(format='ncnn', half=True)
                ncnn_model_path = f'yolov8{model_size}_ncnn_model'
                self.model = YOLO(ncnn_model_path)
                print("✓ Using NCNN optimized model (FP16)")
            except Exception as e:
                print(f"Could not export to NCNN, using standard model: {e}")
        
        print("✓ Model loaded successfully\n")
        
        # Lower confidence for Pi to ensure detection
        self.confidence_threshold = 0.4
        
        # Performance settings
        self.use_threading = use_threading
        self.frame_skip = 2  # Process every Nth frame for speed
        self.frame_count = 0
        self.resize_factor = 0.5  # Resize frames for faster processing
        
        self.person_count = 0
        
        # Threading variables
        if self.use_threading:
            import threading
            self.lock = threading.Lock()
            self.latest_detections = []
            self.processing = False
    
    def detect_humans(self, camera_id=0, resolution=(640, 480), show_count=True):
        """
        Run real-time human detection optimized for Raspberry Pi
        Lower resolution recommended for better FPS on Pi
        """
        print("="*60)
        print("RASPBERRY PI HUMAN DETECTION SYSTEM")
        print("="*60)
        print(f"Model: YOLOv8 optimized for Raspberry Pi")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
        print(f"Frame skip: {self.frame_skip}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  '+' - Increase confidence threshold")
        print("  '-' - Decrease confidence threshold")
        print("  'f' - Toggle frame skip (performance)")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        # Set lower resolution for better performance on Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Disable auto-focus for stability (if supported)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        prev_time = time.time()
        fps = 0
        
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process every Nth frame for better performance
            if self.frame_count % self.frame_skip == 0:
                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), 
                                        fx=self.resize_factor, 
                                        fy=self.resize_factor)
                
                # Run detection
                results = self.model(small_frame, 
                                   conf=self.confidence_threshold, 
                                   classes=[0],  # Only detect persons
                                   verbose=False,
                                   imgsz=320,
                                   device=self.device)  # Lower input size for Pi
                
                detections = []
                self.person_count = 0
                
                for result in results:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Get box coordinates and scale back up
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Scale back to original frame size
                        x1 = int(x1 / self.resize_factor)
                        y1 = int(y1 / self.resize_factor)
                        x2 = int(x2 / self.resize_factor)
                        y2 = int(y2 / self.resize_factor)
                        
                        confidence = float(box.conf[0])
                        
                        self.person_count += 1
                        detections.append((x1, y1, x2, y2, confidence))
            
            # Draw detections on full frame
            person_num = 0
            for x1, y1, x2, y2, confidence in detections:
                person_num += 1
                
                # Draw blue box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Label
                label = f"Person {person_num} ({confidence:.0%})"
                
                # Get label size
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw blue background
                cv2.rectangle(frame, 
                            (x1, y1 - label_height - 10), 
                            (x1 + label_width + 5, y1), 
                            (255, 0, 0), 
                            cv2.FILLED)
                
                # Draw white text
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time + 0.001)
            prev_time = current_time
            
            # Show compact info overlay
            cv2.putText(frame, f"Humans: {self.person_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show on screen (comment out if running headless)
            cv2.imshow('Human Detection - Pi', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
                print(f"Threshold: {self.confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                print(f"Threshold: {self.confidence_threshold:.2f}")
            elif key == ord('f'):
                self.frame_skip = 3 - self.frame_skip  # Toggle between 1 and 2
                print(f"Frame skip: {self.frame_skip}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nSession ended. Last count: {self.person_count} humans detected")


class RaspberryPiHumanDetectorHeadless:
    """
    Headless version for Raspberry Pi without display
    Logs detections and can trigger actions
    """
    def __init__(self, model_size='n'):
        print(f"Loading YOLOv8{model_size} model (headless mode)...")
        
        # Check for GPU availability
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print("✓ GPU (CUDA) detected! Using GPU acceleration.")
            
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("✓ Model loaded\n")
        
        self.confidence_threshold = 0.4
        self.person_count = 0
    
    def detect_and_log(self, camera_id=0, duration_minutes=None, log_file="detections.log"):
        """
        Run detection without display, log results
        duration_minutes: Run for specific duration (None = indefinite)
        """
        print("="*60)
        print("HEADLESS DETECTION MODE")
        print("="*60)
        print(f"Logging to: {log_file}")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        
        with open(log_file, 'a') as log:
            log.write(f"\n--- Session started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process every 3rd frame
                    if frame_count % 3 == 0:
                        start_time = time.time()
                        # Resize for speed
                        small_frame = cv2.resize(frame, (320, 240))
                        
                        results = self.model(small_frame, 
                                           conf=self.confidence_threshold,
                                           classes=[0],
                                           verbose=False,
                                           imgsz=320,
                                           device=self.device)
                        
                        current_count = 0
                        for result in results:
                            current_count = len(result.boxes)
                        
                        # Log if count changed
                        if current_count != self.person_count:
                            end_time = time.time()
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            msg = f"{timestamp} - Humans detected: {current_count} \t(Processing time: {end_time - start_time:.2f}s)"
                            print(msg)
                            log.write(msg + "\n")
                            log.flush()
                            
                            self.person_count = current_count
                            
                            # TRIGGER ACTIONS HERE
                            # Example: Turn on lights, send notification, etc.
                            # if current_count > 0:
                            #     trigger_action()
                    
                    # Check duration
                    if duration_minutes:
                        elapsed = (time.time() - start_time) / 60
                        if elapsed >= duration_minutes:
                            break
                    
                    time.sleep(0.03)  # Small delay to prevent CPU overload
                    
            except KeyboardInterrupt:
                print("\nStopping detection...")
            
            log.write(f"--- Session ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        cap.release()
        print(f"\nSession ended. Check {log_file} for full log")


# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Raspberry Pi Human Detection')
    parser.add_argument('--model', type=str, default='n', 
                       help='Model size: n (nano-fast), s (small)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run without display (log only)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (e.g., 640x480, 1280x720)')
    parser.add_argument('--duration', type=int, default=2,
                       help='Run duration in minutes (headless mode only)')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Note: On Raspberry Pi, 'n' (Nano) model is recommended for best performance
    # 's' (Small) is more accurate but significantly slower
    
    print("\nRASPBERRY PI DETECTION SYSTEM")
    print("="*60)
    print(f"Model: YOLOv8{args.model}")
    print(f"Mode: {'Headless' if args.headless else 'Display'}")
    print(f"Resolution: {width}x{height}")
    print("="*60 + "\n")
    
    if args.headless:
        detector = RaspberryPiHumanDetectorHeadless(model_size=args.model)
        detector.detect_and_log(camera_id=0, duration_minutes=args.duration)
    else:
        detector = RaspberryPiHumanDetector(model_size=args.model)
        detector.detect_humans(camera_id=0, resolution=(width, height))