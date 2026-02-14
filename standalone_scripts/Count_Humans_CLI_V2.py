#!/usr/bin/env python3
"""
Optimized Human Counting System for Raspberry Pi 4
Counts people moving Left <-> Right with maximum performance
All optimizations applied for Pi 4 Model B
"""
# from Claude.ai - (working)

import cv2
import numpy as np
import time
from collections import defaultdict
import torch
import sys
import os
from gpiozero import LED

try:
    from ultralytics import YOLO
    import psutil
except ImportError:
    print("Installing required packages...")
    os.system("pip3 install ultralytics psutil")
    from ultralytics import YOLO
    import psutil

class count_register():

    def __init__(self, count : int = 0 ):
        self.count = count
    
    def increment(self, amount : int = 1):
        self.count += amount

    def decrement(self, amount : int = 1):
        self.count -= amount
        if self.count < 0:
            self.count = 0

    def get_count(self):
        return self.count

    def control_power(self, led: LED):
        if self.count > 0:
            led.on()
        else:
            led.off()

class OptimizedHumanCounter:
    def __init__(self, model_size='n', line_position=0.5, use_onnx=False, 
                 skip_frames=2, roi_enabled=True, roi_width_percent=0.7):
        """
        Initialize Optimized Human Counter for Raspberry Pi 4
        
        Args:
            model_size: 'n' (nano) recommended for Pi 4
            line_position: 0.0 to 1.0 (percentage of screen width)
            use_onnx: Use ONNX runtime for better performance (requires export first)
            skip_frames: Process every Nth frame (2-3 recommended)
            roi_enabled: Only process region around counting line
            roi_width_percent: Width of ROI (0.7 = 70% of frame)
        """
        print(f"[INIT] Loading YOLOv8{model_size} for Raspberry Pi 4...")
        
        # Device detection
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print("[GPU] CUDA detected! Using GPU acceleration.")
        else:
            print("[CPU] Using CPU - optimizations enabled")
        
        # Load model
        model_path = f'yolov8{model_size}.onnx' if use_onnx else f'yolov8{model_size}.pt'
        if use_onnx and not os.path.exists(model_path):
            print(f"[ONNX] {model_path} not found, using .pt model")
            print("[ONNX] To create ONNX: model.export(format='onnx')")
            model_path = f'yolov8{model_size}.pt'
        
        self.model = YOLO(model_path)
        print(f"[MODEL] Loaded: {model_path}")
        
        # Performance optimizations
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.roi_enabled = roi_enabled
        self.roi_width_percent = roi_width_percent
        
        # Tracking state
        self.track_history = defaultdict(lambda: [])
        self.counts = {'left': 0, 'right': 0}
        self.crossed_ids = set()
        
        # Line configuration
        self.line_position = line_position
        self.line_x = 0
        
        # Reduced trace length for performance
        self.trace_length = 10
        
        # Cleanup intervals
        self.cleanup_interval = 300  # Cleanup every 300 frames (~5 sec at 60fps)
        
        # Performance monitoring
        self.last_cleanup_time = time.time()
        
    def get_cpu_temp(self):
        """Get CPU temperature"""
        try:
            temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
            return temp
        except:
            return 0
    
    def check_system_health(self):
        """Monitor system health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            temp = self.get_cpu_temp()
            
            # Warning thresholds
            if temp > 75:
                print(f"[WARNING] High temperature: {temp:.1f}°C - Consider cooling")
            if cpu_percent > 90:
                print(f"[WARNING] High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                print(f"[WARNING] High memory usage: {memory.percent:.1f}%")
                
            return temp, cpu_percent, memory.percent
        except:
            return 0, 0, 0
    
    def cleanup_old_tracks(self, active_ids):
        """Remove inactive tracking data to save memory"""
        # Clean crossed IDs
        self.crossed_ids = self.crossed_ids & active_ids
        
        # Clean track history
        inactive_ids = set(self.track_history.keys()) - active_ids
        for tid in inactive_ids:
            del self.track_history[tid]
        
        if inactive_ids:
            print(f"[CLEANUP] Removed {len(inactive_ids)} inactive tracks")
    
    def run(self, camera_id=0, resolution=(416, 416), conf_threshold=0.4, 
            show_system_stats=True):
        """
        Run the optimized counter
        
        Args:
            camera_id: Camera index (0 for default)
            resolution: (width, height) - (416, 416) recommended for Pi 4
            conf_threshold: Detection confidence (0.4 recommended)
            show_system_stats: Show CPU, RAM, temp stats
        """
        print(f"[START] Camera {camera_id} at {resolution}")
        print(f"[CONFIG] Skip frames: {self.skip_frames}, ROI: {self.roi_enabled}, Conf: {conf_threshold}")
        print("-" * 80)
        print("Format: Humans | Left→ | ←Right | FPS | CPU% | RAM% | Temp°C")
        print("-" * 80)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        
        # Calculate line and ROI positions
        self.line_x = int(resolution[0] * self.line_position)
        
        if self.roi_enabled:
            roi_width = int(resolution[0] * self.roi_width_percent)
            roi_x1 = max(0, self.line_x - roi_width // 2)
            roi_x2 = min(resolution[0], self.line_x + roi_width // 2)
            print(f"[ROI] Processing region: x={roi_x1} to x={roi_x2} (width: {roi_x2-roi_x1}px)")
        else:
            roi_x1, roi_x2 = 0, resolution[0]
        
        # Performance tracking
        last_print_time = time.time()
        processed_frames = 0
        total_frames = 0
        
        try:
            while True:
                person_counter.control_power(led)

                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                total_frames += 1
                self.frame_counter += 1
                
                # Skip frames for performance
                if self.frame_counter % self.skip_frames != 0:
                    continue
                
                processed_frames += 1
                
                # Resize frame if needed
                if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                    frame = cv2.resize(frame, resolution)
                
                # Extract ROI if enabled
                if self.roi_enabled:
                    roi_frame = frame[:, roi_x1:roi_x2].copy()
                else:
                    roi_frame = frame
                
                # Run YOLO tracking with optimized parameters
                results = self.model.track(
                    roi_frame,
                    persist=True,
                    classes=[0],  # Person only
                    conf=conf_threshold,
                    iou=0.5,
                    verbose=False,
                    device=self.device,
                    imgsz=min(resolution),  # Use smaller dimension
                    half=False  # Pi 4 doesn't support FP16 well
                )
                
                current_human_count = 0
                active_ids = set()
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    current_human_count = len(track_ids)
                    active_ids = set(track_ids)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        
                        # Adjust x coordinate if using ROI
                        if self.roi_enabled:
                            x = float(x) + roi_x1
                        else:
                            x = float(x)
                        
                        center = (x, float(y))
                        
                        # Store history
                        track = self.track_history[track_id]
                        track.append(center)
                        if len(track) > self.trace_length:
                            track.pop(0)
                        
                        # Check for line crossing
                        if track_id not in self.crossed_ids and len(track) >= 2:
                            prev_x = track[-2][0]
                            curr_x = track[-1][0]
                            
                            # Moving Right: crossed from left to right
                            if prev_x < self.line_x <= curr_x:
                                self.counts['right'] += 1
                                self.crossed_ids.add(track_id)
                                print(f"[CROSS] ID {track_id} moved RIGHT →")
                                person_counter.increment()
                            
                            # Moving Left: crossed from right to left
                            elif prev_x > self.line_x >= curr_x:
                                self.counts['left'] += 1
                                self.crossed_ids.add(track_id)
                                print(f"[CROSS] ID {track_id} moved LEFT ←")
                                person_counter.decrement()
                
                # Periodic cleanup
                if processed_frames % self.cleanup_interval == 0 and active_ids:
                    self.cleanup_old_tracks(active_ids)
                
                # Print status every second
                current_time = time.time()
                elapsed_time = current_time - last_print_time
                
                if elapsed_time >= 1.0:
                    fps = processed_frames / elapsed_time
                    actual_fps = total_frames / elapsed_time
                    
                    if show_system_stats:
                        temp, cpu, ram = self.check_system_health()
                        print(f"[ {current_human_count:2d} humans | "
                              f"←{self.counts['left']:3d} | "
                              f"{self.counts['right']:3d}→ | "
                              f"{fps:4.1f} FPS ({actual_fps:.1f} actual) | "
                              f"CPU:{cpu:4.1f}% | RAM:{ram:4.1f}% | {temp:4.1f}°C ]"
                              f"\t| In the Classroom : {person_counter.get_count()} persons")
                    else:
                        print(f"[ {current_human_count:2d} humans | "
                              f"←{self.counts['left']:3d} | "
                              f"{self.counts['right']:3d}→ | "
                              f"{fps:4.1f} FPS ({actual_fps:.1f} actual) ]")
                    
                    last_print_time = current_time
                    processed_frames = 0
                    total_frames = 0
        
        except KeyboardInterrupt:
            print("\n[STOP] Stopping counter...")
        
        finally:
            cap.release()
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)
            print(f"Total moved LEFT  ←: {self.counts['left']}")
            print(f"Total moved RIGHT →: {self.counts['right']}")
            print(f"Net flow (R-L):      {self.counts['right'] - self.counts['left']}")
            print(f"Total crossings:     {self.counts['left'] + self.counts['right']}")
            print("="*80)

def export_to_onnx(model_size='n'):
    """Helper function to export model to ONNX format"""
    print(f"[EXPORT] Exporting YOLOv8{model_size} to ONNX format...")
    model = YOLO(f'yolov8{model_size}.pt')
    model.export(format='onnx', imgsz=416)
    print(f"[EXPORT] Done! Model saved as yolov8{model_size}.onnx")

def start_model():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Human Counter for Raspberry Pi 4')
    parser.add_argument('--model', type=str, default='n', 
                        help='Model size: n (nano), s (small), m (medium)')
    parser.add_argument('--cam', type=int, default=0, 
                        help='Camera ID (default: 0)')
    parser.add_argument('--width', type=int, default=416, 
                        help='Frame width (default: 416)')
    parser.add_argument('--height', type=int, default=416, 
                        help='Frame height (default: 416)')
    parser.add_argument('--conf', type=float, default=0.4, 
                        help='Detection confidence threshold (default: 0.4)')
    parser.add_argument('--skip', type=int, default=2, 
                        help='Process every Nth frame (default: 2)')
    parser.add_argument('--line', type=float, default=0.5, 
                        help='Counting line position 0.0-1.0 (default: 0.5 = center)')
    parser.add_argument('--no-roi', action='store_true', 
                        help='Disable ROI optimization')
    parser.add_argument('--roi-width', type=float, default=0.7, 
                        help='ROI width percentage (default: 0.7)')
    parser.add_argument('--onnx', action='store_true', 
                        help='Use ONNX runtime (must export first)')
    parser.add_argument('--export-onnx', action='store_true', 
                        help='Export model to ONNX and exit')
    parser.add_argument('--no-stats', action='store_true', 
                        help='Hide system statistics')
    
    args = parser.parse_args()
    
    # Export ONNX if requested
    if args.export_onnx:
        export_to_onnx(args.model)
        sys.exit(0)
    
    # Create and run counter
    print("\n" + "="*80)
    print("OPTIMIZED HUMAN COUNTER FOR RASPBERRY PI 4")
    print("="*80)
    
    counter = OptimizedHumanCounter(
        model_size=args.model,
        line_position=args.line,
        use_onnx=args.onnx,
        skip_frames=args.skip,
        roi_enabled=not args.no_roi,
        roi_width_percent=args.roi_width
    )
    
    counter.run(
        camera_id=args.cam,
        resolution=(args.width, args.height),
        conf_threshold=args.conf,
        show_system_stats=not args.no_stats
    )


if __name__ == "__main__":
    # Person counter (in the classroom)
    led = LED(17)   # act as the power in the classroom
    led.off()
    
    person_counter = count_register()
    start_model()
