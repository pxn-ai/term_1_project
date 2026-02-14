#!/usr/bin/env python3
"""
Human Counter with Video Streaming API
Stream processed video to web browser with counting overlay
Optimized for Raspberry Pi 4
"""

import cv2
import numpy as np
import time
from collections import defaultdict
import torch
from flask import Flask, Response, jsonify, render_template_string
from flask_cors import CORS
import threading
import io

try:
    from ultralytics import YOLO
    import psutil
except ImportError:
    print("Installing required packages...")
    import os
    os.system("pip3 install ultralytics psutil flask flask-cors")
    from ultralytics import YOLO
    import psutil

app = Flask(__name__)
CORS(app)

class VideoStreamCounter:
    def __init__(self, model_size='n', line_position=0.5, skip_frames=2, 
                 roi_enabled=True, roi_width_percent=0.7):
        """Initialize counter with streaming capability"""
        print(f"[INIT] Loading YOLOv8{model_size}...")
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print("[GPU] CUDA detected!")
        
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # Performance settings
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.roi_enabled = roi_enabled
        self.roi_width_percent = roi_width_percent
        
        # Tracking
        self.track_history = defaultdict(lambda: [])
        self.counts = {'left': 0, 'right': 0}
        self.crossed_ids = set()
        self.line_position = line_position
        self.line_x = 0
        self.trace_length = 10
        
        # Streaming
        self.current_frame = None
        self.output_frame = None
        self.lock = threading.Lock()
        self.is_running = False
        
        # Stats
        self.current_human_count = 0
        self.fps = 0
        self.cpu_percent = 0
        self.temp = 0
        
    def get_cpu_temp(self):
        """Get CPU temperature"""
        try:
            temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
            return temp
        except:
            return 0
    
    def draw_overlays(self, frame, boxes, track_ids):
        """Draw detection boxes, lines, and stats on frame"""
        height, width = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (self.line_x, 0), (self.line_x, height), (0, 255, 255), 3)
        cv2.putText(frame, "COUNT LINE", (self.line_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw ROI boundaries if enabled
        if self.roi_enabled:
            roi_width = int(width * self.roi_width_percent)
            roi_x1 = max(0, self.line_x - roi_width // 2)
            roi_x2 = min(width, self.line_x + roi_width // 2)
            cv2.rectangle(frame, (roi_x1, 0), (roi_x2, height), (100, 100, 100), 1)
        
        # Draw detection boxes and tracks
        if boxes is not None and track_ids is not None:
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                # Box color based on position relative to line
                color = (0, 255, 0) if x < self.line_x else (255, 0, 0)
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw track history
                track = self.track_history[track_id]
                if len(track) > 1:
                    points = np.array(track, dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)
        
        # Draw stats panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Stats text
        y_offset = 25
        cv2.putText(frame, f"Humans in Frame: {self.current_human_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Moved LEFT: {self.counts['left']}  |  Moved RIGHT: {self.counts['right']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}  |  CPU: {self.cpu_percent:.1f}%  |  Temp: {self.temp:.1f}C", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, camera_id=0, resolution=(640, 480), conf_threshold=0.4):
        """Process video and update frames"""
        print(f"[START] Camera {camera_id} at {resolution}")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.line_x = int(resolution[0] * self.line_position)
        
        # ROI calculation
        if self.roi_enabled:
            roi_width = int(resolution[0] * self.roi_width_percent)
            roi_x1 = max(0, self.line_x - roi_width // 2)
            roi_x2 = min(resolution[0], self.line_x + roi_width // 2)
        else:
            roi_x1, roi_x2 = 0, resolution[0]
        
        last_fps_time = time.time()
        frame_count = 0
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_counter += 1
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                self.fps = frame_count / (current_time - last_fps_time)
                self.cpu_percent = psutil.cpu_percent(interval=0)
                self.temp = self.get_cpu_temp()
                frame_count = 0
                last_fps_time = current_time
            
            # Resize if needed
            if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                frame = cv2.resize(frame, resolution)
            
            boxes_to_draw = None
            ids_to_draw = None
            
            # Process frame (with skipping)
            if self.frame_counter % self.skip_frames == 0:
                # Extract ROI
                if self.roi_enabled:
                    roi_frame = frame[:, roi_x1:roi_x2].copy()
                else:
                    roi_frame = frame
                
                # Run YOLO
                results = self.model.track(
                    roi_frame,
                    persist=True,
                    classes=[0],
                    conf=conf_threshold,
                    iou=0.5,
                    verbose=False,
                    device=self.device,
                    imgsz=min(resolution),
                    half=False
                )
                
                self.current_human_count = 0
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    self.current_human_count = len(track_ids)
                    
                    # Adjust coordinates for ROI
                    if self.roi_enabled:
                        boxes[:, 0] += roi_x1
                    
                    boxes_to_draw = boxes
                    ids_to_draw = track_ids
                    
                    # Track and count crossings
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        center = (float(x), float(y))
                        
                        track = self.track_history[track_id]
                        track.append(center)
                        if len(track) > self.trace_length:
                            track.pop(0)
                        
                        # Check crossings
                        if track_id not in self.crossed_ids and len(track) >= 2:
                            prev_x = track[-2][0]
                            curr_x = track[-1][0]
                            
                            if prev_x < self.line_x <= curr_x:
                                self.counts['right'] += 1
                                self.crossed_ids.add(track_id)
                            elif prev_x > self.line_x >= curr_x:
                                self.counts['left'] += 1
                                self.crossed_ids.add(track_id)
            
            # Draw overlays
            output = self.draw_overlays(frame.copy(), boxes_to_draw, ids_to_draw)
            
            # Update output frame (thread-safe)
            with self.lock:
                self.output_frame = output.copy()
        
        cap.release()
        print("[STOP] Video processing stopped")
    
    def get_frame(self):
        """Get current frame for streaming"""
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False

# Global counter instance
counter = None

def generate_frames():
    """Generate frames for MJPEG streaming"""
    global counter
    while True:
        if counter is None:
            time.sleep(0.1)
            continue
        
        frame = counter.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    global counter
    if counter is None:
        return jsonify({'error': 'Counter not started'}), 503
    
    return jsonify({
        'humans_in_frame': counter.current_human_count,
        'moved_left': counter.counts['left'],
        'moved_right': counter.counts['right'],
        'net_flow': counter.counts['right'] - counter.counts['left'],
        'total_crossings': counter.counts['left'] + counter.counts['right'],
        'fps': round(counter.fps, 1),
        'cpu_percent': round(counter.cpu_percent, 1),
        'temperature': round(counter.temp, 1)
    })

@app.route('/control/<action>')
def control(action):
    """Control endpoints"""
    global counter
    
    if action == 'reset':
        if counter:
            counter.counts = {'left': 0, 'right': 0}
            counter.crossed_ids.clear()
        return jsonify({'status': 'reset', 'message': 'Counts reset to zero'})
    
    return jsonify({'error': 'Unknown action'}), 400

@app.route('/')
def index():
    """Web interface"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Human Counter Video Stream</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            font-family: Arial, sans-serif;
            color: white;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .video-container {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        img {
            width: 100%;
            border-radius: 5px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }
        .controls {
            text-align: center;
            margin-top: 20px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #45a049;
        }
        .warning {
            color: #ff6b6b;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Human Counter Video Stream</h1>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="humans">-</div>
                <div class="stat-label">Humans in Frame</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="left">-</div>
                <div class="stat-label">Moved Left ←</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="right">-</div>
                <div class="stat-label">Moved Right →</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">-</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="cpu">-</div>
                <div class="stat-label">CPU Usage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="temp">-</div>
                <div class="stat-label">Temperature</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="resetCounts()">Reset Counts</button>
        </div>
    </div>
    
    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('humans').textContent = data.humans_in_frame;
                    document.getElementById('left').textContent = data.moved_left;
                    document.getElementById('right').textContent = data.moved_right;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('cpu').textContent = data.cpu_percent.toFixed(1) + '%';
                    
                    const tempEl = document.getElementById('temp');
                    tempEl.textContent = data.temperature.toFixed(1) + '°C';
                    if (data.temperature > 75) {
                        tempEl.classList.add('warning');
                    } else {
                        tempEl.classList.remove('warning');
                    }
                })
                .catch(err => console.error('Stats error:', err));
        }
        
        function resetCounts() {
            fetch('/control/reset')
                .then(response => response.json())
                .then(data => alert(data.message));
        }
        
        // Update stats every 2 seconds
        setInterval(updateStats, 2000);
        updateStats();
    </script>
</body>
</html>
    ''')

def start_counter(camera_id=0, resolution=(640, 480), model_size='n', 
                  skip_frames=2, conf_threshold=0.4):
    """Start the counter in a background thread"""
    global counter
    counter = VideoStreamCounter(
        model_size=model_size,
        line_position=0.5,
        skip_frames=skip_frames,
        roi_enabled=True,
        roi_width_percent=0.7
    )
    
    # Run video processing in background thread
    thread = threading.Thread(
        target=counter.process_video,
        args=(camera_id, resolution, conf_threshold),
        daemon=True
    )
    thread.start()
    print(f"[STREAM] Video processing started in background")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    parser.add_argument('--model', type=str, default='n', help='Model size')
    parser.add_argument('--skip', type=int, default=2, help='Skip frames')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP')
    parser.add_argument('--port', type=int, default=5000, help='Port')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HUMAN COUNTER VIDEO STREAMING API")
    print("="*80)
    print(f"Starting video stream on http://{args.host}:{args.port}")
    print(f"Camera: {args.cam}, Resolution: {args.width}x{args.height}")
    print(f"Model: YOLOv8{args.model}, Skip frames: {args.skip}")
    print("="*80)
    print("\nEndpoints:")
    print(f"  Web Interface: http://<pi-ip>:{args.port}/")
    print(f"  Video Stream:  http://<pi-ip>:{args.port}/video_feed")
    print(f"  Statistics:    http://<pi-ip>:{args.port}/stats")
    print(f"  Reset Counts:  http://<pi-ip>:{args.port}/control/reset")
    print("="*80)
    
    # Start counter
    start_counter(
        camera_id=args.cam,
        resolution=(args.width, args.height),
        model_size=args.model,
        skip_frames=args.skip,
        conf_threshold=args.conf
    )
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)