#!/usr/bin/env python3
"""
Raspberry Pi System Monitor API
Run this on your Raspberry Pi
Access from laptop: http://<pi-ip>:7000/stats
"""

from flask import Flask, jsonify
from flask_cors import CORS
import psutil
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

def get_cpu_temp():
    """Get CPU temperature"""
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
        return round(temp, 1)
    except:
        return None

def get_gpu_temp():
    """Get GPU temperature using vcgencmd"""
    try:
        temp = os.popen('vcgencmd measure_temp').readline()
        return float(temp.replace("temp=", "").replace("'C\n", ""))
    except:
        return None

def get_gpu_mem():
    """Get GPU memory usage"""
    try:
        mem = os.popen('vcgencmd get_mem gpu').readline()
        return mem.replace("gpu=", "").strip()
    except:
        return None

def get_throttle_status():
    """Check if Pi is being throttled"""
    try:
        throttle = os.popen('vcgencmd get_throttled').readline()
        return throttle.strip()
    except:
        return None

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get all system statistics"""
    
    # CPU Usage
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_avg = psutil.cpu_percent(interval=0)
    
    # Memory
    memory = psutil.virtual_memory()
    
    # Disk
    disk = psutil.disk_usage('/')
    
    # Network
    net_io = psutil.net_io_counters()
    
    # Temperatures
    cpu_temp = get_cpu_temp()
    gpu_temp = get_gpu_temp()
    
    # GPU Memory
    gpu_mem = get_gpu_mem()
    
    # Throttle status
    throttle = get_throttle_status()
    
    stats = {
        'cpu': {
            'percent_avg': cpu_avg,
            'percent_per_core': cpu_percent,
            'count': psutil.cpu_count(),
            'freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
        },
        'memory': {
            'total_mb': round(memory.total / (1024**2), 1),
            'used_mb': round(memory.used / (1024**2), 1),
            'available_mb': round(memory.available / (1024**2), 1),
            'percent': memory.percent
        },
        'disk': {
            'total_gb': round(disk.total / (1024**3), 1),
            'used_gb': round(disk.used / (1024**3), 1),
            'free_gb': round(disk.free / (1024**3), 1),
            'percent': disk.percent
        },
        'temperature': {
            'cpu_celsius': cpu_temp,
            'gpu_celsius': gpu_temp
        },
        'gpu': {
            'memory': gpu_mem
        },
        'network': {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        },
        'throttle_status': throttle
    }
    
    return jsonify(stats)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Pi Monitor API is running'})

if __name__ == '__main__':
    print("Starting Raspberry Pi Monitor API...")
    print("Install dependencies: pip3 install flask flask-cors psutil")
    print("Access from laptop: http://<your-pi-ip>:5000/stats")
    
    # Run on all interfaces so you can access from laptop
    app.run(host='0.0.0.0', port=7000, debug=False)