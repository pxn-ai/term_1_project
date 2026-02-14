#!/usr/bin/env python3
"""
Advanced Raspberry Pi Monitor - Enhanced Backend
Provides extended metrics including GPU, network I/O, and disk I/O statistics
"""

from flask import Flask, jsonify
from flask_cors import CORS
import psutil
import time
import subprocess
import re
from collections import deque
from threading import Thread, Lock

app = Flask(__name__)
CORS(app)  # Enable CORS for browser access

# Data storage for historical metrics
class MetricsHistory:
    def __init__(self, maxlen=60):
        self.lock = Lock()
        self.maxlen = maxlen
        self.network_counters = deque(maxlen=maxlen)
        self.disk_io_counters = deque(maxlen=maxlen)
        self.last_network = None
        self.last_disk_io = None
        self.last_time = None
        
    def update(self):
        """Update metrics counters"""
        with self.lock:
            current_time = time.time()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if self.last_network and self.last_time:
                time_delta = current_time - self.last_time
                bytes_sent_per_sec = (net_io.bytes_sent - self.last_network.bytes_sent) / time_delta
                bytes_recv_per_sec = (net_io.bytes_recv - self.last_network.bytes_recv) / time_delta
                self.network_counters.append({
                    'bytes_sent_per_sec': bytes_sent_per_sec,
                    'bytes_recv_per_sec': bytes_recv_per_sec,
                    'timestamp': current_time
                })
            self.last_network = net_io
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if self.last_disk_io and self.last_time:
                time_delta = current_time - self.last_time
                read_bytes_per_sec = (disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
                write_bytes_per_sec = (disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta
                self.disk_io_counters.append({
                    'read_bytes_per_sec': read_bytes_per_sec,
                    'write_bytes_per_sec': write_bytes_per_sec,
                    'timestamp': current_time
                })
            self.last_disk_io = disk_io
            self.last_time = current_time
    
    def get_latest_network(self):
        """Get latest network I/O rates"""
        with self.lock:
            if self.network_counters:
                return self.network_counters[-1]
            return {'bytes_sent_per_sec': 0, 'bytes_recv_per_sec': 0}
    
    def get_latest_disk_io(self):
        """Get latest disk I/O rates"""
        with self.lock:
            if self.disk_io_counters:
                return self.disk_io_counters[-1]
            return {'read_bytes_per_sec': 0, 'write_bytes_per_sec': 0}

# Initialize metrics history
metrics_history = MetricsHistory()

def get_gpu_stats():
    """Get GPU statistics (Raspberry Pi specific)"""
    try:
        # Get GPU temperature
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True,
            text=True,
            timeout=1
        )
        temp_match = re.search(r"temp=([\d.]+)'C", result.stdout)
        gpu_temp = float(temp_match.group(1)) if temp_match else 0
        
        # Get GPU memory
        result = subprocess.run(
            ['vcgencmd', 'get_mem', 'gpu'],
            capture_output=True,
            text=True,
            timeout=1
        )
        mem_match = re.search(r"gpu=([\d]+)M", result.stdout)
        gpu_memory = int(mem_match.group(1)) if mem_match else 0
        
        # Estimate GPU usage based on GPU processes (simplified)
        # On Raspberry Pi, actual GPU usage is harder to measure
        # This is a placeholder - returns 0-15% based on system load
        cpu_percent = psutil.cpu_percent()
        gpu_percent = min(cpu_percent * 0.3, 15)  # Rough estimation
        
        return {
            'percent': round(gpu_percent, 1),
            'temperature': gpu_temp,
            'memory_mb': gpu_memory
        }
    except Exception as e:
        # Fallback if vcgencmd is not available or fails
        return {
            'percent': 0,
            'temperature': 0,
            'memory_mb': 128  # Default Raspberry Pi GPU memory
        }

def get_cpu_temp():
    """Get CPU temperature"""
    try:
        # Try vcgencmd first (Raspberry Pi specific)
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True,
            text=True,
            timeout=1
        )
        temp_match = re.search(r"temp=([\d.]+)'C", result.stdout)
        if temp_match:
            return int(float(temp_match.group(1)))
    except:
        pass
    
    try:
        # Fallback to thermal zone
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000
            return int(temp)
    except:
        return 45  # Default fallback

@app.route('/stats')
def get_basic_stats():
    """Basic stats endpoint - same as original dashboard"""
    cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
    
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        'cpu': {
            'percent_avg': round(sum(cpu_percent) / len(cpu_percent), 1),
            'percent_per_core': [round(x, 1) for x in cpu_percent]
        },
        'memory': {
            'percent': round(memory.percent, 1),
            'used_mb': round(memory.used / (1024 * 1024), 1),
            'total_mb': round(memory.total / (1024 * 1024), 1)
        },
        'disk': {
            'percent': round(disk.percent, 1),
            'used_gb': round(disk.used / (1024 * 1024 * 1024), 1),
            'total_gb': round(disk.total / (1024 * 1024 * 1024), 1)
        },
        'temperature': {
            'cpu_celsius': get_cpu_temp()
        }
    }
    
    return jsonify(stats)

@app.route('/stats/advanced')
def get_advanced_stats():
    """Advanced stats endpoint with all metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
    
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    gpu_stats = get_gpu_stats()
    network_stats = metrics_history.get_latest_network()
    disk_io_stats = metrics_history.get_latest_disk_io()
    
    stats = {
        'cpu': {
            'percent_avg': round(sum(cpu_percent) / len(cpu_percent), 1),
            'percent_per_core': [round(x, 1) for x in cpu_percent]
        },
        'memory': {
            'percent': round(memory.percent, 1),
            'used_mb': round(memory.used / (1024 * 1024), 1),
            'total_mb': round(memory.total / (1024 * 1024), 1)
        },
        'disk': {
            'percent': round(disk.percent, 1),
            'used_gb': round(disk.used / (1024 * 1024 * 1024), 1),
            'total_gb': round(disk.total / (1024 * 1024 * 1024), 1)
        },
        'temperature': {
            'cpu_celsius': get_cpu_temp()
        },
        'gpu': gpu_stats,
        'network': {
            'bytes_sent_per_sec': round(network_stats['bytes_sent_per_sec'], 0),
            'bytes_recv_per_sec': round(network_stats['bytes_recv_per_sec'], 0)
        },
        'disk_io': {
            'read_bytes_per_sec': round(disk_io_stats['read_bytes_per_sec'], 0),
            'write_bytes_per_sec': round(disk_io_stats['write_bytes_per_sec'], 0)
        }
    }
    
    return jsonify(stats)

def metrics_updater():
    """Background thread to update metrics continuously"""
    while True:
        try:
            metrics_history.update()
            time.sleep(1)  # Update every second
        except Exception as e:
            print(f"Error updating metrics: {e}")
            time.sleep(1)

if __name__ == '__main__':
    # Start background metrics updater
    updater_thread = Thread(target=metrics_updater, daemon=True)
    updater_thread.start()
    
    print("=" * 60)
    print("🥧 Advanced Raspberry Pi Monitor - Backend Server")
    print("=" * 60)
    print("\nServer starting on port 7000...")
    print("\nEndpoints:")
    print("  • http://[YOUR_PI_IP]:7000/stats          - Basic metrics")
    print("  • http://[YOUR_PI_IP]:7000/stats/advanced - Advanced metrics")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # Run Flask server
    app.run(host='0.0.0.0', port=7000, debug=False)
