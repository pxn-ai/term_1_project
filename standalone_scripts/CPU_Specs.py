import time
import psutil
import subprocess

def get_cpu_usage():
    # interval=1 makes this call block for 1 second to calculate usage
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    return psutil.virtual_memory().percent

def get_gpu_memory():
    try:
        # vcgencmd is specific to Raspberry Pi
        result = subprocess.check_output(['vcgencmd', 'get_mem', 'gpu']).decode('utf-8')
        return result.strip().replace('gpu=', '')
    except FileNotFoundError:
        return "N/A (vcgencmd not found)"
    except Exception as e:
        return f"Error: {e}"

def get_temperature():
    try:
        # Try vcgencmd first
        result = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
        return result.strip().replace('temp=', '')
    except Exception:
        try:
            # Fallback to thermal_zone0 (standard Linux thermal interface)
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return f"{temp:.1f}'C"
        except Exception:
            return "N/A"

def main():
    print("Monitoring System Stats... (Press Ctrl+C to stop)")
    print("-" * 80)
    try:
        while True:
            cpu = get_cpu_usage()
            ram = get_ram_usage()
            gpu_mem = get_gpu_memory()
            temp = get_temperature()
            
            # Print on the same line using carriage return
            print(f"\rCPU Usage: {cpu:<5}% | RAM Usage: {ram:<5}% | GPU Mem: {gpu_mem:<10} | Temp: {temp}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
