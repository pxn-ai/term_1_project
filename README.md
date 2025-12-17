<div align="center">

# 🎓 Smart Classroom Occupancy Counter

### *AI-Powered Human Detection & Tracking System for Raspberry Pi*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white)](https://raspberrypi.org)

*A smart IoT solution that automatically counts people entering and exiting a classroom using computer vision, ultrasonic sensors, and machine learning.*

**University of Moratuwa | ENTC | Term 1 Project**

---

<img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status">
<img src="https://img.shields.io/badge/Hardware-Raspberry%20Pi%204B-red?style=flat-square" alt="Hardware">

</div>

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🔧 Hardware Requirements](#-hardware-requirements)
- [📦 Software Requirements](#-software-requirements)
- [🚀 Installation & Setup](#-installation--setup)
- [💻 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🔌 Hardware Wiring](#-hardware-wiring)
- [🎯 How It Works](#-how-it-works)
- [⚙️ Configuration](#️-configuration)
- [🤝 Contributing](#-contributing)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Motion-Triggered Recording** | Ultrasonic sensors detect movement and automatically start recording |
| 🧠 **AI-Powered Detection** | YOLOv8 + ByteTrack for accurate human tracking |
| 📊 **Entry/Exit Counting** | Virtual counting line tracks people crossing in both directions |
| 📺 **LCD Status Display** | Real-time status updates on 16x2 I2C LCD |
| 😊 **LED Matrix Expressions** | Fun animated faces on 8x8 MAX7219 display |
| 💡 **Smart Light Control** | Automatically manages room lighting based on occupancy |
| 🎛️ **Dual Camera Support** | Works with both PiCamera and USB cameras |
| ⚡ **Multi-threaded Processing** | Concurrent video recording and analysis |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMART CLASSROOM SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  Ultrasonic  │    │   Camera     │    │    GPIO      │         │
│   │   Sensors    │───▶│  (Pi/USB)    │───▶│   Control    │         │
│   │  (L & R)     │    │              │    │   (Lights)   │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
│          │                   │                   ▲                  │
│          ▼                   ▼                   │                  │
│   ┌──────────────────────────────────────────────┴─────┐           │
│   │                   Main.py                          │           │
│   │         (Motion Detection & Recording)             │           │
│   └──────────────────────────────────────────────────┬─┘           │
│                              │                       │              │
│                              ▼                       ▼              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │   YOLOv8     │    │  LCD 16x2    │    │  LED Matrix  │         │
│   │  ByteTrack   │    │   Display    │    │   8x8 Face   │         │
│   │  (Counting)  │    │   (Status)   │    │  (Emotions)  │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hardware Requirements

| Component | Specification | Quantity |
|-----------|---------------|----------|
| 🖥️ **Raspberry Pi** | Model 4B (2GB+ RAM recommended) | 1 |
| 📷 **Camera** | Pi Camera Module v2 / USB Webcam | 1 |
| 📡 **Ultrasonic Sensor** | HC-SR04 | 2 |
| 📟 **LCD Display** | 16x2 I2C Character LCD (PCF8574) | 1 |
| 💡 **LED Matrix** | 8x8 MAX7219 | 1 |
| 🔌 **LED** | Standard LED (for power indicator) | 1 |
| 🔗 **Jumper Wires** | Male-Female, Male-Male | Various |
| ⚡ **Power Supply** | 5V 3A USB-C | 1 |

---

## 📦 Software Requirements

### Python Packages

```txt
ultralytics>=8.0.0    # YOLOv8 object detection
opencv-python>=4.8.0  # Computer vision
numpy>=1.24.0         # Numerical operations
gpiozero>=1.6.2       # GPIO control
picamera2>=0.3.12     # Pi Camera support
RPLCD>=1.3.0          # LCD display driver
luma.led_matrix>=1.7  # LED matrix driver
```

### System Requirements

- **OS:** Raspberry Pi OS (64-bit recommended)
- **Python:** 3.9 or higher
- **SPI:** Enabled for LED matrix
- **I2C:** Enabled for LCD display
- **Camera:** Enabled in raspi-config

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/pxn-ai/term_1_project.git
cd term_1_project
```

### Step 2: Enable Required Interfaces

```bash
sudo raspi-config
```
Navigate to **Interface Options** and enable:
- ✅ Camera
- ✅ SPI
- ✅ I2C

### Step 3: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy gpiozero
pip install RPLCD luma.led_matrix picamera2
```

### Step 5: Download YOLOv8 Model (Auto-downloads on first run)

```bash
# The yolov8n.pt file will be automatically downloaded
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Step 6: Verify Hardware Connections

```bash
# Test sensors and LED
python sensor_check.py

# Test LCD and LED matrix
python demo.py
```

---

## 💻 Usage

### 🎬 Live Mode (Full System)

Run the complete smart classroom system with motion detection:

```bash
python Main.py
```

**Options:**
```bash
python Main.py --model n --line 0.5 --skip 1
```

### 📹 Video Analysis Mode

Analyze a pre-recorded video:

```bash
python Main.py video.mp4 --output result.mp4 --preview
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video` | Path to video file (optional) | Live mode |
| `--output` | Save annotated video | None |
| `--preview` | Show live preview | False |
| `--model` | YOLO model size: `n` (nano), `s` (small) | `n` |
| `--skip` | Process every Nth frame | `1` |
| `--line` | Counting line position (0.0-1.0) | `0.5` |
| `--json` | Save results to JSON | None |

### Quick Demo

```bash
# Run the demo to test all components
python demo.py
```

---

## 📁 Project Structure

```
term_1_project/
│
├── 🚀 Main.py                 # Main application entry point
├── 🧠 Human_Identifier.py     # YOLOv8 human detection & tracking
├── 👁️ eyes.py                 # LED matrix facial expressions
├── 📺 lcd_display.py          # 16x2 LCD display controller
├── 🎭 faces_and_text.py       # LED matrix faces with scrolling text
├── 🔬 sensor_check.py         # Hardware testing utility
├── 🎬 demo.py                 # Component demonstration script
├── 🔧 light_control.sh        # Shell script for virtual env
├── 🤖 yolov8n.pt              # YOLOv8 nano model weights
├── 📖 README.md               # This file
└── 📂 __pycache__/            # Python cache files
```

---

## 🔌 Hardware Wiring

### GPIO Pin Configuration

| Component | GPIO Pin | Physical Pin |
|-----------|----------|--------------|
| **Power LED** | GPIO 17 | Pin 11 |
| **Left Ultrasonic Trigger** | GPIO 22 | Pin 15 |
| **Left Ultrasonic Echo** | GPIO 27 | Pin 13 |
| **Right Ultrasonic Trigger** | GPIO 24 | Pin 18 |
| **Right Ultrasonic Echo** | GPIO 23 | Pin 16 |
| **I2C SDA (LCD)** | GPIO 2 | Pin 3 |
| **I2C SCL (LCD)** | GPIO 3 | Pin 5 |
| **SPI MOSI (LED Matrix)** | GPIO 10 | Pin 19 |
| **SPI SCLK (LED Matrix)** | GPIO 11 | Pin 23 |
| **SPI CE0 (LED Matrix)** | GPIO 8 | Pin 24 |

### Wiring Diagram

```
                    Raspberry Pi 4B
              ┌─────────────────────────┐
              │   ┌───────────────┐     │
              │   │               │     │
    [LCD]─────│───│  I2C (3,5)    │     │
              │   │               │     │
 [MAX7219]────│───│  SPI (19,23,24)     │
              │   │               │     │
[HC-SR04 L]───│───│  GPIO 22,27  │     │
              │   │               │     │
[HC-SR04 R]───│───│  GPIO 23,24  │     │
              │   │               │     │
   [LED]──────│───│  GPIO 17     │     │
              │   └───────────────┘     │
              └─────────────────────────┘
```

---

## 🎯 How It Works

### 1️⃣ Motion Detection
- Two ultrasonic sensors continuously monitor the doorway
- When someone comes within 2 meters, recording begins

### 2️⃣ Video Recording
- Camera captures video while motion is detected
- Recording continues for 5 seconds after last detection

### 3️⃣ AI Analysis
- YOLOv8 nano model detects humans in each frame
- ByteTrack algorithm assigns unique IDs to each person
- Virtual counting line tracks direction of movement

### 4️⃣ Occupancy Update
- System calculates net entries (entered - exited)
- Updates classroom occupancy count
- Controls room lighting based on occupancy

### 5️⃣ Visual Feedback
- LCD shows current status and occupancy count
- LED matrix displays animated facial expressions
- Happy face 😊 when idle, suspicious face 👀 when recording

---

## ⚙️ Configuration

### Adjusting Detection Range

In `Main.py`, modify:
```python
detection_range = 200  # Distance in cm (default: 2 meters)
```

### Camera Selection

```python
USB_Camera_preferred = False  # True for USB webcam, False for PiCamera
```

### Frame Resolution

```python
frame_width, frame_height = 1280, 720  # Video resolution
```

### Counting Line Position

```python
# 0.0 = left edge, 1.0 = right edge
python Main.py --line 0.5  # Center of frame
```

---

## 🖥️ Running on a Laptop (Simulation Mode)

To run the video analysis portion on any laptop without Raspberry Pi hardware:

### 1. Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 2. Analyze Video Files
```bash
python Human_Identifier.py --video your_video.mp4 --output result.mp4 --preview
```

> ⚠️ **Note:** GPIO-dependent features (sensors, LCD, LED matrix) require Raspberry Pi hardware. The video analysis module works independently on any system.

---

## 📊 Performance Tips

| Tip | Description |
|-----|-------------|
| 🔧 Use `--skip 2` | Process every 2nd frame for faster analysis |
| 📉 Use model `n` | Nano model is optimized for Raspberry Pi |
| 🔄 Reduce resolution | Lower resolution = faster processing |
| 🧊 Add heatsink | Prevent thermal throttling on Pi |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Run `sudo raspi-config` and enable camera |
| LCD not displaying | Check I2C address: `sudo i2cdetect -y 1` |
| LED matrix issues | Verify SPI is enabled in raspi-config |
| Sensors not working | Check GPIO connections and pin numbers |
| Model download fails | Manually download `yolov8n.pt` |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the **University of Moratuwa ENTC curriculum**.

---

<div align="center">

### Made with ❤️ by ENTC Students

**University of Moratuwa, Sri Lanka**

⭐ Star this repo if you found it helpful!

</div>
