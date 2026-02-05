# 🚁 Drone Detection System

Real-time drone detection system using YOLOv8 with tracking, alerts, and a professional Streamlit interface.

## � Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Use the System
1. Click **"Initialize System"** in the sidebar
2. Select **Webcam** or upload a **Video File**
3. Adjust **Confidence Threshold** (0.3-0.7 recommended)
4. Click **"Start"** to begin detection
5. Watch drones get detected in real-time!

## ✨ Features

- ✅ **Real-time Detection**: YOLOv8 at 15-30 FPS
- ✅ **Multi-Drone Tracking**: Unique IDs that persist across frames
- ✅ **Audio Alerts**: Siren sound when drones detected (3-second cooldown)
- ✅ **Visual Indicators**: Bounding boxes, tracking IDs, movement trails
- ✅ **Professional UI**: Streamlit interface with real-time metrics
- ✅ **Detection Logging**: CSV export with timestamps and coordinates
- ✅ **Configurable**: Adjustable confidence, volume, and settings

## 📁 Project Structure

```
drone-detection/
├── app.py              # Main Streamlit application
├── detector.py         # YOLOv8 detection module
├── tracker.py          # Multi-drone tracking
├── alert_system.py     # Audio alert system
├── utils.py            # Helper functions
├── config.py           # Configuration settings
├── best.pt             # Trained YOLOv8 model
├── requirements.txt    # Dependencies
├── assets/
│   └── siren.wav      # Alert sound
└── logs/              # Detection logs (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence (0.0-1.0)
ALERT_COOLDOWN = 3.0        # Seconds between alerts
ALERT_VOLUME = 0.7          # Volume (0.0-1.0)
MAX_DISAPPEARED = 30        # Frames to keep disappeared drones
```

## 🎯 Controls

| Button | Action |
|--------|--------|
| Initialize System | Load model and components |
| Start | Begin detection |
| Stop | Stop detection |
| Reset Statistics | Clear logs and counters |
| Test Alert | Play alert sound |
| Save Detection Log | Export to CSV |

## 🔧 Troubleshooting

**Camera not detected:**
- Try camera index 1 or 2 in sidebar
- Check camera permissions

**Low FPS (<10):**
- Lower confidence threshold
- Close other applications
- Use smaller video resolution

**No detections:**
- Lower confidence threshold to 0.3
- Ensure good lighting
- Check if drone is clearly visible

**No alert sound:**
- Check system volume
- Verify `assets/siren.wav` exists

## 📊 Performance

**Expected on Intel i7-1065G7:**
- FPS: 15-30 (CPU mode)
- Latency: 30-60ms per frame
- Memory: ~500MB

## 🎓 Technical Details

**Model**: YOLOv8n (nano) trained on 8025 drone images
**Training**: Google Colab, T4 GPU, 100 epochs
**Tracking**: Centroid-based algorithm with persistent IDs
**Alerts**: pygame.mixer for non-blocking audio
**UI**: Streamlit for rapid development

## 📝 Detection Log Format

CSV exports include:
- Timestamp
- Frame number
- Drone ID
- Confidence score
- Bounding box coordinates (x1, y1, x2, y2)
- Center position (center_x, center_y)

## 🎯 Use Cases

- Security and perimeter monitoring
- Event no-fly zone enforcement
- Research and drone behavior analysis
- Educational demonstrations

## 📦 Dependencies

- ultralytics (YOLOv8)
- opencv-python (Computer Vision)
- streamlit (Web UI)
- pygame (Audio)
- numpy, pandas (Data processing)

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics
- **Dataset**: Kaggle drone-yolo-detection
- **Libraries**: OpenCV, Streamlit, pygame, PyTorch

---

**Built for real-time drone detection and monitoring** 🚁
