"""
Configuration settings for Drone Detection System
Centralized configuration for easy adjustments without touching code
"""

import os

# ==================== MODEL SETTINGS ====================
MODEL_PATH = "drone_best_2.pt"  # Path to trained YOLOv8 model
CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold (0.0 - 1.0)
IOU_THRESHOLD = 0.45  # Intersection over Union threshold for NMS

# ==================== VIDEO SETTINGS ====================
DEFAULT_VIDEO_WIDTH = 640  # Default video width for processing
DEFAULT_VIDEO_HEIGHT = 480  # Default video height for processing
MAX_FPS = 30  # Maximum FPS to process
WEBCAM_INDEX = 0  # Default webcam index (0 = primary camera)

# ==================== TRACKING SETTINGS ====================
MAX_DISAPPEARED = 30  # Max frames to keep track of disappeared drone
MAX_DISTANCE = 50  # Max distance for matching detections to tracks
TRACK_HISTORY_LENGTH = 30  # Number of frames to keep in track history

# ==================== ALERT SETTINGS ====================
ALERT_SOUND_PATH = os.path.join("assets", "siren.wav")
ALERT_COOLDOWN = 3.0  # Seconds between alerts (avoid spam)
ALERT_VOLUME = 0.7  # Volume level (0.0 - 1.0)

# ==================== UI SETTINGS ====================
# Colors (BGR format for OpenCV)
COLOR_BBOX = (0, 255, 0)  # Green for bounding boxes
COLOR_TEXT = (255, 255, 255)  # White for text
COLOR_ALERT = (0, 0, 255)  # Red for alert indicator
COLOR_TRACK_LINE = (255, 0, 255)  # Magenta for tracking lines

# Font settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BBOX_THICKNESS = 2

# ==================== LOGGING SETTINGS ====================
LOG_DIR = "logs"
LOG_DETECTIONS = True  # Whether to log detections to CSV
SAVE_DETECTION_IMAGES = False  # Whether to save images with detections

# ==================== STREAMLIT SETTINGS ====================
PAGE_TITLE = "🚁 Drone Detection System"
PAGE_ICON = "🚁"
LAYOUT = "wide"

# Sidebar settings
SIDEBAR_TITLE = "⚙️ Control Panel"
CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 1.0
CONFIDENCE_STEP = 0.05

# ==================== PERFORMANCE SETTINGS ====================
SKIP_FRAMES = 0  # Number of frames to skip (0 = process all frames)
USE_GPU = True  # Use GPU if available
HALF_PRECISION = False  # Use FP16 for faster inference (if GPU supports)

# ==================== FILE PATHS ====================
# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("test_videos", exist_ok=True)
