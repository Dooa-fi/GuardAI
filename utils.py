"""
Utility functions for Drone Detection System
Helper functions used across multiple modules
"""

import cv2
import time
import os
import csv
from datetime import datetime
import numpy as np
from config import *


class FPSCalculator:
    """Calculate FPS for video processing"""
    
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.start_time = time.time()
    
    def update(self):
        """Update with current frame time"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Calculate and return current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.frame_times) - 1) / time_diff
        return fps


def draw_bounding_box(frame, bbox, label, confidence, color=COLOR_BBOX, track_id=None):
    """
    Draw bounding box with label on frame
    
    Args:
        frame: Image frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        label: Class label
        confidence: Detection confidence
        color: Box color (BGR)
        track_id: Optional tracking ID
    
    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
    
    # Prepare label text
    if track_id is not None:
        text = f"ID:{track_id} {label} {confidence:.2f}"
    else:
        text = f"{label} {confidence:.2f}"
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x1, y1 - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        COLOR_TEXT,
        FONT_THICKNESS
    )
    
    return frame


def draw_tracking_line(frame, points, color=COLOR_TRACK_LINE, thickness=2):
    """
    Draw tracking line showing drone movement path
    
    Args:
        frame: Image frame
        points: List of (x, y) center points
        color: Line color (BGR)
        thickness: Line thickness
    
    Returns:
        Modified frame
    """
    if len(points) < 2:
        return frame
    
    # Draw lines connecting points
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        
        # Draw line with fading effect (older points are more transparent)
        alpha = i / len(points)
        pt1 = tuple(map(int, points[i - 1]))
        pt2 = tuple(map(int, points[i]))
        
        cv2.line(frame, pt1, pt2, color, thickness)
    
    return frame


def format_confidence(confidence):
    """Format confidence score as percentage string"""
    return f"{confidence * 100:.1f}%"


def get_bbox_center(bbox):
    """
    Calculate center point of bounding box
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
    
    Returns:
        Distance as float
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def create_log_entry(timestamp, drone_id, confidence, bbox, frame_number):
    """
    Create detection log entry
    
    Args:
        timestamp: Detection timestamp
        drone_id: Tracking ID
        confidence: Detection confidence
        bbox: Bounding box coordinates
        frame_number: Frame number
    
    Returns:
        Dictionary with log data
    """
    x1, y1, x2, y2 = bbox
    
    return {
        'timestamp': timestamp,
        'frame': frame_number,
        'drone_id': drone_id,
        'confidence': confidence,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'center_x': (x1 + x2) / 2,
        'center_y': (y1 + y2) / 2
    }


def save_detection_log(log_entries, filename=None):
    """
    Save detection log to CSV file
    
    Args:
        log_entries: List of log entry dictionaries
        filename: Optional custom filename
    """
    if not log_entries:
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(LOG_DIR, f"detections_{timestamp}.csv")
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'frame', 'drone_id', 'confidence', 
                     'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in log_entries:
            writer.writerow(entry)
    
    print(f"Detection log saved to: {filename}")


def resize_frame(frame, width=None, height=None):
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        width: Target width (optional)
        height: Target height (optional)
    
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is not None:
        aspect_ratio = width / w
        new_height = int(h * aspect_ratio)
        return cv2.resize(frame, (width, new_height))
    
    if height is not None:
        aspect_ratio = height / h
        new_width = int(w * aspect_ratio)
        return cv2.resize(frame, (new_width, height))


def draw_fps(frame, fps, position=(10, 30)):
    """
    Draw FPS counter on frame
    
    Args:
        frame: Image frame
        fps: FPS value
        position: Text position (x, y)
    
    Returns:
        Modified frame
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    return frame


def draw_alert_indicator(frame, is_alert_active):
    """
    Draw alert indicator on frame
    
    Args:
        frame: Image frame
        is_alert_active: Boolean indicating if alert is active
    
    Returns:
        Modified frame
    """
    if not is_alert_active:
        return frame
    
    h, w = frame.shape[:2]
    
    # Draw red border
    cv2.rectangle(frame, (0, 0), (w, h), COLOR_ALERT, 10)
    
    # Draw alert text
    text = "⚠ DRONE DETECTED ⚠"
    font_scale = 1.5
    thickness = 3
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Center text at top
    x = (w - text_width) // 2
    y = text_height + 20
    
    # Draw background
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        COLOR_ALERT,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_TEXT,
        thickness
    )
    
    return frame


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [LOG_DIR, "assets", "test_videos"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_timestamp():
    """Get current timestamp as formatted string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_frame(frame, filename=None):
    """
    Save frame to file
    
    Args:
        frame: Image frame
        filename: Optional custom filename
    
    Returns:
        Saved filename
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(LOG_DIR, f"detection_{timestamp}.jpg")
    
    cv2.imwrite(filename, frame)
    return filename
