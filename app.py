"""
Drone Detection System - Streamlit Application
Main application integrating detector, tracker, and alert system
"""

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import pandas as pd

from detector import DroneDetector
from tracker import DroneTracker
from alert_system import AlertSystem
from utils import (
    draw_bounding_box, draw_tracking_line, draw_fps, 
    draw_alert_indicator, FPSCalculator, get_timestamp,
    create_log_entry, save_detection_log
)
from config import *

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .status-running {
        color: #00cc00;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'tracker' not in st.session_state:
        st.session_state.tracker = None
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = None
    if 'fps_calculator' not in st.session_state:
        st.session_state.fps_calculator = FPSCalculator()
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'video_source' not in st.session_state:
        st.session_state.video_source = None


def load_components():
    """Load detector, tracker, and alert system"""
    try:
        with st.spinner("Loading detection model..."):
            st.session_state.detector = DroneDetector()
        
        st.session_state.tracker = DroneTracker()
        st.session_state.alert_system = AlertSystem()
        
        return True
    except Exception as e:
        st.error(f"Error loading components: {e}")
        return False


def process_frame(frame, conf_threshold):
    """
    Process single frame through detection pipeline
    
    Args:
        frame: Input frame
        conf_threshold: Confidence threshold
    
    Returns:
        Processed frame, tracked objects, fps
    """
    # Run detection
    detections = st.session_state.detector.detect(frame, conf_threshold)
    
    # Update tracker
    tracked_objects = st.session_state.tracker.update(detections)
    
    # Trigger alert if drones detected
    if len(tracked_objects) > 0:
        st.session_state.alert_system.trigger_alert()
    
    # Update FPS
    st.session_state.fps_calculator.update()
    fps = st.session_state.fps_calculator.get_fps()
    
    # Draw visualizations
    display_frame = frame.copy()
    
    # Draw tracked objects
    for obj in tracked_objects:
        display_frame = draw_bounding_box(
            display_frame,
            obj['bbox'],
            obj['class_name'],
            obj['confidence'],
            track_id=obj['id']
        )
        
        # Draw tracking line
        if len(obj['history']) > 1:
            display_frame = draw_tracking_line(display_frame, obj['history'])
    
    # Draw FPS
    display_frame = draw_fps(display_frame, fps)
    
    # Draw alert indicator
    if st.session_state.alert_system.is_alert_active():
        display_frame = draw_alert_indicator(display_frame, True)
    
    # Log detections
    if LOG_DETECTIONS and len(tracked_objects) > 0:
        for obj in tracked_objects:
            log_entry = create_log_entry(
                get_timestamp(),
                obj['id'],
                obj['confidence'],
                obj['bbox'],
                st.session_state.frame_count
            )
            st.session_state.detection_log.append(log_entry)
    
    # Update statistics
    st.session_state.total_detections += len(tracked_objects)
    st.session_state.frame_count += 1
    
    return display_frame, tracked_objects, fps


def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🚁 Drone Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"## {SIDEBAR_TITLE}")
        
        # Load components button
        if st.session_state.detector is None:
            if st.button("🔧 Initialize System", use_container_width=True):
                if load_components():
                    st.success("✓ System initialized!")
                    st.rerun()
        else:
            st.success("✓ System Ready")
        
        st.markdown("---")
        
        # Video source selection
        st.markdown("### 📹 Video Source")
        source_type = st.radio(
            "Select source:",
            ["Webcam", "Video File"],
            key="source_type"
        )
        
        if source_type == "Webcam":
            camera_index = st.number_input(
                "Camera Index",
                min_value=0,
                max_value=5,
                value=WEBCAM_INDEX,
                help="Usually 0 for primary camera"
            )
            video_source = camera_index
        else:
            uploaded_file = st.file_uploader(
                "Upload video file",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            if uploaded_file:
                # Save uploaded file temporarily
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_source = tfile.name
            else:
                video_source = None
        
        st.markdown("---")
        
        # Detection settings
        st.markdown("### ⚙️ Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=CONFIDENCE_MIN,
            max_value=CONFIDENCE_MAX,
            value=CONFIDENCE_THRESHOLD,
            step=CONFIDENCE_STEP,
            help="Minimum confidence for detections"
        )
        
        # Alert settings
        st.markdown("### 🔔 Alert Settings")
        enable_alerts = st.checkbox("Enable Audio Alerts", value=True)
        
        if enable_alerts and st.session_state.alert_system:
            alert_volume = st.slider(
                "Alert Volume",
                min_value=0.0,
                max_value=1.0,
                value=ALERT_VOLUME,
                step=0.1
            )
            st.session_state.alert_system.set_volume(alert_volume)
            
            if st.button("🔊 Test Alert"):
                st.session_state.alert_system.test_alert()
        
        st.markdown("---")
        
        # Control buttons
        st.markdown("### 🎮 Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.running or video_source is None):
                st.session_state.running = True
                st.session_state.video_source = video_source
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.running):
                st.session_state.running = False
                if st.session_state.alert_system:
                    st.session_state.alert_system.stop_alert()
                st.rerun()
        
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.detection_log = []
            st.session_state.total_detections = 0
            st.session_state.frame_count = 0
            if st.session_state.tracker:
                st.session_state.tracker.reset()
            st.rerun()
        
        st.markdown("---")
        
        # System info
        st.markdown("### ℹ️ System Info")
        if st.session_state.detector:
            model_info = st.session_state.detector.get_model_info()
            st.text(f"Device: {model_info['device'].upper()}")
            st.text(f"Model: {MODEL_PATH}")
    
    # Main content area
    if st.session_state.detector is None:
        st.info("👈 Click 'Initialize System' in the sidebar to start")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fps_placeholder = st.empty()
    with col2:
        detections_placeholder = st.empty()
    with col3:
        active_drones_placeholder = st.empty()
    with col4:
        frames_placeholder = st.empty()
    
    # Alert status
    alert_placeholder = st.empty()
    
    # Video display
    video_placeholder = st.empty()
    
    # Detection log
    with st.expander("📊 Detection Log", expanded=False):
        log_placeholder = st.empty()
    
    # Processing loop
    if st.session_state.running and st.session_state.video_source is not None:
        # Open video source
        cap = cv2.VideoCapture(st.session_state.video_source)
        
        if not cap.isOpened():
            st.error("❌ Could not open video source")
            st.session_state.running = False
            return
        
        # Processing loop
        while st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("⚠️ End of video or camera disconnected")
                st.session_state.running = False
                break
            
            # Process frame
            display_frame, tracked_objects, fps = process_frame(frame, conf_threshold)
            
            # Update metrics
            fps_placeholder.metric("FPS", f"{fps:.1f}")
            detections_placeholder.metric("Total Detections", st.session_state.total_detections)
            
            active_count = st.session_state.tracker.get_active_drone_count()
            active_drones_placeholder.metric("Active Drones", active_count)
            frames_placeholder.metric("Frames Processed", st.session_state.frame_count)
            
            # Update alert status
            if st.session_state.alert_system.is_alert_active():
                alert_placeholder.markdown(
                    '<div class="alert-box">⚠️ DRONE DETECTED ⚠️</div>',
                    unsafe_allow_html=True
                )
            else:
                alert_placeholder.empty()
            
            # Display frame
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update detection log
            if len(st.session_state.detection_log) > 0:
                df = pd.DataFrame(st.session_state.detection_log[-50:])  # Last 50 entries
                log_placeholder.dataframe(df, use_container_width=True)
            
            # Small delay to prevent overwhelming the UI
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        
        # Save log if detections were made
        if len(st.session_state.detection_log) > 0:
            if st.button("💾 Save Detection Log"):
                save_detection_log(st.session_state.detection_log)
                st.success("✓ Log saved!")
    
    else:
        # Show status when not running
        if st.session_state.video_source is None:
            st.info("📹 Please select a video source from the sidebar")
        else:
            st.info("▶️ Click 'Start' to begin detection")
        
        # Show example metrics
        fps_placeholder.metric("FPS", "0.0")
        detections_placeholder.metric("Total Detections", st.session_state.total_detections)
        active_drones_placeholder.metric("Active Drones", 0)
        frames_placeholder.metric("Frames Processed", st.session_state.frame_count)


if __name__ == "__main__":
    main()
