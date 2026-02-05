"""
Drone Detection Module
Handles YOLO model loading and inference for drone detection
"""

from ultralytics import YOLO
import cv2
import numpy as np
from config import *


class DroneDetector:
    """
    YOLOv8 based drone detector
    Handles model loading, inference, and detection filtering
    """
    
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
        """
        Initialize drone detector
        
        Args:
            model_path: Path to YOLOv8 model file
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = 'cuda' if USE_GPU else 'cpu'
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move to appropriate device
            if USE_GPU:
                try:
                    self.model.to('cuda')
                    print("✓ Model loaded on GPU")
                except:
                    self.model.to('cpu')
                    self.device = 'cpu'
                    print("⚠ GPU not available, using CPU")
            else:
                self.model.to('cpu')
                print("✓ Model loaded on CPU")
            
            print(f"✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def detect(self, frame, conf_threshold=None):
        """
        Run detection on a single frame
        
        Args:
            frame: Input image frame (BGR format)
            conf_threshold: Optional confidence threshold override
        
        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int
            - class_name: str
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Use provided threshold or default
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=conf,
                iou=IOU_THRESHOLD,
                verbose=False,
                device=self.device
            )
            
            # Parse results
            detections = []
            
            if len(results) > 0:
                result = results[0]  # Get first result
                
                # Extract boxes, confidences, and classes
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidences
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    
                    # Get class names
                    names = result.names
                    
                    # Create detection list
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        detection = {
                            'bbox': box.tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': names[cls_id]
                        }
                        detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def get_detections(self, frame, conf_threshold=None):
        """
        Alias for detect() method for compatibility
        
        Args:
            frame: Input image frame
            conf_threshold: Optional confidence threshold
        
        Returns:
            List of detections
        """
        return self.detect(frame, conf_threshold)
    
    def is_drone_detected(self, frame, conf_threshold=None):
        """
        Check if any drone is detected in frame
        
        Args:
            frame: Input image frame
            conf_threshold: Optional confidence threshold
        
        Returns:
            Boolean indicating if drone detected
        """
        detections = self.detect(frame, conf_threshold)
        return len(detections) > 0
    
    def get_detection_count(self, frame, conf_threshold=None):
        """
        Get number of drones detected in frame
        
        Args:
            frame: Input image frame
            conf_threshold: Optional confidence threshold
        
        Returns:
            Number of detections
        """
        detections = self.detect(frame, conf_threshold)
        return len(detections)
    
    def set_confidence_threshold(self, threshold):
        """
        Update confidence threshold
        
        Args:
            threshold: New confidence threshold (0.0 - 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.conf_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_model_info(self):
        """
        Get model information
        
        Returns:
            Dictionary with model info
        """
        if self.model is None:
            return None
        
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.conf_threshold,
            'iou_threshold': IOU_THRESHOLD
        }


# Test function
if __name__ == "__main__":
    print("Testing DroneDetector...")
    
    try:
        # Initialize detector
        detector = DroneDetector()
        
        # Print model info
        info = detector.get_model_info()
        print(f"\nModel Info:")
        print(f"  Path: {info['model_path']}")
        print(f"  Device: {info['device']}")
        print(f"  Confidence: {info['confidence_threshold']}")
        print(f"  IOU: {info['iou_threshold']}")
        
        # Test with dummy frame
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        print(f"\n✓ Detection test passed (found {len(detections)} objects in dummy frame)")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
