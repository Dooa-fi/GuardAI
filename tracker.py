"""
Drone Tracking Module
Tracks drones across frames with persistent IDs using centroid tracking
"""

import numpy as np
from collections import OrderedDict
from config import *
from utils import get_bbox_center, calculate_distance


class DroneTracker:
    """
    Simple centroid-based tracker for drones
    Assigns unique IDs and tracks drones across frames
    """
    
    def __init__(self, max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE):
        """
        Initialize drone tracker
        
        Args:
            max_disappeared: Max frames to keep disappeared drone
            max_distance: Max distance for matching detections to tracks
        """
        self.next_object_id = 1  # Start IDs from 1
        self.objects = OrderedDict()  # {id: centroid}
        self.disappeared = OrderedDict()  # {id: frame_count}
        self.history = OrderedDict()  # {id: [centroids]}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """
        Register new object with unique ID
        
        Args:
            centroid: (x, y) center point
        
        Returns:
            Assigned object ID
        """
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.history[object_id] = [centroid]
        self.next_object_id += 1
        
        return object_id
    
    def deregister(self, object_id):
        """
        Remove object from tracking
        
        Args:
            object_id: ID to remove
        """
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.history:
            del self.history[object_id]
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries from detector
            frame: Optional frame (not used, for compatibility)
        
        Returns:
            List of tracked objects with IDs:
            [{
                'id': int,
                'bbox': [x1, y1, x2, y2],
                'centroid': (x, y),
                'confidence': float,
                'class_name': str,
                'history': [(x, y), ...]
            }]
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return []
        
        # Extract centroids from detections
        input_centroids = []
        for detection in detections:
            centroid = get_bbox_center(detection['bbox'])
            input_centroids.append(centroid)
        
        input_centroids = np.array(input_centroids)
        
        # If no objects being tracked, register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid)
        
        else:
            # Get current tracked object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            
            # Compute distance matrix between tracked and detected
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i, j] = calculate_distance(obj_centroid, input_centroid)
            
            # Find best matches using minimum distance
            # Sort by distance and match greedily
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match existing objects to new detections
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is within threshold
                if distances[row, col] > self.max_distance:
                    continue
                
                # Update object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update history
                self.history[object_id].append(tuple(input_centroids[col]))
                
                # Keep history limited
                if len(self.history[object_id]) > TRACK_HISTORY_LENGTH:
                    self.history[object_id].pop(0)
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched existing objects (disappeared)
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])
        
        # Build tracked objects list with full info
        tracked_objects = []
        
        for i, detection in enumerate(detections):
            centroid = tuple(input_centroids[i])
            
            # Find which object ID this detection belongs to
            object_id = None
            for oid, obj_centroid in self.objects.items():
                if np.allclose(obj_centroid, centroid, atol=1e-5):
                    object_id = oid
                    break
            
            if object_id is not None:
                tracked_obj = {
                    'id': object_id,
                    'bbox': detection['bbox'],
                    'centroid': centroid,
                    'confidence': detection['confidence'],
                    'class_name': detection['class_name'],
                    'history': self.history[object_id].copy()
                }
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def get_tracked_objects(self):
        """
        Get currently tracked objects
        
        Returns:
            Dictionary of {id: centroid}
        """
        return self.objects.copy()
    
    def get_active_drone_count(self):
        """
        Get count of currently tracked drones
        
        Returns:
            Number of active tracks
        """
        return len(self.objects)
    
    def reset(self):
        """Reset tracker state"""
        self.next_object_id = 1
        self.objects.clear()
        self.disappeared.clear()
        self.history.clear()
    
    def get_track_history(self, object_id):
        """
        Get tracking history for specific object
        
        Args:
            object_id: Object ID
        
        Returns:
            List of centroid positions or None
        """
        return self.history.get(object_id, None)
    
    def get_tracker_stats(self):
        """
        Get tracker statistics
        
        Returns:
            Dictionary with tracker stats
        """
        return {
            'active_tracks': len(self.objects),
            'total_registered': self.next_object_id - 1,
            'disappeared_tracks': sum(1 for d in self.disappeared.values() if d > 0),
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance
        }


# Test function
if __name__ == "__main__":
    print("Testing DroneTracker...")
    
    try:
        # Initialize tracker
        tracker = DroneTracker()
        
        # Simulate detections across frames
        print("\nFrame 1: 2 drones detected")
        detections_1 = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class_name': 'drone'},
            {'bbox': [300, 300, 400, 400], 'confidence': 0.85, 'class_name': 'drone'}
        ]
        tracked_1 = tracker.update(detections_1)
        print(f"  Tracked: {len(tracked_1)} objects")
        for obj in tracked_1:
            print(f"    ID {obj['id']}: centroid {obj['centroid']}")
        
        print("\nFrame 2: Same drones moved slightly")
        detections_2 = [
            {'bbox': [105, 105, 205, 205], 'confidence': 0.88, 'class_name': 'drone'},
            {'bbox': [305, 305, 405, 405], 'confidence': 0.87, 'class_name': 'drone'}
        ]
        tracked_2 = tracker.update(detections_2)
        print(f"  Tracked: {len(tracked_2)} objects")
        for obj in tracked_2:
            print(f"    ID {obj['id']}: centroid {obj['centroid']}")
        
        print("\nFrame 3: One drone disappeared")
        detections_3 = [
            {'bbox': [110, 110, 210, 210], 'confidence': 0.91, 'class_name': 'drone'}
        ]
        tracked_3 = tracker.update(detections_3)
        print(f"  Tracked: {len(tracked_3)} objects")
        for obj in tracked_3:
            print(f"    ID {obj['id']}: centroid {obj['centroid']}")
        
        # Get stats
        stats = tracker.get_tracker_stats()
        print(f"\nTracker Stats:")
        print(f"  Active tracks: {stats['active_tracks']}")
        print(f"  Total registered: {stats['total_registered']}")
        print(f"  Disappeared tracks: {stats['disappeared_tracks']}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
