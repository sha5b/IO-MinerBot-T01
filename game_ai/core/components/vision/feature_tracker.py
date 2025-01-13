"""Feature tracking component for detecting and tracking distinctive features."""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

class FeatureTracker:
    """Handles detection and tracking of distinctive features between frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature tracker component.
        
        Args:
            config (dict): Configuration settings including feature detection parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize SIFT feature detector
        self.feature_detector = cv2.SIFT_create()
        
        # Store previous frame data
        self.last_keypoints = None
        self.last_descriptors = None
        
    def track_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect and track features between frames.
        
        Args:
            frame (numpy.ndarray): Current frame in BGR format
            
        Returns:
            dict: Feature tracking results including keypoints and matches
        """
        try:
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            # If this is the first frame, store and return initial features
            if self.last_keypoints is None or self.last_descriptors is None:
                self.last_keypoints = keypoints
                self.last_descriptors = descriptors
                return {
                    'keypoints': self._format_keypoints(keypoints),
                    'matches': []
                }
            
            # Match features with previous frame if we have enough keypoints
            matches = []
            if (len(keypoints) > 0 and len(self.last_keypoints) > 0 and
                descriptors is not None and self.last_descriptors is not None):
                
                # Create feature matcher
                matcher = cv2.BFMatcher()
                raw_matches = matcher.knnMatch(self.last_descriptors, descriptors, k=2)
                
                # Apply ratio test to find good matches
                threshold = self.config['vision']['feature_match_threshold']
                for m, n in raw_matches:
                    if m.distance < threshold * n.distance:
                        matches.append({
                            'queryIdx': m.queryIdx,
                            'trainIdx': m.trainIdx,
                            'distance': float(m.distance),
                            'point1': tuple(map(float, self.last_keypoints[m.queryIdx].pt)),
                            'point2': tuple(map(float, keypoints[m.trainIdx].pt))
                        })
            
            # Update stored features
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            
            return {
                'keypoints': self._format_keypoints(keypoints),
                'matches': matches
            }
            
        except Exception as e:
            self.logger.error(f"Feature tracking error: {e}")
            return {'keypoints': [], 'matches': []}
    
    def _format_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[Dict[str, float]]:
        """
        Convert OpenCV keypoints to serializable format.
        
        Args:
            keypoints (list): List of OpenCV KeyPoint objects
            
        Returns:
            list: List of formatted keypoint dictionaries
        """
        return [
            {
                'x': float(kp.pt[0]),
                'y': float(kp.pt[1]),
                'size': float(kp.size),
                'angle': float(kp.angle),
                'response': float(kp.response),
                'octave': int(kp.octave)
            }
            for kp in keypoints
        ]
    
    def get_movement_vectors(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate movement vectors from feature matches.
        
        Args:
            matches (list): List of feature matches
            
        Returns:
            list: List of movement vectors
        """
        vectors = []
        for match in matches:
            x1, y1 = match['point1']
            x2, y2 = match['point2']
            
            vectors.append({
                'start': (x1, y1),
                'end': (x2, y2),
                'displacement': (x2 - x1, y2 - y1),
                'magnitude': ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            })
        
        return vectors
    
    def reset(self) -> None:
        """Reset feature tracker state."""
        self.last_keypoints = None
        self.last_descriptors = None
        
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update feature tracking configuration.
        
        Args:
            new_config (dict): New configuration settings
        """
        try:
            if 'vision' in new_config:
                self.config['vision'].update(new_config['vision'])
            self.logger.info("Feature tracker configuration updated")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
