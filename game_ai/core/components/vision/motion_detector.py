"""Motion detection component for tracking movement between frames."""

import cv2
import numpy as np
from typing import Dict, List, Any
import logging

class MotionDetector:
    """Handles detection of motion between consecutive frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize motion detector component.
        
        Args:
            config (dict): Configuration settings including motion thresholds
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.last_gray = None
        self.motion_history = None
        
    def detect_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect motion between consecutive frames.
        
        Args:
            frame (numpy.ndarray): Current frame in BGR format
            
        Returns:
            dict: Motion detection results including regions and magnitude
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.last_gray is None:
                self.last_gray = gray
                return {'regions': [], 'magnitude': 0}
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(self.last_gray, gray)
            
            # Threshold to get motion mask
            _, motion_mask = cv2.threshold(
                frame_diff,
                self.config['vision']['motion_threshold'],
                255,
                cv2.THRESH_BINARY
            )
            
            # Find motion regions
            contours, _ = cv2.findContours(
                motion_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            motion_regions = []
            total_motion = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.config['vision']['min_contour_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_regions.append({
                        'position': (x + w//2, y + h//2),
                        'size': (w, h),
                        'area': area
                    })
                    total_motion += area
            
            # Calculate motion magnitude as percentage of frame area
            frame_area = gray.shape[0] * gray.shape[1]
            motion_magnitude = total_motion / frame_area if frame_area > 0 else 0
            
            # Update last frame
            self.last_gray = gray
            
            return {
                'regions': motion_regions,
                'magnitude': motion_magnitude,
                'frame_diff': frame_diff,  # For visualization if needed
                'motion_mask': motion_mask
            }
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            return {'regions': [], 'magnitude': 0}
            
    def reset(self) -> None:
        """Reset motion detector state."""
        self.last_gray = None
        self.motion_history = None
        
    def get_motion_history(self) -> np.ndarray:
        """
        Get accumulated motion history.
        
        Returns:
            numpy.ndarray: Motion history image or None if not available
        """
        return self.motion_history.copy() if self.motion_history is not None else None
        
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update motion detection configuration.
        
        Args:
            new_config (dict): New configuration settings
        """
        try:
            # Update only vision-related settings
            if 'vision' in new_config:
                self.config['vision'].update(new_config['vision'])
            self.logger.info("Motion detector configuration updated")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
