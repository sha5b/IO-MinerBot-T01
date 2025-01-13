"""Object detection component for identifying game elements using color ranges."""

import cv2
import numpy as np
from typing import Dict, List, Any
import logging

class ObjectDetector:
    """Handles detection of game objects using color-based segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize object detector component.
        
        Args:
            config (dict): Configuration settings including color ranges and thresholds
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Default color ranges if not provided in config
        self.color_ranges = config.get('color_ranges', {
            'player': [(0, 100, 100), (10, 255, 255)],    # Red-ish
            'enemy': [(0, 0, 100), (180, 30, 255)],       # Gray/Dark
            'resource': [(20, 100, 100), (30, 255, 255)], # Yellow-ish
            'terrain': [(35, 50, 50), (85, 255, 255)]     # Green-ish
        })
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the frame using color segmentation.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            list: List of detected objects with their properties
        """
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            objects = []
            
            for obj_type, (lower, upper) in self.color_ranges.items():
                # Create mask for this color range
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Find contours
                contours, _ = cv2.findContours(
                    mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Process each contour
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if (self.config['vision']['min_contour_area'] < 
                        area < 
                        self.config['vision']['max_contour_area']):
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        center = (x + w//2, y + h//2)
                        
                        objects.append({
                            'type': obj_type,
                            'position': center,
                            'size': (w, h),
                            'area': area,
                            'bbox': [x, y, x + w, y + h]
                        })
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Object detection error: {e}")
            return []
            
    def update_color_range(self, obj_type: str, lower: tuple, upper: tuple) -> None:
        """
        Update color range for a specific object type.
        
        Args:
            obj_type (str): Type of object to update
            lower (tuple): Lower HSV bounds (h, s, v)
            upper (tuple): Upper HSV bounds (h, s, v)
        """
        try:
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError("Color bounds must be 3-element tuples (H,S,V)")
                
            self.color_ranges[obj_type] = [lower, upper]
            self.logger.info(f"Updated color range for {obj_type}")
            
        except Exception as e:
            self.logger.error(f"Error updating color range: {e}")
            
    def get_color_ranges(self) -> Dict[str, List[tuple]]:
        """
        Get current color ranges for all object types.
        
        Returns:
            dict: Dictionary of object types and their color ranges
        """
        return self.color_ranges.copy()
