"""Visualization module for displaying AI's view and detections."""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional

class Visualizer:
    """Handles visualization of AI's view and detections."""
    
    def __init__(self, scale: float = 0.5):
        """
        Initialize the visualizer.
        
        Args:
            scale (float): Scale factor for display (0.5 = half size)
        """
        self.scale = scale
        self.logger = logging.getLogger(__name__)
        # Create single window
        self.window_name = "AI Vision"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, 800, 600)  # Set default size
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
    
    def show_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                  current_objective: str = "") -> None:
        """
        Display frame with detections and current objective.
        
        Args:
            frame (np.ndarray): Current frame to display
            detections (list): List of YOLO detections
            current_objective (str): Current AI objective
        """
        try:
            # Create a copy of the frame for visualization
            display_frame = frame.copy()
            
            # Draw detections with different colors for different classes
            colors = {
                'tree': (0, 255, 0),      # Green for resources
                'stone': (128, 128, 128),  # Gray for resources
                'zombie': (0, 0, 255),     # Red for threats
                'skeleton': (0, 0, 255),
                'creeper': (0, 0, 255),
                'cow': (255, 255, 0),      # Yellow for passive mobs
                'pig': (255, 255, 0),
                'sheep': (255, 255, 0)
            }
            
            # Draw detections
            for det in detections:
                bbox = det.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    obj_class = det.get('class', 'unknown')
                    color = colors.get(obj_class, (0, 255, 0))  # Default to green
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    label = f"{obj_class} {det.get('confidence', 0):.2f}"
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(display_frame, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1),
                                color, -1)  # Filled rectangle
                    # Draw label text
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add current objective with background
            if current_objective:
                objective_text = f"Objective: {current_objective}"
                # Draw objective background
                text_size = cv2.getTextSize(objective_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(display_frame, 
                            (10, 10), 
                            (10 + text_size[0], 40),
                            (0, 0, 0), -1)  # Black background
                # Draw objective text
                cv2.putText(display_frame, objective_text,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize frame for display
            height, width = display_frame.shape[:2]
            new_height = int(height * self.scale)
            new_width = int(width * self.scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            # Show frame with detections
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(self.window_name, display_frame)
            cv2.waitKey(1)  # Brief display update
                
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
    
    def cleanup(self) -> None:
        """Clean up OpenCV windows."""
        try:
            cv2.destroyWindow(self.window_name)
            cv2.destroyAllWindows()
            # Force window cleanup
            cv2.waitKey(1)
        except Exception as e:
            print(f"Cleanup error: {e}")
