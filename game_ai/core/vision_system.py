"""Vision system for capturing and analyzing game screen using OpenCV."""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .components.vision import (
    ScreenCapture,
    ObjectDetector,
    MotionDetector,
    FeatureTracker,
    SceneAnalyzer
)
from .visualizer import Visualizer
from .ollama_interface import OllamaInterface

class VisionSystem:
    """Handles all visual input processing and analysis using OpenCV."""
    
    def __init__(self, config_path: Optional[Path] = None, show_visualization: bool = True):
        """
        Initialize the vision system.
        
        Args:
            config_path (Path, optional): Path to configuration file
            show_visualization (bool): Whether to show visualization window
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.last_frame = None
        
        # Initialize components
        self.screen_capture = ScreenCapture(self.config)
        self.object_detector = ObjectDetector(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.feature_tracker = FeatureTracker(self.config)
        self.scene_analyzer = SceneAnalyzer(self.config)
        
        # Initialize visualization and LLM interface
        self.visualizer = Visualizer() if show_visualization else None
        self.llm = OllamaInterface(config_path)
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load vision system configuration."""
        default_config = {
            'capture': {
                'monitor': 1,
                'window_title': None,
                'fps': 30
            },
            'vision': {
                'motion_threshold': 25,
                'feature_match_threshold': 0.7,
                'min_contour_area': 100,
                'max_contour_area': 10000
            },
            'color_ranges': {
                'player': [(0, 100, 100), (10, 255, 255)],    # Red-ish
                'enemy': [(0, 0, 100), (180, 30, 255)],       # Gray/Dark
                'resource': [(20, 100, 100), (30, 255, 255)], # Yellow-ish
                'terrain': [(35, 50, 50), (85, 255, 255)]     # Green-ish
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f).get('vision', {})
                    return {**default_config, **loaded_config}
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                return default_config
        return default_config
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """
        Capture the current game screen.
        
        Returns:
            numpy.ndarray: Captured frame or None if capture failed
        """
        frame = self.screen_capture.capture_frame()
        if frame is not None:
            self.last_frame = frame
        return frame
    
    def process_frame(self, frame: Optional[np.ndarray] = None, current_objective: str = "") -> Dict[str, Any]:
        """
        Process the captured frame to extract game state.
        
        Args:
            frame (numpy.ndarray, optional): Frame to process, uses last captured if None
            current_objective (str): Current AI objective
            
        Returns:
            dict: Processed game state information
        """
        if frame is None:
            frame = self.last_frame
        if frame is None:
            raise ValueError("No frame available for processing")
            
        try:
            # Convert to HSV and grayscale for different analyses
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process frame using components
            objects = self.object_detector.detect_objects(frame)
            features = self.feature_tracker.track_features(frame)
            motion = self.motion_detector.detect_motion(frame)
            structure = self.scene_analyzer.analyze_scene(frame)
            
            # Combine all data
            game_state = {
                'frame_shape': frame.shape,
                'timestamp': datetime.now().isoformat(),
                'objects': objects,
                'features': features,
                'motion': motion,
                'structure': structure
            }
            
            # Update visualization
            if self.visualizer:
                try:
                    self.visualizer.show_frame(
                        frame,
                        objects,
                        current_objective
                    )
                except Exception as e:
                    self.logger.error(f"Visualization error: {e}")
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            raise
    
    def get_available_windows(self) -> List[str]:
        """
        Get list of available window titles.
        
        Returns:
            list: List of window titles
        """
        return self.screen_capture.get_available_windows()

    def set_capture_window(self, title: str) -> bool:
        """
        Set window to capture by title.
        
        Args:
            title (str): Window title
            
        Returns:
            bool: True if window was found and set
        """
        return self.screen_capture.set_capture_window(title)

    def set_capture_monitor(self, monitor_num: int) -> None:
        """
        Set monitor to capture.
        
        Args:
            monitor_num (int): Monitor number
        """
        self.config['capture']['monitor'] = monitor_num
            
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.screen_capture.cleanup()
            if self.visualizer:
                self.visualizer.cleanup()
            self.llm.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
