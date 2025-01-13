"""Vision system for capturing and analyzing game screen using OpenCV."""

import cv2
import numpy as np
from mss import mss
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from .window_manager import WindowManager
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
        self.sct = mss()
        self.config = self._load_config(config_path)
        self.last_frame = None
        self.window_manager = WindowManager()
        self.visualizer = Visualizer() if show_visualization else None
        self.llm = OllamaInterface(config_path)
        
        # Initialize feature tracking
        self.feature_detector = cv2.SIFT_create()
        self.last_keypoints = None
        self.last_descriptors = None
        
        # Initialize motion detection
        self.last_gray = None
        self.motion_history = None
        
        # Color ranges for different elements
        self.color_ranges = {
            'player': [(0, 100, 100), (10, 255, 255)],  # Red-ish
            'enemy': [(0, 0, 100), (180, 30, 255)],     # Gray/Dark
            'resource': [(20, 100, 100), (30, 255, 255)],  # Yellow-ish
            'terrain': [(35, 50, 50), (85, 255, 255)]   # Green-ish
        }
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load vision system configuration."""
        default_config = {
            'monitor': 1,
            'window_title': None,
            'vision': {
                'motion_threshold': 25,
                'feature_match_threshold': 0.7,
                'min_contour_area': 100,
                'max_contour_area': 10000
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
    
    def capture_screen(self) -> np.ndarray:
        """Capture the current game screen."""
        try:
            # If window title is set, try window capture first
            if self.config.get('window_title'):
                if not self.window_manager.active_window or \
                   self.window_manager.get_window_title() != self.config['window_title']:
                    if not self.window_manager.set_active_window(self.config['window_title']):
                        self.logger.warning(f"Window '{self.config['window_title']}' not found")
                        return self._capture_monitor()
                
                try:
                    frame = self.window_manager.capture_window()
                    if frame is not None and frame.size > 0:
                        self.last_frame = frame
                        return frame
                except Exception as e:
                    self.logger.error(f"Window capture error: {e}")
                    return self._capture_monitor()
            
            return self._capture_monitor()
            
        except Exception as e:
            self.logger.error(f"Screen capture error: {e}")
            raise

    def _capture_monitor(self) -> np.ndarray:
        """Capture monitor screen using mss."""
        try:
            monitor_num = self.config.get('monitor', 1)
            if monitor_num >= len(self.sct.monitors):
                monitor_num = 1
                
            monitor = self.sct.monitors[monitor_num]
            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return self.last_frame
                
        except Exception as e:
            self.logger.error(f"Monitor capture error: {e}")
            raise

    def process_frame(self, frame: Optional[np.ndarray] = None, current_objective: str = "") -> Dict[str, Any]:
        """Process the captured frame to extract game state."""
        if frame is None:
            frame = self.last_frame
        if frame is None:
            raise ValueError("No frame available for processing")
            
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Convert to grayscale for feature/motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect objects by color
            objects = self._detect_objects_by_color(hsv)
            
            # Track features
            features = self._track_features(gray)
            
            # Detect motion
            motion = self._detect_motion(gray)
            
            # Analyze scene structure
            structure = self._analyze_scene_structure(gray)
            
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
    
    def _detect_objects_by_color(self, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using color ranges."""
        objects = []
        
        for obj_type, (lower, upper) in self.color_ranges.items():
            # Create mask for this color range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.config['vision']['min_contour_area'] < area < self.config['vision']['max_contour_area']:
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
    
    def _track_features(self, gray: np.ndarray) -> Dict[str, Any]:
        """Track distinctive features between frames."""
        try:
            # Detect keypoints
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if self.last_keypoints is None or self.last_descriptors is None:
                self.last_keypoints = keypoints
                self.last_descriptors = descriptors
                return {'keypoints': [], 'matches': []}
            
            # Match features with previous frame
            if len(keypoints) > 0 and len(self.last_keypoints) > 0 and \
               descriptors is not None and self.last_descriptors is not None:
                
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(self.last_descriptors, descriptors, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < self.config['vision']['feature_match_threshold'] * n.distance:
                        good_matches.append({
                            'queryIdx': m.queryIdx,
                            'trainIdx': m.trainIdx,
                            'distance': float(m.distance)
                        })
                
                # Update last frame data
                self.last_keypoints = keypoints
                self.last_descriptors = descriptors
                
                return {
                    'keypoints': [
                        {'x': kp.pt[0], 'y': kp.pt[1], 'size': kp.size, 'angle': kp.angle}
                        for kp in keypoints
                    ],
                    'matches': good_matches
                }
            
            return {'keypoints': [], 'matches': []}
            
        except Exception as e:
            self.logger.error(f"Feature tracking error: {e}")
            return {'keypoints': [], 'matches': []}
    
    def _detect_motion(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect motion between frames."""
        try:
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
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
            
            # Update last frame
            self.last_gray = gray
            
            return {
                'regions': motion_regions,
                'magnitude': total_motion / (gray.shape[0] * gray.shape[1])
            }
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            return {'regions': [], 'magnitude': 0}
    
    def _analyze_scene_structure(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze overall scene structure."""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
            
            detected_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    detected_lines.append({
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'angle': float(angle),
                        'length': float(length)
                    })
            
            # Analyze regions
            regions = []
            height, width = gray.shape
            grid_size = 3
            
            cell_h = height // grid_size
            cell_w = width // grid_size
            
            for y in range(grid_size):
                for x in range(grid_size):
                    region = gray[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
                    
                    # Calculate region features
                    mean_intensity = np.mean(region)
                    std_intensity = np.std(region)
                    
                    regions.append({
                        'position': (x * cell_w + cell_w//2, y * cell_h + cell_h//2),
                        'size': (cell_w, cell_h),
                        'intensity': {
                            'mean': float(mean_intensity),
                            'std': float(std_intensity)
                        }
                    })
            
            return {
                'lines': detected_lines,
                'regions': regions,
                'edge_density': np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
            }
            
        except Exception as e:
            self.logger.error(f"Scene structure analysis error: {e}")
            return {'lines': [], 'regions': [], 'edge_density': 0}
    
    def get_available_windows(self) -> List[str]:
        """Get list of available window titles."""
        return self.window_manager.get_window_list()

    def set_capture_window(self, title: str) -> bool:
        """Set window to capture by title."""
        if self.window_manager.set_active_window(title):
            self.config['window_title'] = title
            return True
        return False

    def set_capture_monitor(self, monitor_num: int) -> None:
        """Set monitor to capture."""
        if 0 <= monitor_num < len(self.sct.monitors):
            self.config['monitor'] = monitor_num
            self.config['window_title'] = None
        else:
            raise ValueError(f"Invalid monitor number: {monitor_num}")
            
    def cleanup(self):
        """Clean up resources."""
        if self.visualizer:
            self.visualizer.cleanup()
        self.llm.cleanup()
