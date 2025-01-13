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
            
            # Transform raw data into structured game state
            game_state = self._transform_game_state(
                frame.shape,
                objects,
                features,
                motion,
                structure
            )
            
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
            
    def _transform_game_state(self, frame_shape: tuple, objects: List[Dict[str, Any]],
                            features: List[Dict[str, Any]], motion: Dict[str, Any],
                            structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw vision data into structured game state.
        
        Args:
            frame_shape: Shape of the frame
            objects: Detected objects
            features: Tracked features
            motion: Motion detection results
            structure: Scene structure analysis
            
        Returns:
            dict: Structured game state
        """
        try:
            # Extract player information
            player_objects = [obj for obj in objects if obj.get('class') == 'player']
            player = {
                'health': 100,  # Default value, should be updated from UI elements
                'position': player_objects[0]['position'] if player_objects else (0, 0),
                'status': []
            }
            
            # Extract threats
            threats = []
            for obj in objects:
                if obj.get('class') in ['zombie', 'skeleton', 'creeper']:
                    threats.append({
                        'type': obj['class'],
                        'distance': self._calculate_distance(player['position'], obj['position']),
                        'position': obj['position'],
                        'threat_level': 0.8 if obj['class'] == 'creeper' else 0.6
                    })
            
            # Extract resources and opportunities
            resources = []
            opportunities = []
            for obj in objects:
                if obj.get('class') in ['tree', 'stone', 'water']:
                    resources.append({
                        'type': obj['class'],
                        'position': obj['position'],
                        'distance': self._calculate_distance(player['position'], obj['position'])
                    })
                    opportunities.append({
                        'type': f'gather_{obj["class"]}',
                        'position': obj['position'],
                        'distance': self._calculate_distance(player['position'], obj['position']),
                        'value': 0.7
                    })
            
            # Structure environment information
            environment = {
                'terrain_type': self._determine_terrain_type(structure),
                'visibility': 1.0 - motion.get('blur_factor', 0),
                'threats': threats,
                'opportunities': opportunities,
                'obstacles': self._extract_obstacles(structure),
                'explored_chunks': 1,  # Placeholder
                'total_chunks': 100    # Placeholder
            }
            
            # Structure inventory information
            inventory = {
                'items': {},  # Should be updated from UI elements
                'capacity': 100,
                'thresholds': {
                    'wood': 25,
                    'stone': 25,
                    'food': 25
                }
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'player': player,
                'environment': environment,
                'inventory': inventory
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming game state: {e}")
            return {
                'player': {'health': 100, 'position': (0, 0), 'status': []},
                'environment': {'terrain_type': 'unknown', 'visibility': 1.0},
                'inventory': {'items': {}, 'capacity': 100}
            }
    
    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate Euclidean distance between two points."""
        return float(np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2))
    
    def _determine_terrain_type(self, structure: Dict[str, Any]) -> str:
        """Determine terrain type from scene structure."""
        edge_density = structure.get('edge_density', 0)
        if edge_density > 0.3:
            return 'difficult'
        elif edge_density > 0.1:
            return 'normal'
        else:
            return 'easy'
    
    def _extract_obstacles(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract obstacles from scene structure."""
        obstacles = []
        for line in structure.get('lines', []):
            if line['length'] > 50:  # Only consider significant lines as obstacles
                obstacles.append({
                    'position': line['start'],
                    'size': (line['length'], 10)  # Approximate size
                })
        return obstacles
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.screen_capture.cleanup()
            if self.visualizer:
                self.visualizer.cleanup()
            self.llm.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
