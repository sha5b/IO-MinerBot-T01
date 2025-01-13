
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
            'player': [(0, 100, 100), (10, 255, 255)],  # Red-ish for player character
            'enemy': [(0, 0, 50), (180, 50, 200)],      # Gray/Dark for hostile mobs
            'resource': [(20, 100, 100), (40, 255, 255)], # Yellow/Brown for resources like trees
            'terrain': [(35, 50, 50), (85, 255, 255)]   # Green-ish for terrain/blocks
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
            
            # Analyze detected objects
            analysis = {
                'environment': {
                    'terrain_type': self._analyze_terrain(objects, structure),
                    'terrain_analysis': structure,
                    'threats': [
                        {
                            'type': obj['type'],
                            'position': obj['position'],
                            'distance': self._calculate_distance(
                                obj['position'],
                                next((p['position'] for p in objects if p['type'] == 'player'), (0, 0))
                            )
                        }
                        for obj in objects if obj['type'] == 'enemy'
                    ],
                    'resources': [
                        {
                            'type': obj['type'],
                            'position': obj['position'],
                            'distance': self._calculate_distance(
                                obj['position'],
                                next((p['position'] for p in objects if p['type'] == 'player'), (0, 0))
                            )
                        }
                        for obj in objects if obj['type'] == 'resource'
                    ],
                    'passive_mobs': [
                        {
                            'type': obj['type'],
                            'position': obj['position'],
                            'distance': self._calculate_distance(
                                obj['position'],
                                next((p['position'] for p in objects if p['type'] == 'player'), (0, 0))
                            )
                        }
                        for obj in objects if obj['type'] not in ['enemy', 'resource', 'player']
                    ]
                },
                'player': {
                    'position': next((obj['position'] for obj in objects if obj['type'] == 'player'), None),
                    'status': self._analyze_player_status(objects, motion)
                }
            }

            # Structure and filter data for memory storage
            game_state = {
                'frame_shape': frame.shape,
                'timestamp': datetime.now().isoformat(),
                'objects': self._filter_relevant_objects(objects),
                'features': self._summarize_features(features),
                'motion': self._summarize_motion(motion),
                'structure': self._summarize_structure(structure),
                'analysis': self._enhance_analysis(analysis)
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
    
    def _analyze_terrain(self, objects: List[Dict[str, Any]], structure: Dict[str, Any]) -> str:
        """Analyze terrain type based on detected objects and structure."""
        # Count terrain objects
        terrain_objects = [obj for obj in objects if obj['type'] == 'terrain']
        
        if not terrain_objects:
            return 'unknown'
            
        # Analyze terrain distribution
        terrain_coverage = sum(obj['area'] for obj in terrain_objects)
        total_area = structure.get('edge_density', 0) * 100
        
        if terrain_coverage > total_area * 0.6:
            return 'dense'
        elif terrain_coverage > total_area * 0.3:
            return 'moderate'
        else:
            return 'sparse'
            
    def _analyze_player_status(self, objects: List[Dict[str, Any]], motion: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player status based on detected objects and motion."""
        player_objects = [obj for obj in objects if obj['type'] == 'player']
        
        if not player_objects:
            return {}
            
        # Basic status analysis
        status = {
            'moving': motion['magnitude'] > 0.1,
            'near_resources': any(
                obj['type'] == 'resource' and 
                self._calculate_distance(player_objects[0]['position'], obj['position']) < 100
                for obj in objects
            ),
            'near_threats': any(
                obj['type'] == 'enemy' and
                self._calculate_distance(player_objects[0]['position'], obj['position']) < 150
                for obj in objects
            )
        }
        
        return status
        
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two points."""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

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
            
    def _filter_relevant_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and prioritize relevant objects based on distance and type."""
        if not objects:
            return []
            
        # Sort objects by distance from player
        player_pos = next((obj['position'] for obj in objects if obj['type'] == 'player'), None)
        if player_pos:
            for obj in objects:
                if obj['type'] != 'player':
                    obj['distance'] = self._calculate_distance(obj['position'], player_pos)
            
            # Sort non-player objects by distance
            sorted_objects = [obj for obj in objects if obj['type'] == 'player']
            sorted_objects.extend(
                sorted([obj for obj in objects if obj['type'] != 'player'], 
                      key=lambda x: x.get('distance', float('inf')))
            )
            return sorted_objects[:20]  # Limit to 20 most relevant objects
        return objects[:20]

    def _summarize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize feature tracking data."""
        return {
            'keypoint_count': len(features.get('keypoints', [])),
            'match_count': len(features.get('matches', [])),
            'significant_changes': any(
                match['distance'] < self.config['vision']['feature_match_threshold'] * 0.5
                for match in features.get('matches', [])
            )
        }

    def _summarize_motion(self, motion: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize motion detection data."""
        regions = motion.get('regions', [])
        return {
            'magnitude': motion.get('magnitude', 0),
            'active_regions': len(regions),
            'significant_motion': motion.get('magnitude', 0) > 0.1
        }

    def _summarize_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize scene structure data."""
        lines = structure.get('lines', [])
        regions = structure.get('regions', [])
        
        # Analyze line orientations
        horizontal_lines = sum(1 for line in lines if abs(line['angle']) < 10 or abs(line['angle']) > 170)
        vertical_lines = sum(1 for line in lines if abs(line['angle'] - 90) < 10)
        
        return {
            'edge_density': structure.get('edge_density', 0),
            'line_count': len(lines),
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'region_count': len(regions),
            'complexity': len(lines) * structure.get('edge_density', 0)
        }

    def _enhance_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw analysis into meaningful tactical summaries."""
        env = analysis.get('environment', {})
        player = analysis.get('player', {})
        
        # Analyze threats
        threats = env.get('threats', [])
        threat_zones = {
            'immediate': sum(1 for t in threats if t.get('distance', float('inf')) < 100),
            'close': sum(1 for t in threats if 100 <= t.get('distance', float('inf')) < 300),
            'distant': sum(1 for t in threats if t.get('distance', float('inf')) >= 300)
        }
        
        # Analyze resources
        resources = env.get('resources', [])
        resource_zones = {
            'nearby': sum(1 for r in resources if r.get('distance', float('inf')) < 150),
            'reachable': sum(1 for r in resources if 150 <= r.get('distance', float('inf')) < 300)
        }
        
        # Analyze terrain
        terrain_type = env.get('terrain_type', 'unknown')
        terrain_advantage = 'neutral'
        if terrain_type == 'dense':
            terrain_advantage = 'favorable'  # Good cover
        elif terrain_type == 'sparse':
            terrain_advantage = 'unfavorable'  # Exposed
            
        # Analyze tactical position
        position_quality = 'neutral'
        if threat_zones['immediate'] == 0 and resource_zones['nearby'] > 0:
            position_quality = 'advantageous'
        elif threat_zones['immediate'] > 0 or (threat_zones['close'] > 2):
            position_quality = 'compromised'
            
        # Create tactical summary
        return {
            'environment': {
                'terrain_type': terrain_type,
                'terrain_advantage': terrain_advantage,
                'threat_assessment': {
                    'level': 'high' if threat_zones['immediate'] > 1 else 
                            'medium' if threat_zones['immediate'] > 0 or threat_zones['close'] > 2 else 
                            'low',
                    'distribution': threat_zones
                },
                'resource_assessment': {
                    'availability': 'high' if resource_zones['nearby'] > 2 else
                                  'medium' if resource_zones['nearby'] > 0 or resource_zones['reachable'] > 2 else
                                  'low',
                    'distribution': resource_zones
                }
            },
            'tactical_position': {
                'quality': position_quality,
                'mobility': 'restricted' if terrain_type == 'dense' else 'unrestricted',
                'defensibility': 'high' if terrain_type == 'dense' and threat_zones['immediate'] == 0 else
                               'low' if terrain_type == 'sparse' and threat_zones['immediate'] > 0 else
                               'medium'
            }
        }

    def cleanup(self):
        """Clean up resources."""
        if self.visualizer:
            self.visualizer.cleanup()
        self.llm.cleanup()
