"""Vision system for capturing and analyzing game screen."""

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from .window_manager import WindowManager
from .visualizer import Visualizer

class VisionSystem:
    """Handles all visual input processing and analysis."""
    
    def __init__(self, config_path: Optional[Path] = None, show_visualization: bool = True):
        """
        Initialize the vision system.
        
        Args:
            config_path (Path, optional): Path to configuration file
            show_visualization (bool): Whether to show visualization window
        """
        self.logger = logging.getLogger(__name__)
        self.sct = mss()
        self.model = None
        self.config = self._load_config(config_path)
        self.last_frame = None
        self.window_manager = WindowManager()
        self.visualizer = Visualizer() if show_visualization else None
        self._initialize_yolo()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load vision system configuration."""
        default_config = {
            'monitor': 1,
            'window_title': None,  # If set, captures specific window instead of monitor
            'confidence_threshold': 0.5,
            'yolo': {
                'model_path': 'models/yolov8n.pt',
                'confidence': 0.25,
                'iou': 0.45
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
    
    def _initialize_yolo(self) -> None:
        """Initialize YOLO model for object detection."""
        try:
            model_path = Path(self.config['yolo']['model_path'])
            if model_path.exists():
                # Configure YOLO to suppress debug output
                import logging
                logging.getLogger("ultralytics").setLevel(logging.WARNING)
                
                # Initialize YOLO with configuration
                self.model = YOLO(str(model_path))
                # Configure model parameters
                self.model.conf = float(self.config['yolo']['confidence'])
                self.model.iou = float(self.config['yolo']['iou'])
                # Suppress YOLO output
                self.model.predictor = None  # Reset predictor to avoid verbose output
                self.logger.info("YOLO model loaded successfully")
            else:
                self.logger.warning(f"YOLO model not found at {model_path}")
        except Exception as e:
            self.logger.error(f"Error initializing YOLO: {e}")
    
    def capture_screen(self) -> np.ndarray:
        """Capture the current game screen."""
        try:
            # If window title is set, try window capture first
            if self.config.get('window_title'):
                if not self.window_manager.active_window or \
                   self.window_manager.get_window_title() != self.config['window_title']:
                    # Set or update active window
                    if not self.window_manager.set_active_window(self.config['window_title']):
                        self.logger.warning(f"Window '{self.config['window_title']}' not found")
                        return self._capture_monitor()
                
                try:
                    # Try window capture
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
            # Use configured monitor or fall back to primary
            monitor_num = self.config.get('monitor', 1)
            if monitor_num >= len(self.sct.monitors):
                monitor_num = 1  # Fall back to primary monitor
                
            monitor = self.sct.monitors[monitor_num]
            with mss() as sct:  # Create new mss instance to ensure clean capture
                screenshot = sct.grab(monitor)
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
            # Basic frame processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Detect objects first for visualization
            objects = self.detect_objects(frame)
            
            # Extract game state
            game_state = {
                'frame_shape': frame.shape,
                'timestamp': datetime.now().isoformat(),
                'player': self._detect_player(frame),
                'inventory': {
                    'items': {},  # Will be populated by UI analysis
                    'capacity': 36,
                    'low_thresholds': {'wood': 10, 'stone': 10, 'food': 5}
                },
                'analysis': {
                    'player': self._detect_player(frame),
                    'environment': self._analyze_environment(frame, edges),
                    'objects': objects,
                    'ui_elements': self._detect_ui_elements(frame)
                }
            }
            
            # Update inventory based on UI analysis
            ui_elements = self._detect_ui_elements(frame)
            if ui_elements.get('elements'):
                # Basic inventory estimation from UI elements
                game_state['inventory']['items'] = {
                    'wood': len([e for e in ui_elements['elements'] if e.get('type') == 'wood_item']),
                    'stone': len([e for e in ui_elements['elements'] if e.get('type') == 'stone_item']),
                    'food': len([e for e in ui_elements['elements'] if e.get('type') == 'food_item'])
                }
            
            # Update visualization with error handling
            if self.visualizer:
                try:
                    self.visualizer.show_frame(frame, objects, current_objective)
                except Exception as e:
                    self.logger.error(f"Visualization error: {e}")
            
            return game_state
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            raise

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and classify objects in the frame using YOLO."""
        if self.model is None:
            return []
            
        try:
            results = self.model(
                frame,
                conf=self.config['yolo']['confidence'],
                iou=self.config['yolo']['iou']
            )
            
            detections = []
            for result in results:
                for box in result.boxes:
                    obj = {
                        'class': result.names[int(box.cls[0].item())],
                        'confidence': float(box.conf[0].item()),
                        'bbox': [float(x.item()) for x in box.xyxy[0]],
                        'center': self._calculate_center([float(x.item()) for x in box.xyxy[0]])
                    }
                    detections.append(obj)
            
            return detections
        except Exception as e:
            self.logger.error(f"Object detection error: {e}")
            return []

    def _detect_player(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect player position and state."""
        if self.model:
            results = self.detect_objects(frame)
            for obj in results:
                if obj['class'] == 'person':  # Assuming player is detected as person
                    return {
                        'detected': True,
                        'position': obj['center'],
                        'confidence': obj['confidence'],
                        'bbox': obj['bbox']
                    }
        
        return {
            'detected': False,
            'position': None,
            'confidence': 0.0,
            'bbox': None
        }
    
    def _analyze_environment(self, frame: np.ndarray, edges: np.ndarray) -> Dict[str, Any]:
        """Analyze the game environment."""
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Analyze colors for terrain understanding
            color_ranges = {
                'grass': ([35, 50, 50], [85, 255, 255]),
                'water': ([90, 50, 50], [130, 255, 255]),
                'stone': ([0, 0, 30], [180, 30, 150]),
                'wood': ([10, 50, 50], [30, 255, 255])
            }
            
            # Detect terrain types
            terrain_analysis = {}
            for terrain, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                coverage = np.count_nonzero(mask) / (frame.shape[0] * frame.shape[1])
                terrain_analysis[terrain] = coverage
            
            # Detect objects using YOLO
            objects = self.detect_objects(frame)
            
            # Categorize detected objects
            threats = []
            resources = []
            passive_mobs = []
            
            for obj in objects:
                if obj['class'] in ['zombie', 'skeleton', 'creeper']:
                    threats.append({
                        'type': obj['class'],
                        'position': obj['center'],
                        'distance': self._calculate_distance(obj['center'], (frame.shape[1]/2, frame.shape[0]/2))
                    })
                elif obj['class'] in ['tree', 'stone']:
                    resources.append({
                        'type': obj['class'],
                        'position': obj['center'],
                        'distance': self._calculate_distance(obj['center'], (frame.shape[1]/2, frame.shape[0]/2))
                    })
                elif obj['class'] in ['cow', 'pig', 'sheep']:
                    passive_mobs.append({
                        'type': obj['class'],
                        'position': obj['center'],
                        'distance': self._calculate_distance(obj['center'], (frame.shape[1]/2, frame.shape[0]/2))
                    })
            
            # Analyze lighting conditions
            lighting = self._analyze_lighting(frame)
            
            # Structure environment data according to game rules
            return {
                'terrain_type': max(terrain_analysis.items(), key=lambda x: x[1])[0],
                'terrain_analysis': terrain_analysis,
                'threats': threats,
                'resources': resources,
                'passive_mobs': passive_mobs,
                'lighting_conditions': lighting,
                'edge_density': np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1]),
                'obstacles': self._detect_obstacles(edges),
                'time_of_day': self._estimate_time_of_day(lighting['brightness'])
            }
            
        except Exception as e:
            self.logger.error(f"Environment analysis error: {e}")
            # Return safe default structure
            return {
                'terrain_type': 'unknown',
                'terrain_analysis': {},
                'threats': [],
                'resources': [],
                'passive_mobs': [],
                'lighting_conditions': {'brightness': 0.0, 'contrast': 0.0, 'saturation': 0.0},
                'edge_density': 0.0,
                'obstacles': [],
                'time_of_day': 'day'
            }
            
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        
    def _detect_obstacles(self, edges: np.ndarray) -> List[Dict[str, Any]]:
        """Detect obstacles from edge detection."""
        obstacles = []
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append({
                    'position': (x + w/2, y + h/2),
                    'size': (w, h),
                    'area': area
                })
        
        return obstacles
        
    def _estimate_time_of_day(self, brightness: float) -> str:
        """Estimate time of day based on brightness."""
        if brightness > 150:
            return 'day'
        elif brightness > 100:
            return 'dusk'
        else:
            return 'night'
    
    def _analyze_lighting(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze scene lighting conditions."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            contrast = np.std(hsv[:, :, 2])
            saturation = np.mean(hsv[:, :, 1])
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'saturation': float(saturation)
            }
        except Exception as e:
            self.logger.error(f"Lighting analysis error: {e}")
            return {
                'brightness': 0.0,
                'contrast': 0.0,
                'saturation': 0.0
            }
    
    def _calculate_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _detect_ui_elements(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect and analyze UI elements."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ui_elements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                if w/h > 0.1 and w/h < 10:  # Filter extreme aspect ratios
                    ui_elements.append({
                        'type': 'unknown',
                        'position': (x, y),
                        'size': (w, h),
                        'area': area
                    })
        
        return {
            'elements': ui_elements,
            'count': len(ui_elements)
        }

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
            self.config['window_title'] = None  # Clear window capture
        else:
            raise ValueError(f"Invalid monitor number: {monitor_num}")
            
    def cleanup(self):
        """Clean up resources."""
        if self.visualizer:
            self.visualizer.cleanup()
