"""Scene analysis component for analyzing overall scene structure."""

import cv2
import numpy as np
from typing import Dict, List, Any
import logging

class SceneAnalyzer:
    """Handles analysis of scene structure and composition."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scene analyzer component.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
    def analyze_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze overall scene structure.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            dict: Scene analysis results including lines, regions, and edge density
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Line detection using Hough transform
            lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180, 
                50,
                minLineLength=100,
                maxLineGap=10
            )
            
            # Process detected lines
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
            regions = self._analyze_regions(gray)
            
            # Calculate edge density
            edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
            
            return {
                'lines': detected_lines,
                'regions': regions,
                'edge_density': float(edge_density),
                'edges': edges  # For visualization if needed
            }
            
        except Exception as e:
            self.logger.error(f"Scene analysis error: {e}")
            return {
                'lines': [],
                'regions': [],
                'edge_density': 0.0
            }
    
    def _analyze_regions(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze image regions for intensity patterns.
        
        Args:
            gray (numpy.ndarray): Grayscale image
            
        Returns:
            list: List of region analysis results
        """
        regions = []
        height, width = gray.shape
        grid_size = 3  # 3x3 grid
        
        cell_h = height // grid_size
        cell_w = width // grid_size
        
        for y in range(grid_size):
            for x in range(grid_size):
                # Extract region
                region = gray[
                    y*cell_h:(y+1)*cell_h,
                    x*cell_w:(x+1)*cell_w
                ]
                
                # Calculate region features
                mean_intensity = float(np.mean(region))
                std_intensity = float(np.std(region))
                
                # Calculate texture features using GLCM
                texture_features = self._calculate_texture_features(region)
                
                regions.append({
                    'position': (
                        x * cell_w + cell_w//2,
                        y * cell_h + cell_h//2
                    ),
                    'size': (cell_w, cell_h),
                    'intensity': {
                        'mean': mean_intensity,
                        'std': std_intensity
                    },
                    'texture': texture_features
                })
        
        return regions
    
    def _calculate_texture_features(self, region: np.ndarray) -> Dict[str, float]:
        """
        Calculate texture features for a region using GLCM.
        
        Args:
            region (numpy.ndarray): Image region
            
        Returns:
            dict: Texture features
        """
        try:
            # Normalize region to 8-bit
            region_8bit = cv2.normalize(
                region,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
            
            # Calculate gradient magnitude
            gradient_x = cv2.Sobel(region_8bit, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(region_8bit, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            return {
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'entropy': float(self._calculate_entropy(region_8bit))
            }
            
        except Exception as e:
            self.logger.error(f"Texture analysis error: {e}")
            return {
                'gradient_mean': 0.0,
                'gradient_std': 0.0,
                'entropy': 0.0
            }
    
    def _calculate_entropy(self, region: np.ndarray) -> float:
        """
        Calculate entropy of an image region.
        
        Args:
            region (numpy.ndarray): Image region
            
        Returns:
            float: Entropy value
        """
        histogram = cv2.calcHist([region], [0], None, [256], [0, 256])
        histogram = histogram.ravel() / histogram.sum()
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))
    
    def get_dominant_lines(self, lines: List[Dict[str, Any]], 
                          angle_threshold: float = 10.0) -> List[Dict[str, Any]]:
        """
        Find dominant lines by clustering similar angles.
        
        Args:
            lines (list): Detected lines
            angle_threshold (float): Angle difference threshold for clustering
            
        Returns:
            list: Dominant lines
        """
        if not lines:
            return []
            
        # Group lines by similar angles
        angle_groups = {}
        for line in lines:
            angle = line['angle']
            grouped = False
            
            for base_angle in angle_groups:
                if abs(angle - base_angle) < angle_threshold:
                    angle_groups[base_angle].append(line)
                    grouped = True
                    break
                    
            if not grouped:
                angle_groups[angle] = [line]
        
        # Find longest line in each group
        dominant_lines = []
        for group in angle_groups.values():
            longest_line = max(group, key=lambda x: x['length'])
            dominant_lines.append(longest_line)
        
        return dominant_lines
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update scene analyzer configuration.
        
        Args:
            new_config (dict): New configuration settings
        """
        try:
            if 'vision' in new_config:
                self.config['vision'].update(new_config['vision'])
            self.logger.info("Scene analyzer configuration updated")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
