"""Spatial memory component for managing world state and navigation."""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict

class ChunkCoord:
    """Represents a chunk coordinate in 3D space."""
    
    def __init__(self, x: int, y: int, z: int):
        """
        Initialize chunk coordinate.
        
        Args:
            x (int): X coordinate of the chunk
            y (int): Y coordinate of the chunk
            z (int): Z coordinate of the chunk
        """
        self.x = x
        self.y = y
        self.z = z
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __eq__(self, other):
        if not isinstance(other, ChunkCoord):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class SpatialMemory:
    """Manages spatial information and navigation in the game world."""
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        """
        Initialize spatial memory system.
        
        Args:
            config (dict): Configuration settings
            storage_path (Path): Path for spatial data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.storage_path = storage_path / 'spatial'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Spatial settings
        self.chunk_size = config.get('chunk_size', 16)
        self.max_path_history = config.get('max_path_history', 1000)
        
        # Spatial data structures
        self.chunks: Dict[ChunkCoord, Dict[str, Any]] = {}
        self.points_of_interest: Dict[str, Dict[str, Any]] = {}
        self.explored_chunks: Set[ChunkCoord] = set()
        self.path_history: List[Tuple[float, float, float]] = []
        self.block_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        
        # Load existing spatial data
        self._load_spatial_data()
    
    def update_block(self, x: int, y: int, z: int, block_data: Dict[str, Any]) -> None:
        """
        Update information about a block at given coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            z (int): Z coordinate
            block_data (dict): Information about the block
        """
        try:
            chunk_coord = self.get_chunk_coord(x, y, z)
            if chunk_coord not in self.chunks:
                self.chunks[chunk_coord] = {'blocks': {}}
            
            block_key = (x % self.chunk_size, y % self.chunk_size, z % self.chunk_size)
            self.chunks[chunk_coord]['blocks'][str(block_key)] = block_data
            self.block_cache[(x, y, z)] = block_data
            
            # Persist changes
            self._persist_spatial_data()
            
        except Exception as e:
            self.logger.error(f"Error updating block: {e}")
    
    def add_point_of_interest(self, name: str, x: float, y: float, z: float, 
                            poi_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a point of interest to memory.
        
        Args:
            name (str): Name of the point of interest
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            poi_type (str): Type of point of interest
            metadata (dict, optional): Additional information about the POI
        """
        try:
            self.points_of_interest[name] = {
                'position': (x, y, z),
                'type': poi_type,
                'metadata': metadata or {},
                'last_visited': datetime.now().isoformat()
            }
            
            # Persist changes
            self._persist_spatial_data()
            
        except Exception as e:
            self.logger.error(f"Error adding point of interest: {e}")
    
    def update_position(self, x: float, y: float, z: float) -> None:
        """
        Update the AI's position and track explored areas.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
        """
        try:
            self.path_history.append((x, y, z))
            if len(self.path_history) > self.max_path_history:
                self.path_history = self.path_history[-self.max_path_history:]
            
            chunk_coord = self.get_chunk_coord(x, y, z)
            self.explored_chunks.add(chunk_coord)
            
            # Persist changes
            self._persist_spatial_data()
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    def get_chunk_coord(self, x: float, y: float, z: float) -> ChunkCoord:
        """
        Convert world coordinates to chunk coordinates.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            
        Returns:
            ChunkCoord: Chunk coordinates
        """
        return ChunkCoord(
            int(x // self.chunk_size),
            int(y // self.chunk_size),
            int(z // self.chunk_size)
        )
    
    def get_nearby_blocks(self, x: float, y: float, z: float, 
                         radius: int) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
        """
        Get information about blocks within a radius.
        
        Args:
            x (float): Center X coordinate
            y (float): Center Y coordinate
            z (float): Center Z coordinate
            radius (int): Search radius
            
        Returns:
            dict: Dictionary of block positions and their data
        """
        nearby_blocks = {}
        try:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        pos = (int(x + dx), int(y + dy), int(z + dz))
                        if pos in self.block_cache:
                            nearby_blocks[pos] = self.block_cache[pos]
        except Exception as e:
            self.logger.error(f"Error getting nearby blocks: {e}")
            
        return nearby_blocks
    
    def get_nearest_poi(self, x: float, y: float, z: float, 
                       poi_type: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find the nearest point of interest.
        
        Args:
            x (float): Current X coordinate
            y (float): Current Y coordinate
            z (float): Current Z coordinate
            poi_type (str, optional): Type of POI to search for
            
        Returns:
            tuple: POI name and data, or None if not found
        """
        try:
            nearest = None
            min_dist = float('inf')
            
            for name, poi in self.points_of_interest.items():
                if poi_type and poi['type'] != poi_type:
                    continue
                    
                px, py, pz = poi['position']
                dist = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    nearest = (name, poi)
            
            return nearest
            
        except Exception as e:
            self.logger.error(f"Error finding nearest POI: {e}")
            return None
    
    def get_unexplored_directions(self, x: float, y: float, z: float) -> List[Tuple[float, float, float]]:
        """
        Get vectors pointing towards unexplored areas.
        
        Args:
            x (float): Current X coordinate
            y (float): Current Y coordinate
            z (float): Current Z coordinate
            
        Returns:
            list: List of normalized direction vectors
        """
        try:
            current_chunk = self.get_chunk_coord(x, y, z)
            unexplored_vectors = []
            
            # Check surrounding chunks
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                            
                        neighbor = ChunkCoord(
                            current_chunk.x + dx,
                            current_chunk.y + dy,
                            current_chunk.z + dz
                        )
                        
                        if neighbor not in self.explored_chunks:
                            # Calculate center of unexplored chunk
                            center_x = (neighbor.x * self.chunk_size) + (self.chunk_size / 2)
                            center_y = (neighbor.y * self.chunk_size) + (self.chunk_size / 2)
                            center_z = (neighbor.z * self.chunk_size) + (self.chunk_size / 2)
                            
                            # Calculate direction vector
                            dx = center_x - x
                            dy = center_y - y
                            dz = center_z - z
                            
                            # Normalize vector
                            magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5
                            if magnitude > 0:
                                unexplored_vectors.append((
                                    dx / magnitude,
                                    dy / magnitude,
                                    dz / magnitude
                                ))
            
            return unexplored_vectors
            
        except Exception as e:
            self.logger.error(f"Error getting unexplored directions: {e}")
            return []
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about world exploration.
        
        Returns:
            dict: Statistics about explored areas and resources
        """
        try:
            stats = {
                'explored_chunks': len(self.explored_chunks),
                'total_blocks_seen': len(self.block_cache),
                'points_of_interest': len(self.points_of_interest),
                'path_length': len(self.path_history),
                'resource_counts': defaultdict(int)
            }
            
            # Count resources
            for block_data in self.block_cache.values():
                stats['resource_counts'][block_data['type']] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting exploration stats: {e}")
            return {}
    
    def _persist_spatial_data(self) -> None:
        """Persist spatial data to storage."""
        try:
            data = {
                'chunks': {str(k): v for k, v in self.chunks.items()},
                'points_of_interest': self.points_of_interest,
                'explored_chunks': [str(chunk) for chunk in self.explored_chunks],
                'path_history': self.path_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.storage_path / 'world_data.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error persisting spatial data: {e}")
    
    def _load_spatial_data(self) -> None:
        """Load spatial data from storage."""
        try:
            data_file = self.storage_path / 'world_data.json'
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load chunks
                    self.chunks = {eval(k): v for k, v in data.get('chunks', {}).items()}
                    
                    # Load other data
                    self.points_of_interest = data.get('points_of_interest', {})
                    self.explored_chunks = {eval(chunk) for chunk in data.get('explored_chunks', [])}
                    self.path_history = data.get('path_history', [])
                    
                    # Rebuild block cache
                    self.block_cache.clear()
                    for chunk_coord, chunk_data in self.chunks.items():
                        for block_key, block_data in chunk_data.get('blocks', {}).items():
                            x, y, z = eval(block_key)
                            abs_x = chunk_coord.x * self.chunk_size + x
                            abs_y = chunk_coord.y * self.chunk_size + y
                            abs_z = chunk_coord.z * self.chunk_size + z
                            self.block_cache[(abs_x, abs_y, abs_z)] = block_data
                    
        except Exception as e:
            self.logger.error(f"Error loading spatial data: {e}")
    
    def clear(self) -> None:
        """Clear all spatial data."""
        self.chunks.clear()
        self.points_of_interest.clear()
        self.explored_chunks.clear()
        self.path_history.clear()
        self.block_cache.clear()
        self._persist_spatial_data()
