"""Memory management system for storing and retrieving game state information."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import time
import logging
from datetime import datetime, timedelta
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
    """Manages spatial information about the game world."""
    
    def __init__(self, chunk_size: int = 16):
        """
        Initialize spatial memory system.
        
        Args:
            chunk_size (int): Size of each chunk (default: 16 for Minecraft-like games)
        """
        self.chunk_size = chunk_size
        self.chunks: Dict[ChunkCoord, Dict[str, Any]] = {}
        self.points_of_interest: Dict[str, Dict[str, Any]] = {}
        self.explored_chunks: Set[ChunkCoord] = set()
        self.path_history: List[Tuple[float, float, float]] = []
        self.block_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    
    def get_chunk_coord(self, x: float, y: float, z: float) -> ChunkCoord:
        """Convert world coordinates to chunk coordinates."""
        return ChunkCoord(
            int(x // self.chunk_size),
            int(y // self.chunk_size),
            int(z // self.chunk_size)
        )
    
    def update_block(self, x: int, y: int, z: int, block_data: Dict[str, Any]) -> None:
        """
        Update information about a block at given coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            z (int): Z coordinate
            block_data (dict): Information about the block
        """
        chunk_coord = self.get_chunk_coord(x, y, z)
        if chunk_coord not in self.chunks:
            self.chunks[chunk_coord] = {'blocks': {}}
        
        block_key = (x % self.chunk_size, y % self.chunk_size, z % self.chunk_size)
        self.chunks[chunk_coord]['blocks'][str(block_key)] = block_data
        self.block_cache[(x, y, z)] = block_data
    
    def add_point_of_interest(self, name: str, x: float, y: float, z: float, poi_type: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a point of interest to memory.
        
        Args:
            name (str): Name of the point of interest
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            poi_type (str): Type of point of interest (e.g., 'home', 'resource', 'danger')
            metadata (dict, optional): Additional information about the POI
        """
        self.points_of_interest[name] = {
            'position': (x, y, z),
            'type': poi_type,
            'metadata': metadata or {},
            'last_visited': datetime.now().isoformat()
        }
    
    def update_position(self, x: float, y: float, z: float) -> None:
        """
        Update the AI's position and track explored areas.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
        """
        self.path_history.append((x, y, z))
        chunk_coord = self.get_chunk_coord(x, y, z)
        self.explored_chunks.add(chunk_coord)
    
    def get_nearby_blocks(self, x: float, y: float, z: float, radius: int) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
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
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    pos = (int(x + dx), int(y + dy), int(z + dz))
                    if pos in self.block_cache:
                        nearby_blocks[pos] = self.block_cache[pos]
        return nearby_blocks
    
    def get_nearest_poi(self, x: float, y: float, z: float, poi_type: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
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
    
    def get_resource_distribution(self, resource_type: str, radius: int = 32) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Get distribution of resources in the explored area.
        
        Args:
            resource_type (str): Type of resource to look for (e.g., 'tree', 'stone')
            radius (int): Search radius from current position
            
        Returns:
            dict: Resource locations grouped by type
        """
        resources = defaultdict(list)
        
        for pos, block_data in self.block_cache.items():
            if block_data['type'] == resource_type:
                resources[resource_type].append(pos)
        
        return dict(resources)
    
    def find_nearest_resource(self, x: float, y: float, z: float, resource_type: str) -> Optional[Tuple[int, int, int]]:
        """
        Find the nearest resource block of a specific type.
        
        Args:
            x (float): Current X coordinate
            y (float): Current Y coordinate
            z (float): Current Z coordinate
            resource_type (str): Type of resource to find
            
        Returns:
            tuple: Coordinates of nearest resource, or None if not found
        """
        nearest_pos = None
        min_dist = float('inf')
        
        for pos, block_data in self.block_cache.items():
            if block_data['type'] == resource_type:
                bx, by, bz = pos
                dist = ((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) ** 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = pos
        
        return nearest_pos
    
    def get_unexplored_directions(self, x: float, y: float, z: float) -> List[Tuple[float, float, float]]:
        """
        Get vectors pointing towards unexplored areas.
        
        Args:
            x (float): Current X coordinate
            y (float): Current Y coordinate
            z (float): Current Z coordinate
            
        Returns:
            list: List of normalized direction vectors towards unexplored chunks
        """
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
    
    def get_path_to(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Get a path between two points using known safe blocks.
        
        Args:
            start (tuple): Starting coordinates (x, y, z)
            end (tuple): Ending coordinates (x, y, z)
            
        Returns:
            list: List of waypoints forming a path, or empty if no path found
        """
        # For now, return a simple direct path
        # This can be expanded to use A* pathfinding with block data
        return [start, end]
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about world exploration.
        
        Returns:
            dict: Statistics about explored areas and resources
        """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spatial memory to serializable dictionary."""
        return {
            'chunks': {str(k): v for k, v in self.chunks.items()},
            'points_of_interest': self.points_of_interest,
            'explored_chunks': [str(chunk) for chunk in self.explored_chunks],
            'path_history': self.path_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpatialMemory':
        """Create SpatialMemory instance from dictionary."""
        spatial_memory = cls()
        spatial_memory.chunks = {eval(k): v for k, v in data.get('chunks', {}).items()}
        spatial_memory.points_of_interest = data.get('points_of_interest', {})
        spatial_memory.explored_chunks = {eval(chunk) for chunk in data.get('explored_chunks', [])}
        spatial_memory.path_history = data.get('path_history', [])
        return spatial_memory

class MemoryManager:
    """Manages all memory operations including state tracking and persistence."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the memory management system.
        
        Args:
            config_path (Path, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.current_state: Dict[str, Any] = {}
        self.short_term: List[Dict[str, Any]] = []
        self.long_term: Dict[str, List[Dict[str, Any]]] = {}
        self.last_cleanup = time.time()
        
        # Initialize spatial memory
        self.spatial_memory = SpatialMemory()
        
        # Initialize memory storage paths
        self.memory_path = Path("memory")
        self._initialize_storage()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """
        Load memory system configuration.
        
        Args:
            config_path (Path): Path to config file
            
        Returns:
            dict: Configuration settings
        """
        default_config = {
            'short_term_limit': 100,
            'long_term_limit': 1000,
            'cleanup_interval': 3600  # 1 hour
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f).get('memory', {})
                    return {**default_config, **loaded_config}
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                return default_config
        return default_config
    
    def _initialize_storage(self) -> None:
        """Initialize memory storage directories."""
        try:
            # Create memory directory structure
            for dir_name in ['current_state', 'short_term', 'long_term', 'spatial']:
                path = self.memory_path / dir_name
                path.mkdir(parents=True, exist_ok=True)
            
            # Load existing spatial memory if available
            spatial_file = self.memory_path / 'spatial' / 'world_data.json'
            if spatial_file.exists():
                try:
                    with open(spatial_file, 'r') as f:
                        spatial_data = json.load(f)
                        self.spatial_memory = SpatialMemory.from_dict(spatial_data)
                except Exception as e:
                    self.logger.error(f"Error loading spatial memory: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing storage: {e}")
            raise
    
    def update_current_state(self, state: Dict[str, Any]) -> None:
        """
        Update the current game state.
        
        Args:
            state (dict): New game state information
        """
        try:
            # Update in-memory state
            self.current_state = state
            
            # Update spatial memory if position is available
            if 'position' in state:
                x, y, z = state['position']
                self.spatial_memory.update_position(x, y, z)
            
            # Update block information if available
            if 'visible_blocks' in state:
                for block in state['visible_blocks']:
                    self.spatial_memory.update_block(
                        block['x'], block['y'], block['z'],
                        {
                            'type': block['type'],
                            'properties': block.get('properties', {}),
                            'last_seen': datetime.now().isoformat()
                        }
                    )
            
            # Persist to disk
            timestamp = datetime.now().isoformat()
            state_with_meta = {
                'timestamp': timestamp,
                'state': state,
                'spatial': self.spatial_memory.to_dict()
            }
            
            with open(self.memory_path / 'current_state' / 'latest.json', 'w') as f:
                json.dump(state_with_meta, f, indent=2)
                
            # Add to short-term memory
            self.store_short_term(state_with_meta)
            
        except Exception as e:
            self.logger.error(f"Error updating current state: {e}")
            raise
    
    def store_short_term(self, state: Dict[str, Any]) -> None:
        """
        Store state in short-term memory.
        
        Args:
            state (dict): State information to store
        """
        try:
            self.short_term.append(state)
            
            # Enforce short-term memory limit
            while len(self.short_term) > self.config['short_term_limit']:
                # Move oldest state to long-term memory before removing
                oldest = self.short_term.pop(0)
                self.store_long_term(oldest)
                
            # Persist short-term memory
            with open(self.memory_path / 'short_term' / 'memory.json', 'w') as f:
                json.dump(self.short_term, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error storing short-term memory: {e}")
            raise
    
    def store_long_term(self, state: Dict[str, Any]) -> None:
        """
        Store state in long-term memory.
        
        Args:
            state (dict): State information to store
        """
        try:
            # Organize by date for easier retrieval
            timestamp = datetime.fromisoformat(state['timestamp'])
            date_key = timestamp.date().isoformat()
            
            if date_key not in self.long_term:
                self.long_term[date_key] = []
            
            self.long_term[date_key].append(state)
            
            # Persist to disk
            long_term_file = self.memory_path / 'long_term' / f"{date_key}.json"
            with open(long_term_file, 'w') as f:
                json.dump(self.long_term[date_key], f, indent=2)
                
            # Check if cleanup is needed
            self._check_cleanup()
            
        except Exception as e:
            self.logger.error(f"Error storing long-term memory: {e}")
            raise
    
    def get_relevant_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query parameters.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Relevant memory entries
        """
        try:
            relevant_memories = []
            
            # Search short-term memory
            for memory in self.short_term:
                if self._matches_query(memory, query):
                    relevant_memories.append(memory)
            
            # Search long-term memory if needed
            if len(relevant_memories) < query.get('limit', 10):
                relevant_memories.extend(self._search_long_term(query))
            
            return relevant_memories[:query.get('limit', 10)]
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            return []
    
    def _matches_query(self, memory: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if memory matches search criteria."""
        try:
            for key, value in query.items():
                if key == 'limit':
                    continue
                    
                if key == 'time_range':
                    timestamp = datetime.fromisoformat(memory['timestamp'])
                    if not (value[0] <= timestamp <= value[1]):
                        return False
                elif key in memory['state']:
                    if memory['state'][key] != value:
                        return False
            return True
            
        except Exception:
            return False
    
    def _search_long_term(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search long-term memory storage."""
        relevant = []
        
        try:
            # If time range specified, only search relevant files
            if 'time_range' in query:
                start_date = query['time_range'][0].date()
                end_date = query['time_range'][1].date()
                date_range = [start_date + timedelta(days=x) 
                            for x in range((end_date - start_date).days + 1)]
            else:
                # Search recent files by default
                date_range = [datetime.now().date()]
            
            # Search each relevant file
            for date in date_range:
                file_path = self.memory_path / 'long_term' / f"{date.isoformat()}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        memories = json.load(f)
                        for memory in memories:
                            if self._matches_query(memory, query):
                                relevant.append(memory)
            
            return relevant
            
        except Exception as e:
            self.logger.error(f"Error searching long-term memory: {e}")
            return []
    
    def _check_cleanup(self) -> None:
        """Check if memory cleanup is needed and perform if necessary."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.config['cleanup_interval']:
            self.cleanup_old_data()
            self.last_cleanup = current_time
    
    def cleanup_old_data(self) -> None:
        """Remove old data to maintain memory limits."""
        try:
            # Clean up long-term memory
            long_term_files = sorted(list((self.memory_path / 'long_term').glob('*.json')))
            while len(long_term_files) > self.config['long_term_limit']:
                oldest_file = long_term_files.pop(0)
                oldest_file.unlink()
            
            # Save current spatial memory state
            spatial_file = self.memory_path / 'spatial' / 'world_data.json'
            with open(spatial_file, 'w') as f:
                json.dump(self.spatial_memory.to_dict(), f, indent=2)
            
            # Trim path history if too long
            max_path_history = self.config.get('max_path_history', 1000)
            if len(self.spatial_memory.path_history) > max_path_history:
                self.spatial_memory.path_history = self.spatial_memory.path_history[-max_path_history:]
            
            self.logger.info("Memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
