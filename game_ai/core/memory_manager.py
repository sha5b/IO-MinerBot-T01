"""Memory management system for storing and retrieving game state information."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .components.memory import (
    StateTracker,
    ShortTermMemory,
    LongTermMemory,
    SpatialMemory
)

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
        
        # Initialize memory storage path
        self.memory_path = Path("memory")
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.state_tracker = StateTracker(self.config, self.memory_path)
        self.short_term = ShortTermMemory(self.config, self.memory_path)
        self.long_term = LongTermMemory(self.config, self.memory_path)
        self.spatial = SpatialMemory(self.config, self.memory_path)
        
        # Register memory overflow handler
        self.short_term.register_overflow_callback(self._handle_memory_overflow)
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load memory system configuration."""
        default_config = {
            'short_term_limit': 100,
            'long_term_limit': 1000,
            'cleanup_interval': 3600,  # 1 hour
            'chunk_size': 16,
            'max_path_history': 1000,
            'compression_age_days': 30,
            'cleanup_threshold_mb': 1000,
            'save_state_history': True
        }
        
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f).get('memory', {})
                    return {**default_config, **loaded_config}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            
        return default_config
    
    def update_current_state(self, state: Dict[str, Any]) -> None:
        """
        Update the current game state.
        
        Args:
            state (dict): New game state information
        """
        try:
            # Update state tracker
            self.state_tracker.update_state(state)
            
            # Store in short-term memory
            self.short_term.store_state(state)
            
            # Update spatial memory if position available
            if 'position' in state:
                x, y, z = state['position']
                self.spatial.update_position(x, y, z)
            
            # Update block information if available
            if 'visible_blocks' in state:
                for block in state['visible_blocks']:
                    self.spatial.update_block(
                        block['x'], block['y'], block['z'],
                        {
                            'type': block['type'],
                            'properties': block.get('properties', {}),
                            'last_seen': datetime.now().isoformat()
                        }
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating current state: {e}")
            raise
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current game state.
        
        Returns:
            dict: Current state
        """
        return self.state_tracker.get_current_state()
    
    def get_recent_states(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent state history.
        
        Args:
            limit (int, optional): Maximum number of states to return
            
        Returns:
            list: Recent states
        """
        return self.short_term.get_recent_states(limit)
    
    def store_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Store an event in memory.
        
        Args:
            event_type (str): Type of event
            event_data (dict): Event data
        """
        self.short_term.store_event(event_type, event_data)
    
    def get_relevant_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query parameters.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Relevant memory entries
        """
        # Search short-term memory first
        memories = self.short_term.search_memories(query)
        
        # If we need more results, search long-term memory
        if len(memories) < query.get('limit', 10):
            long_term_memories = self.long_term.retrieve_memories(query)
            memories.extend(long_term_memories)
            
            # Sort by timestamp and apply limit
            memories.sort(key=lambda x: x['timestamp'])
            memories = memories[:query.get('limit', 10)]
        
        return memories
    
    def add_point_of_interest(self, name: str, x: float, y: float, z: float, 
                            poi_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a point of interest to spatial memory.
        
        Args:
            name (str): Name of the point of interest
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            poi_type (str): Type of point of interest
            metadata (dict, optional): Additional information about the POI
        """
        self.spatial.add_point_of_interest(name, x, y, z, poi_type, metadata)
    
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
        return self.spatial.get_nearest_poi(x, y, z, poi_type)
    
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
        return self.spatial.get_unexplored_directions(x, y, z)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            dict: Memory statistics
        """
        return {
            'short_term': self.short_term.get_memory_stats(),
            'long_term': self.long_term.get_memory_stats(),
            'spatial': self.spatial.get_exploration_stats()
        }
    
    def _handle_memory_overflow(self, buffer_type: str, item: Dict[str, Any]) -> None:
        """Handle memory overflow by moving to long-term storage."""
        try:
            if buffer_type == 'state':
                self.long_term.store_memory(item)
            elif buffer_type == 'event':
                self.long_term.store_memory(item)
        except Exception as e:
            self.logger.error(f"Error handling memory overflow: {e}")
    
    def cleanup_old_data(self) -> None:
        """Clean up old data to maintain memory limits."""
        try:
            # Clear short-term memory
            self.short_term.clear()
            
            # Compress and cleanup long-term memory
            self.long_term._cleanup_old_data()
            
            self.logger.info("Memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
