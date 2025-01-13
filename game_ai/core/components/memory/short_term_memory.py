"""Short-term memory component for managing recent game states and events."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import json
from collections import deque

class ShortTermMemory:
    """Handles storage and retrieval of recent game states and events."""
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        """
        Initialize short-term memory.
        
        Args:
            config (dict): Configuration settings
            storage_path (Path): Path for memory storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.storage_path = storage_path / 'short_term'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory settings
        self.capacity = config.get('short_term_limit', 100)
        self.memory_buffer = deque(maxlen=self.capacity)
        self.event_buffer = deque(maxlen=self.capacity)
        
        # Memory indexing
        self.memory_index: Dict[str, List[int]] = {}  # Type -> list of indices
        self.event_index: Dict[str, List[int]] = {}   # Event type -> list of indices
        
        # Memory overflow callbacks
        self.overflow_callbacks: List[callable] = []
        
        # Load existing memories if available
        self._load_memories()
    
    def store_state(self, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a game state in short-term memory.
        
        Args:
            state (dict): Game state to store
            metadata (dict, optional): Additional metadata about the state
        """
        try:
            # Create memory entry
            memory = {
                'type': 'state',
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'metadata': metadata or {}
            }
            
            # Add to buffer
            self.memory_buffer.append(memory)
            
            # Update indices
            self._update_memory_indices(memory, len(self.memory_buffer) - 1)
            
            # Check for overflow
            if len(self.memory_buffer) == self.capacity:
                self._handle_overflow('state', memory)
            
            # Persist memory
            self._persist_memories()
            
        except Exception as e:
            self.logger.error(f"Error storing state: {e}")
    
    def store_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Store an event in short-term memory.
        
        Args:
            event_type (str): Type of event
            event_data (dict): Event data
        """
        try:
            # Create event entry
            event = {
                'type': 'event',
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': event_data
            }
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Update indices
            self._update_event_indices(event, len(self.event_buffer) - 1)
            
            # Check for overflow
            if len(self.event_buffer) == self.capacity:
                self._handle_overflow('event', event)
            
            # Persist events
            self._persist_memories()
            
        except Exception as e:
            self.logger.error(f"Error storing event: {e}")
    
    def get_recent_states(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get most recent states.
        
        Args:
            limit (int, optional): Maximum number of states to return
            
        Returns:
            list: Recent states
        """
        if limit is None:
            return list(self.memory_buffer)
        return list(self.memory_buffer)[-limit:]
    
    def get_recent_events(self, event_type: Optional[str] = None, 
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get most recent events.
        
        Args:
            event_type (str, optional): Type of events to retrieve
            limit (int, optional): Maximum number of events to return
            
        Returns:
            list: Recent events
        """
        if event_type:
            indices = self.event_index.get(event_type, [])
            events = [self.event_buffer[i] for i in indices if i < len(self.event_buffer)]
        else:
            events = list(self.event_buffer)
            
        if limit:
            return events[-limit:]
        return events
    
    def search_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search memories based on query parameters.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Matching memories
        """
        matches = []
        
        try:
            # Search states
            for memory in self.memory_buffer:
                if self._matches_query(memory, query):
                    matches.append(memory)
            
            # Search events
            for event in self.event_buffer:
                if self._matches_query(event, query):
                    matches.append(event)
            
            # Sort by timestamp
            matches.sort(key=lambda x: x['timestamp'])
            
            return matches[:query.get('limit', len(matches))]
            
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []
    
    def register_overflow_callback(self, callback: callable) -> None:
        """
        Register callback for memory overflow.
        
        Args:
            callback (callable): Function to call when memory overflows
        """
        if callback not in self.overflow_callbacks:
            self.overflow_callbacks.append(callback)
    
    def _update_memory_indices(self, memory: Dict[str, Any], index: int) -> None:
        """Update memory type indices."""
        memory_type = memory.get('metadata', {}).get('type', 'default')
        if memory_type not in self.memory_index:
            self.memory_index[memory_type] = []
        self.memory_index[memory_type].append(index)
    
    def _update_event_indices(self, event: Dict[str, Any], index: int) -> None:
        """Update event type indices."""
        event_type = event['event_type']
        if event_type not in self.event_index:
            self.event_index[event_type] = []
        self.event_index[event_type].append(index)
    
    def _handle_overflow(self, buffer_type: str, item: Dict[str, Any]) -> None:
        """Handle buffer overflow by notifying callbacks."""
        for callback in self.overflow_callbacks:
            try:
                callback(buffer_type, item)
            except Exception as e:
                self.logger.error(f"Error in overflow callback: {e}")
    
    def _matches_query(self, item: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if memory/event matches query parameters."""
        try:
            for key, value in query.items():
                if key == 'limit':
                    continue
                    
                if key == 'time_range':
                    timestamp = datetime.fromisoformat(item['timestamp'])
                    if not (value[0] <= timestamp <= value[1]):
                        return False
                elif key == 'type':
                    if item.get('type') != value:
                        return False
                elif key == 'event_type':
                    if item.get('event_type') != value:
                        return False
                elif key in item.get('state', {}):
                    if item['state'][key] != value:
                        return False
                elif key in item.get('data', {}):
                    if item['data'][key] != value:
                        return False
            return True
            
        except Exception:
            return False
    
    def _persist_memories(self) -> None:
        """Persist current memories to storage."""
        try:
            memory_file = self.storage_path / 'memory.json'
            with open(memory_file, 'w') as f:
                json.dump({
                    'states': list(self.memory_buffer),
                    'events': list(self.event_buffer),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error persisting memories: {e}")
    
    def _load_memories(self) -> None:
        """Load persisted memories if available."""
        try:
            memory_file = self.storage_path / 'memory.json'
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load states
                    for state in data.get('states', []):
                        self.memory_buffer.append(state)
                        self._update_memory_indices(
                            state, 
                            len(self.memory_buffer) - 1
                        )
                    
                    # Load events
                    for event in data.get('events', []):
                        self.event_buffer.append(event)
                        self._update_event_indices(
                            event,
                            len(self.event_buffer) - 1
                        )
                    
        except Exception as e:
            self.logger.error(f"Error loading memories: {e}")
    
    def clear(self) -> None:
        """Clear all short-term memories."""
        self.memory_buffer.clear()
        self.event_buffer.clear()
        self.memory_index.clear()
        self.event_index.clear()
        self._persist_memories()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            dict: Memory statistics
        """
        return {
            'state_count': len(self.memory_buffer),
            'event_count': len(self.event_buffer),
            'memory_types': {k: len(v) for k, v in self.memory_index.items()},
            'event_types': {k: len(v) for k, v in self.event_index.items()},
            'capacity': self.capacity
        }
