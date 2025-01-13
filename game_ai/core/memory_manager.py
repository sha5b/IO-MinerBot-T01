"""Memory management system for storing and retrieving game state information."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import logging
from datetime import datetime, timedelta

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
            for dir_name in ['current_state', 'short_term', 'long_term']:
                path = self.memory_path / dir_name
                path.mkdir(parents=True, exist_ok=True)
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
            
            # Persist to disk
            timestamp = datetime.now().isoformat()
            state_with_meta = {
                'timestamp': timestamp,
                'state': state
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
                
            self.logger.info("Memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
