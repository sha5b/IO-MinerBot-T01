"""Long-term memory component for persistent storage of game states and events."""

from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import shutil

class LongTermMemory:
    """Handles persistent storage and retrieval of game memories."""
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        """
        Initialize long-term memory.
        
        Args:
            config (dict): Configuration settings
            storage_path (Path): Path for memory storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.storage_path = storage_path / 'long_term'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory settings
        self.max_files = config.get('long_term_limit', 1000)
        self.compression_age = config.get('compression_age_days', 30)
        self.cleanup_threshold = config.get('cleanup_threshold_mb', 1000)
        
        # Memory indexing
        self.memory_index: Dict[str, Dict[str, List[str]]] = {}  # date -> type -> files
        self.event_index: Dict[str, Dict[str, List[str]]] = {}   # date -> type -> files
        
        # Build indices
        self._build_indices()
    
    def store_memory(self, memory: Dict[str, Any]) -> None:
        """
        Store a memory in long-term storage.
        
        Args:
            memory (dict): Memory to store
        """
        try:
            # Get date from memory timestamp
            timestamp = datetime.fromisoformat(memory['timestamp'])
            date_key = timestamp.date().isoformat()
            
            # Create date directory if needed
            date_dir = self.storage_path / date_key
            date_dir.mkdir(exist_ok=True)
            
            # Create memory file
            file_name = f"memory_{timestamp.strftime('%H%M%S')}_{memory['type']}.json"
            memory_file = date_dir / file_name
            
            # Store memory
            with open(memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
            
            # Update indices
            self._update_indices(date_key, memory['type'], file_name)
            
            # Check storage limits
            self._check_storage_limits()
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
    
    def store_batch(self, memories: List[Dict[str, Any]]) -> None:
        """
        Store multiple memories efficiently.
        
        Args:
            memories (list): List of memories to store
        """
        try:
            # Group memories by date
            date_groups: Dict[str, List[Dict[str, Any]]] = {}
            for memory in memories:
                timestamp = datetime.fromisoformat(memory['timestamp'])
                date_key = timestamp.date().isoformat()
                if date_key not in date_groups:
                    date_groups[date_key] = []
                date_groups[date_key].append(memory)
            
            # Store each date group
            for date_key, group in date_groups.items():
                date_dir = self.storage_path / date_key
                date_dir.mkdir(exist_ok=True)
                
                # Create batch file
                batch_file = date_dir / f"batch_{datetime.now().strftime('%H%M%S')}.json"
                with open(batch_file, 'w') as f:
                    json.dump(group, f, indent=2)
                
                # Update indices
                for memory in group:
                    self._update_indices(date_key, memory['type'], batch_file.name)
            
            # Check storage limits
            self._check_storage_limits()
            
        except Exception as e:
            self.logger.error(f"Error storing memory batch: {e}")
    
    def retrieve_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query parameters.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Matching memories
        """
        try:
            matches = []
            
            # Get date range from query
            date_range = self._get_date_range(query)
            
            # Search each date in range
            for search_date in date_range:
                date_key = search_date.isoformat()
                
                # Get relevant files from indices
                memory_files = self._get_indexed_files(date_key, query)
                
                # Search files
                for file_path in memory_files:
                    matches.extend(self._search_file(file_path, query))
            
            # Sort by timestamp and apply limit
            matches.sort(key=lambda x: x['timestamp'])
            return matches[:query.get('limit', len(matches))]
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            dict: Memory statistics
        """
        try:
            total_size = 0
            file_count = 0
            memory_types = set()
            event_types = set()
            
            # Calculate statistics
            for date_dir in self.storage_path.iterdir():
                if date_dir.is_dir():
                    for file in date_dir.iterdir():
                        if file.is_file():
                            total_size += file.stat().st_size
                            file_count += 1
                            
                            # Get types from indices
                            date_key = date_dir.name
                            if date_key in self.memory_index:
                                memory_types.update(self.memory_index[date_key].keys())
                            if date_key in self.event_index:
                                event_types.update(self.event_index[date_key].keys())
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'memory_types': list(memory_types),
                'event_types': list(event_types),
                'date_range': self._get_storage_date_range()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def _build_indices(self) -> None:
        """Build memory indices from storage."""
        try:
            self.memory_index.clear()
            self.event_index.clear()
            
            for date_dir in self.storage_path.iterdir():
                if date_dir.is_dir():
                    date_key = date_dir.name
                    
                    # Initialize date indices
                    self.memory_index[date_key] = {}
                    self.event_index[date_key] = {}
                    
                    # Process each file
                    for file in date_dir.iterdir():
                        if file.is_file() and file.suffix == '.json':
                            try:
                                with open(file, 'r') as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        # Batch file
                                        for item in data:
                                            self._update_indices(
                                                date_key,
                                                item['type'],
                                                file.name
                                            )
                                    else:
                                        # Single memory file
                                        self._update_indices(
                                            date_key,
                                            data['type'],
                                            file.name
                                        )
                            except Exception as e:
                                self.logger.error(f"Error processing file {file}: {e}")
                                
        except Exception as e:
            self.logger.error(f"Error building indices: {e}")
    
    def _update_indices(self, date_key: str, memory_type: str, file_name: str) -> None:
        """Update memory indices with new file."""
        # Update appropriate index based on type
        if memory_type == 'event':
            if date_key not in self.event_index:
                self.event_index[date_key] = {}
            if memory_type not in self.event_index[date_key]:
                self.event_index[date_key][memory_type] = []
            if file_name not in self.event_index[date_key][memory_type]:
                self.event_index[date_key][memory_type].append(file_name)
        else:
            if date_key not in self.memory_index:
                self.memory_index[date_key] = {}
            if memory_type not in self.memory_index[date_key]:
                self.memory_index[date_key][memory_type] = []
            if file_name not in self.memory_index[date_key][memory_type]:
                self.memory_index[date_key][memory_type].append(file_name)
    
    def _get_date_range(self, query: Dict[str, Any]) -> List[date]:
        """Get list of dates to search based on query."""
        if 'time_range' in query:
            start_date = query['time_range'][0].date()
            end_date = query['time_range'][1].date()
            return [
                start_date + timedelta(days=x)
                for x in range((end_date - start_date).days + 1)
            ]
        else:
            # Default to recent dates
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            return [
                start_date + timedelta(days=x)
                for x in range(8)
            ]
    
    def _get_indexed_files(self, date_key: str, query: Dict[str, Any]) -> List[Path]:
        """Get relevant files for date based on query and indices."""
        files = set()
        
        # Get memory files
        if date_key in self.memory_index:
            for memory_type, file_names in self.memory_index[date_key].items():
                if 'type' not in query or query['type'] == memory_type:
                    files.update(file_names)
        
        # Get event files
        if date_key in self.event_index:
            for event_type, file_names in self.event_index[date_key].items():
                if 'event_type' not in query or query['event_type'] == event_type:
                    files.update(file_names)
        
        return [self.storage_path / date_key / file_name for file_name in files]
    
    def _search_file(self, file_path: Path, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search a single file for matching memories."""
        matches = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Handle both single memories and batches
                memories = data if isinstance(data, list) else [data]
                
                for memory in memories:
                    if self._matches_query(memory, query):
                        matches.append(memory)
                        
        except Exception as e:
            self.logger.error(f"Error searching file {file_path}: {e}")
            
        return matches
    
    def _matches_query(self, memory: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if memory matches query parameters."""
        try:
            for key, value in query.items():
                if key in ['limit', 'time_range']:
                    continue
                    
                if key == 'type':
                    if memory.get('type') != value:
                        return False
                elif key == 'event_type':
                    if memory.get('event_type') != value:
                        return False
                elif key in memory.get('state', {}):
                    if memory['state'][key] != value:
                        return False
                elif key in memory.get('data', {}):
                    if memory['data'][key] != value:
                        return False
            return True
            
        except Exception:
            return False
    
    def _check_storage_limits(self) -> None:
        """Check and enforce storage limits."""
        try:
            # Check total size
            total_size = sum(
                f.stat().st_size
                for f in self.storage_path.rglob('*.json')
            ) / (1024 * 1024)  # Convert to MB
            
            if total_size > self.cleanup_threshold:
                self._cleanup_old_data()
            
            # Check file count
            date_dirs = sorted(
                [d for d in self.storage_path.iterdir() if d.is_dir()],
                key=lambda x: x.name
            )
            
            while len(date_dirs) > self.max_files:
                # Remove oldest directory
                shutil.rmtree(date_dirs[0])
                date_dirs.pop(0)
                
                # Update indices
                date_key = date_dirs[0].name
                self.memory_index.pop(date_key, None)
                self.event_index.pop(date_key, None)
                
        except Exception as e:
            self.logger.error(f"Error checking storage limits: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data to maintain storage limits."""
        try:
            # Get list of date directories older than compression age
            cutoff_date = datetime.now().date() - timedelta(days=self.compression_age)
            old_dirs = [
                d for d in self.storage_path.iterdir()
                if d.is_dir() and d.name <= cutoff_date.isoformat()
            ]
            
            for date_dir in old_dirs:
                # Compress directory contents
                self._compress_directory(date_dir)
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def _compress_directory(self, directory: Path) -> None:
        """Compress directory contents into a single file."""
        try:
            # Read all memories from directory
            memories = []
            for file in directory.glob('*.json'):
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        memories.extend(data)
                    else:
                        memories.append(data)
            
            if memories:
                # Create compressed file
                compressed_file = directory / 'compressed.json'
                with open(compressed_file, 'w') as f:
                    json.dump(memories, f)
                
                # Remove original files
                for file in directory.glob('*.json'):
                    if file != compressed_file:
                        file.unlink()
                        
        except Exception as e:
            self.logger.error(f"Error compressing directory {directory}: {e}")
    
    def _get_storage_date_range(self) -> Dict[str, str]:
        """Get date range of stored memories."""
        try:
            date_dirs = [d for d in self.storage_path.iterdir() if d.is_dir()]
            if date_dirs:
                dates = sorted(d.name for d in date_dirs)
                return {
                    'start_date': dates[0],
                    'end_date': dates[-1]
                }
            return {
                'start_date': None,
                'end_date': None
            }
        except Exception as e:
            self.logger.error(f"Error getting storage date range: {e}")
            return {
                'start_date': None,
                'end_date': None
            }
