"""State tracking component for managing current game state."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import json

class StateTracker:
    """Handles tracking and management of current game state."""
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        """
        Initialize state tracker.
        
        Args:
            config (dict): Configuration settings
            storage_path (Path): Path for state storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.storage_path = storage_path / 'current_state'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current state tracking
        self.current_state: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.last_update = datetime.now()
        
        # State change callbacks
        self.change_callbacks: List[callable] = []
        
        # Load last state if available
        self._load_last_state()
    
    def update_state(self, new_state: Dict[str, Any], persist: bool = True) -> None:
        """
        Update current game state.
        
        Args:
            new_state (dict): New state information
            persist (bool): Whether to persist state to disk
        """
        try:
            # Detect changes
            changes = self._detect_changes(new_state)
            
            # Update state
            timestamp = datetime.now()
            state_with_meta = {
                'timestamp': timestamp.isoformat(),
                'state': new_state,
                'changes': changes
            }
            
            # Add to history
            history_limit = self.config.get('state_history_limit', 100)
            self.state_history.append(state_with_meta)
            if len(self.state_history) > history_limit:
                self.state_history.pop(0)
            
            # Update current state
            self.current_state = new_state
            self.last_update = timestamp
            
            # Notify callbacks of changes
            self._notify_changes(changes)
            
            # Persist if requested
            if persist:
                self._persist_state(state_with_meta)
                
        except Exception as e:
            self.logger.error(f"Error updating state: {e}")
            raise
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current game state.
        
        Returns:
            dict: Current state
        """
        return self.current_state.copy()
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent state history.
        
        Args:
            limit (int, optional): Maximum number of states to return
            
        Returns:
            list: Recent state history
        """
        if limit is None:
            return self.state_history.copy()
        return self.state_history[-limit:]
    
    def register_change_callback(self, callback: callable) -> None:
        """
        Register callback for state changes.
        
        Args:
            callback (callable): Function to call on state changes
        """
        if callback not in self.change_callbacks:
            self.change_callbacks.append(callback)
    
    def unregister_change_callback(self, callback: callable) -> None:
        """
        Unregister change callback.
        
        Args:
            callback (callable): Callback to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def _detect_changes(self, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect changes between current and new state.
        
        Args:
            new_state (dict): New state to compare
            
        Returns:
            dict: Detected changes
        """
        if not self.current_state:
            return {'type': 'initial', 'changes': new_state}
            
        changes = {
            'type': 'update',
            'added': {},
            'modified': {},
            'removed': {}
        }
        
        # Find added and modified fields
        for key, value in new_state.items():
            if key not in self.current_state:
                changes['added'][key] = value
            elif self.current_state[key] != value:
                changes['modified'][key] = {
                    'old': self.current_state[key],
                    'new': value
                }
        
        # Find removed fields
        for key in self.current_state:
            if key not in new_state:
                changes['removed'][key] = self.current_state[key]
        
        return changes
    
    def _notify_changes(self, changes: Dict[str, Any]) -> None:
        """
        Notify registered callbacks of state changes.
        
        Args:
            changes (dict): Detected changes
        """
        for callback in self.change_callbacks:
            try:
                callback(changes)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")
    
    def _persist_state(self, state_with_meta: Dict[str, Any]) -> None:
        """
        Persist state to storage.
        
        Args:
            state_with_meta (dict): State with metadata to persist
        """
        try:
            # Save latest state
            latest_file = self.storage_path / 'latest.json'
            with open(latest_file, 'w') as f:
                json.dump(state_with_meta, f, indent=2)
                
            # Save timestamped state if configured
            if self.config.get('save_state_history', False):
                timestamp = datetime.fromisoformat(state_with_meta['timestamp'])
                dated_file = self.storage_path / f"state_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                with open(dated_file, 'w') as f:
                    json.dump(state_with_meta, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error persisting state: {e}")
    
    def _load_last_state(self) -> None:
        """Load last persisted state if available."""
        try:
            latest_file = self.storage_path / 'latest.json'
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    state_with_meta = json.load(f)
                    self.current_state = state_with_meta['state']
                    self.last_update = datetime.fromisoformat(state_with_meta['timestamp'])
                    self.state_history.append(state_with_meta)
                    
        except Exception as e:
            self.logger.error(f"Error loading last state: {e}")
    
    def get_state_age(self) -> float:
        """
        Get age of current state in seconds.
        
        Returns:
            float: Seconds since last update
        """
        return (datetime.now() - self.last_update).total_seconds()
    
    def clear_history(self) -> None:
        """Clear state history while maintaining current state."""
        self.state_history = []
        if self.current_state:
            self.state_history.append({
                'timestamp': self.last_update.isoformat(),
                'state': self.current_state,
                'changes': {'type': 'clear_history'}
            })
    
    def reset(self) -> None:
        """Reset state tracker to initial state."""
        self.current_state = {}
        self.state_history = []
        self.last_update = datetime.now()
        self._notify_changes({'type': 'reset'})
