"""Action optimization component."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class ActionOptimizer:
    """Optimizes action sequences for efficiency."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize action optimizer.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Optimization settings
        self.settings = {
            'max_sequence_length': config.get('action', {}).get('max_sequence_length', 20),
            'min_duration': config.get('action', {}).get('min_duration', 0.1),
            'max_duration': config.get('action', {}).get('max_duration', 5.0),
            'merge_threshold': 0.1,  # Time difference threshold for merging
            'split_threshold': 2.0   # Duration threshold for splitting
        }
        
        # Action type compatibility for merging
        self.mergeable_types = {
            'movement': ['movement'],
            'look': ['look'],
            'action': []  # Combat actions shouldn't be merged
        }
    
    def optimize_sequence(self, sequence: List[Dict[str, Any]], 
                         game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize action sequence.
        
        Args:
            sequence (list): Action sequence to optimize
            game_state (dict): Current game state
            
        Returns:
            list: Optimized action sequence
        """
        try:
            # Remove redundant actions
            optimized = self._remove_redundant_actions(sequence)
            
            # Merge similar consecutive actions
            optimized = self._merge_similar_actions(optimized)
            
            # Split long actions
            optimized = self._split_long_actions(optimized)
            
            # Reorder for efficiency
            optimized = self._reorder_actions(optimized, game_state)
            
            # Limit sequence length
            if len(optimized) > self.settings['max_sequence_length']:
                optimized = optimized[:self.settings['max_sequence_length']]
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Error optimizing sequence: {e}")
            return sequence
    
    def _remove_redundant_actions(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove unnecessary repeated actions."""
        try:
            filtered = []
            last_action = None
            
            for action in sequence:
                if not last_action or not self._is_redundant(action, last_action):
                    filtered.append(action)
                    last_action = action
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error removing redundant actions: {e}")
            return sequence
    
    def _is_redundant(self, action1: Dict[str, Any], 
                     action2: Dict[str, Any]) -> bool:
        """Check if actions are redundant."""
        try:
            # Check if actions cancel each other
            if self._are_opposite_actions(action1, action2):
                return True
            
            # Check if actions are identical
            return (
                action1['type'] == action2['type'] and
                action1['key'] == action2['key'] and
                abs(action1['duration'] - action2['duration']) < self.settings['merge_threshold']
            )
            
        except Exception as e:
            self.logger.error(f"Error checking redundancy: {e}")
            return False
    
    def _are_opposite_actions(self, action1: Dict[str, Any], 
                            action2: Dict[str, Any]) -> bool:
        """Check if actions cancel each other out."""
        opposites = {
            'forward': 'backward',
            'backward': 'forward',
            'left': 'right',
            'right': 'left',
            'look_left': 'look_right',
            'look_right': 'look_left',
            'look_up': 'look_down',
            'look_down': 'look_up'
        }
        
        return (
            action1['type'] == action2['type'] and
            action1['key'] in opposites and
            action2['key'] == opposites[action1['key']] and
            abs(action1['duration'] - action2['duration']) < self.settings['merge_threshold']
        )
    
    def _merge_similar_actions(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar consecutive actions."""
        try:
            merged = []
            current_action = None
            
            for action in sequence:
                if not current_action:
                    current_action = action.copy()
                elif self._can_merge(current_action, action):
                    # Merge actions
                    current_action['duration'] += action['duration']
                else:
                    # Ensure duration is within limits
                    current_action['duration'] = min(
                        self.settings['max_duration'],
                        current_action['duration']
                    )
                    merged.append(current_action)
                    current_action = action.copy()
            
            if current_action:
                current_action['duration'] = min(
                    self.settings['max_duration'],
                    current_action['duration']
                )
                merged.append(current_action)
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging actions: {e}")
            return sequence
    
    def _can_merge(self, action1: Dict[str, Any], 
                  action2: Dict[str, Any]) -> bool:
        """Check if actions can be merged."""
        try:
            # Check if types are mergeable
            if action1['type'] not in self.mergeable_types:
                return False
            if action2['type'] not in self.mergeable_types[action1['type']]:
                return False
            
            # Check if keys match
            if action1['key'] != action2['key']:
                return False
            
            # Check if combined duration would be valid
            combined_duration = action1['duration'] + action2['duration']
            if combined_duration > self.settings['max_duration']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking merge compatibility: {e}")
            return False
    
    def _split_long_actions(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split actions that exceed duration threshold."""
        try:
            split_sequence = []
            
            for action in sequence:
                if action['duration'] > self.settings['split_threshold']:
                    # Calculate number of splits needed
                    num_splits = int(action['duration'] / self.settings['split_threshold']) + 1
                    split_duration = action['duration'] / num_splits
                    
                    # Create split actions
                    for _ in range(num_splits):
                        split_action = action.copy()
                        split_action['duration'] = split_duration
                        split_sequence.append(split_action)
                else:
                    split_sequence.append(action)
            
            return split_sequence
            
        except Exception as e:
            self.logger.error(f"Error splitting actions: {e}")
            return sequence
    
    def _reorder_actions(self, sequence: List[Dict[str, Any]], 
                        game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reorder actions for optimal execution."""
        try:
            # Group actions by type
            groups = {
                'look': [],
                'movement': [],
                'action': [],
                'other': []
            }
            
            for action in sequence:
                action_type = action['type']
                if action_type in groups:
                    groups[action_type].append(action)
                else:
                    groups['other'].append(action)
            
            # Combine in optimal order:
            # 1. Look actions (to orient correctly)
            # 2. Movement actions
            # 3. Action/interaction actions
            # 4. Other actions
            optimized = []
            optimized.extend(groups['look'])
            optimized.extend(groups['movement'])
            optimized.extend(groups['action'])
            optimized.extend(groups['other'])
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Error reordering actions: {e}")
            return sequence
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update optimization settings.
        
        Args:
            new_settings (dict): New setting values
        """
        try:
            self.settings.update(new_settings)
            self.logger.info("Updated optimization settings")
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")
    
    def add_mergeable_types(self, action_type: str, mergeable_with: List[str]) -> None:
        """
        Add new mergeable action types.
        
        Args:
            action_type (str): Action type
            mergeable_with (list): List of compatible types
        """
        try:
            if action_type not in self.mergeable_types:
                self.mergeable_types[action_type] = []
            
            for other_type in mergeable_with:
                if other_type not in self.mergeable_types[action_type]:
                    self.mergeable_types[action_type].append(other_type)
            
            self.logger.info(f"Added mergeable types for {action_type}")
            
        except Exception as e:
            self.logger.error(f"Error adding mergeable types: {e}")
