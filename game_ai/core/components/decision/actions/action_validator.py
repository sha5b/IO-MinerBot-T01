"""Action validation component."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class ActionValidator:
    """Validates and constrains game actions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize action validator.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Action constraints
        self.constraints = {
            'duration': {
                'min': config.get('action', {}).get('min_duration', 0.1),
                'max': config.get('action', {}).get('max_duration', 5.0)
            },
            'movement': {
                'speed_limit': 100,  # Base movement speed
                'terrain_modifiers': {
                    'normal': 1.0,
                    'difficult': 0.7,
                    'easy': 1.3
                }
            },
            'combat': {
                'min_health': 20,  # Minimum health for combat actions
                'cooldown': 0.5    # Time between combat actions
            }
        }
        
        # Valid action types and keys
        self.valid_types = {
            'movement': ['forward', 'backward', 'left', 'right', 'jump', 'crouch'],
            'look': ['look_left', 'look_right', 'look_up', 'look_down'],
            'action': ['mouse1', 'mouse2', 'mouse3'],
            'hotbar': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'menu': ['inventory', 'crafting', 'escape']
        }
        
        # Action cooldowns
        self.cooldowns: Dict[str, float] = {}
    
    def validate_action(self, action: Dict[str, Any], 
                       game_state: Dict[str, Any]) -> bool:
        """
        Validate if an action can be executed.
        
        Args:
            action (dict): Action to validate
            game_state (dict): Current game state
            
        Returns:
            bool: True if action is valid
        """
        try:
            # Check basic format
            if not self._validate_format(action):
                return False
            
            # Check action type and key
            if not self._validate_type_and_key(action):
                return False
            
            # Check duration constraints
            if not self._validate_duration(action):
                return False
            
            # Check cooldowns
            if not self._check_cooldown(action):
                return False
            
            # Check state-based constraints
            if not self._validate_state_constraints(action, game_state):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating action: {e}")
            return False
    
    def validate_sequence(self, sequence: List[Dict[str, Any]], 
                         game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate and filter action sequence.
        
        Args:
            sequence (list): Action sequence to validate
            game_state (dict): Current game state
            
        Returns:
            list: Valid actions from sequence
        """
        try:
            valid_actions = []
            
            for action in sequence:
                if self.validate_action(action, game_state):
                    valid_actions.append(action)
                    self._update_cooldown(action)
            
            return valid_actions
            
        except Exception as e:
            self.logger.error(f"Error validating sequence: {e}")
            return []
    
    def _validate_format(self, action: Dict[str, Any]) -> bool:
        """Validate action format."""
        try:
            # Check required fields
            if not all(k in action for k in ['type', 'key', 'duration']):
                self.logger.warning("Missing required fields")
                return False
            
            # Check field types
            if not isinstance(action['type'], str):
                self.logger.warning("Invalid type field")
                return False
            if not isinstance(action['key'], str):
                self.logger.warning("Invalid key field")
                return False
            if not isinstance(action['duration'], (int, float)):
                self.logger.warning("Invalid duration field")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating format: {e}")
            return False
    
    def _validate_type_and_key(self, action: Dict[str, Any]) -> bool:
        """Validate action type and key."""
        try:
            action_type = action['type']
            action_key = action['key']
            
            # Check if type is valid
            if action_type not in self.valid_types:
                self.logger.warning(f"Invalid action type: {action_type}")
                return False
            
            # Check if key is valid for type
            if action_key not in self.valid_types[action_type]:
                self.logger.warning(f"Invalid key for type {action_type}: {action_key}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating type/key: {e}")
            return False
    
    def _validate_duration(self, action: Dict[str, Any]) -> bool:
        """Validate action duration."""
        try:
            duration = action['duration']
            min_duration = self.constraints['duration']['min']
            max_duration = self.constraints['duration']['max']
            
            if not min_duration <= duration <= max_duration:
                self.logger.warning(f"Duration {duration} outside valid range")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating duration: {e}")
            return False
    
    def _check_cooldown(self, action: Dict[str, Any]) -> bool:
        """Check if action is on cooldown."""
        try:
            action_id = f"{action['type']}_{action['key']}"
            
            if action_id in self.cooldowns:
                time_since = datetime.now().timestamp() - self.cooldowns[action_id]
                cooldown = self.constraints['combat']['cooldown'] if action['type'] == 'action' else 0
                
                if time_since < cooldown:
                    self.logger.debug(f"Action {action_id} on cooldown")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking cooldown: {e}")
            return False
    
    def _validate_state_constraints(self, action: Dict[str, Any],
                                  game_state: Dict[str, Any]) -> bool:
        """Validate state-based constraints."""
        try:
            # Check health for combat actions
            if action['type'] == 'action':
                health = game_state.get('player', {}).get('health', 100)
                if health < self.constraints['combat']['min_health']:
                    self.logger.warning("Health too low for combat action")
                    return False
            
            # Check movement constraints
            if action['type'] == 'movement':
                terrain = game_state.get('environment', {}).get('terrain_type', 'normal')
                modifier = self.constraints['movement']['terrain_modifiers'].get(terrain, 1.0)
                
                if modifier <= 0:
                    self.logger.warning("Movement blocked by terrain")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating state constraints: {e}")
            return False
    
    def _update_cooldown(self, action: Dict[str, Any]) -> None:
        """Update action cooldown."""
        try:
            action_id = f"{action['type']}_{action['key']}"
            self.cooldowns[action_id] = datetime.now().timestamp()
        except Exception as e:
            self.logger.error(f"Error updating cooldown: {e}")
    
    def update_constraints(self, new_constraints: Dict[str, Any]) -> None:
        """
        Update action constraints.
        
        Args:
            new_constraints (dict): New constraint values
        """
        try:
            for category, values in new_constraints.items():
                if category in self.constraints:
                    self.constraints[category].update(values)
            
            self.logger.info("Updated action constraints")
            
        except Exception as e:
            self.logger.error(f"Error updating constraints: {e}")
    
    def add_valid_action(self, action_type: str, action_key: str) -> None:
        """
        Add new valid action type/key.
        
        Args:
            action_type (str): Action type
            action_key (str): Action key
        """
        try:
            if action_type not in self.valid_types:
                self.valid_types[action_type] = []
            
            if action_key not in self.valid_types[action_type]:
                self.valid_types[action_type].append(action_key)
                self.logger.info(f"Added valid action: {action_type}/{action_key}")
                
        except Exception as e:
            self.logger.error(f"Error adding valid action: {e}")
