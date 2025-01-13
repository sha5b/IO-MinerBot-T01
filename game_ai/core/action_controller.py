"""Action controller for executing game actions through input simulation."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

class ActionController:
    """Handles execution of game actions through input simulation."""
    
    def __init__(self, config_path: Optional[Path] = None, game_type: str = "default"):
        """
        Initialize the action controller.
        
        Args:
            config_path (Path, optional): Path to configuration file
            game_type (str): Type of game to load controls for (e.g., "minecraft", "fps_template")
        """
        self.logger = logging.getLogger(__name__)
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.config = self._load_config(config_path)
        self.control_maps = self._load_control_maps()
        self.active_actions: Dict[str, Any] = {}
        self.last_action_time = time.time()
        self.game_type = game_type
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load action controller configuration."""
        default_config = {
            'input_delay': 0.05,
            'action_timeout': 5.0
        }
        
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f).get('controls', {})
                    return {**default_config, **loaded_config}
            return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return default_config
    
    def _load_control_maps(self) -> Dict[str, Any]:
        """Load control mapping configuration."""
        try:
            control_maps_path = Path("game_ai/config/control_maps.json")
            if control_maps_path.exists():
                with open(control_maps_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading control maps: {e}")
            return {}
    
    def set_game(self, game_type: str) -> bool:
        """
        Set the current game type for control mapping.
        
        Args:
            game_type (str): Game type to use ("minecraft", "fps_template", etc.)
            
        Returns:
            bool: True if game type was found and set
        """
        if game_type in self.control_maps.get('games', {}):
            self.game_type = game_type
            self.logger.info(f"Set game type to: {game_type}")
            return True
        elif game_type == "default":
            self.game_type = "default"
            self.logger.info("Using default control scheme")
            return True
        else:
            self.logger.warning(f"Unknown game type: {game_type}, falling back to default")
            self.game_type = "default"
            return False
    
    def get_game_controls(self) -> Dict[str, Any]:
        """Get current game's control scheme."""
        return self.control_maps.get('games', {}).get(self.game_type, 
               self.control_maps.get('games', {}).get('default', {}))
    
    def map_action(self, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map game action to input sequences."""
        try:
            self.logger.info(f"Mapping action: {action}")
            game_controls = self.get_game_controls()
            self.logger.debug(f"Current game controls: {game_controls}")
            
            action_type = action.get('type', '')
            key = action.get('key', '')
            
            # Get action configuration
            action_config = self.control_maps.get('action_types', {}).get(action_type, {})
            is_continuous = action_config.get('continuous', False)
            default_duration = action_config.get('default_duration', 0.1)
            duration = action.get('duration', default_duration)
            
            mapped_actions = []
            
            # Map based on action type
            if action_type == 'look':
                # Handle look actions directly with mouse movement
                if key.startswith('look_'):
                    mapped_actions.extend(self._create_mouse_actions(
                        key, is_continuous, duration))
                else:
                    self.logger.error(f"Invalid look key: {key}")
                    
            elif action_type == 'movement':
                movement_controls = game_controls.get('keyboard', {}).get('movement', {})
                self.logger.debug(f"Movement controls: {movement_controls}")
                
                if key in movement_controls:
                    control = movement_controls[key]
                    self.logger.info(f"Mapped movement key '{key}' to control '{control}'")
                    mapped_actions.extend(self._create_keyboard_actions(
                        control, is_continuous, duration))
                else:
                    self.logger.error(f"Movement key '{key}' not found in controls: {movement_controls.keys()}")
                        
            elif action_type == 'action':
                # Get the control mapping
                control = None
                if key in game_controls.get('keyboard', {}).get('actions', {}):
                    control = game_controls['keyboard']['actions'][key]
                
                # Handle mouse actions defined in keyboard section
                if control and control.startswith('mouse'):
                    mapped_actions.extend(self._create_mouse_actions(
                        control, False, duration))
                # Handle regular keyboard actions
                elif control:
                    mapped_actions.extend(self._create_keyboard_actions(
                        control, False, duration))
                # Handle direct mouse actions
                elif key.startswith('mouse'):
                    mapped_actions.extend(self._create_mouse_actions(
                        key, False, duration))
                        
            elif action_type == 'menu':
                if key in game_controls.get('keyboard', {}).get('menu', {}):
                    control = game_controls['keyboard']['menu'][key]
                    mapped_actions.extend(self._create_keyboard_actions(
                        control, False, duration))
                        
            elif action_type == 'hotbar':
                if key in game_controls.get('keyboard', {}).get('hotbar', {}):
                    control = game_controls['keyboard']['hotbar'][key]
                    mapped_actions.extend(self._create_keyboard_actions(
                        control, False, duration))
                        
            return mapped_actions
            
        except Exception as e:
            self.logger.error(f"Error mapping action: {e}")
            return []
    
    def _create_keyboard_actions(self, key: str, continuous: bool, duration: float) -> List[Dict[str, Any]]:
        """Create keyboard action sequence."""
        if continuous:
            return [{
                'type': 'keyboard',
                'key': key,
                'press_type': 'hold',
                'duration': duration
            }]
        else:
            return [{
                'type': 'keyboard',
                'key': key,
                'press_type': 'tap',
                'duration': duration
            }]
    
    def _create_mouse_actions(self, action_input: str, continuous: bool, duration: float) -> List[Dict[str, Any]]:
        """Create mouse action sequence."""
        # Handle mouse movement (looking around)
        if action_input.startswith('look_'):
            direction = action_input.split('_')[1]  # look_left, look_right, look_up, look_down
            movement_map = {
                'left': (-50, 0),
                'right': (50, 0),
                'up': (0, -30),
                'down': (0, 30)
            }
            delta_x, delta_y = movement_map.get(direction, (0, 0))
            return [{
                'type': 'mouse',
                'subtype': 'move',
                'relative': True,
                'position': (delta_x, delta_y),
                'duration': duration
            }]
        
        # Handle mouse buttons
        button_map = {
            'mouse1': Button.left,
            'mouse2': Button.right,
            'mouse3': Button.middle
        }
        mapped_button = button_map.get(action_input, Button.left)
        
        if continuous:
            return [{
                'type': 'mouse',
                'subtype': 'click',
                'button': mapped_button,
                'press_type': 'hold',
                'duration': duration
            }]
        else:
            return [{
                'type': 'mouse',
                'subtype': 'click',
                'button': mapped_button,
                'press_type': 'click',
                'duration': duration
            }]
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a mapped action through input simulation."""
        try:
            # Log incoming action
            self.logger.info(f"Executing action: {action}")
            
            # Validate action format
            if not self.validate_action(action, {}):
                self.logger.error(f"Invalid action format: {action}")
                return False
            
            # Respect input delay
            current_time = time.time()
            time_since_last = current_time - self.last_action_time
            if time_since_last < self.config['input_delay']:
                time.sleep(self.config['input_delay'] - time_since_last)
            
            # Map high-level actions to low-level inputs
            mapped_actions = self.map_action(action)
            if not mapped_actions:
                self.logger.error(f"Failed to map action: {action}")
                return False
            
            # Log mapped actions
            self.logger.info(f"Mapped to low-level actions: {mapped_actions}")
            
            # Execute each mapped action
            success = True
            for mapped_action in mapped_actions:
                if mapped_action['type'] == 'keyboard':
                    success = success and self._execute_keyboard_action(mapped_action)
                elif mapped_action['type'] == 'mouse':
                    success = success and self._execute_mouse_action(mapped_action)
                else:
                    self.logger.error(f"Unknown mapped action type: {mapped_action['type']}")
                    success = False
                
                if not success:
                    self.logger.error(f"Failed to execute mapped action: {mapped_action}")
                    break
            
            self.last_action_time = time.time()
            
            # Log execution result
            if success:
                self.logger.info("Action executed successfully")
            else:
                self.logger.error("Action execution failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return False
    
    def _execute_keyboard_action(self, action: Dict[str, Any]) -> bool:
        """Execute keyboard input."""
        try:
            self.logger.info(f"Executing keyboard action: {action}")
            key = action['key']
            
            if isinstance(key, str):
                key = key.lower()
                # Map special keys
                special_keys = {
                    'space': Key.space,
                    'shift': Key.shift,
                    'ctrl': Key.ctrl,
                    'alt': Key.alt,
                    'esc': Key.esc,
                    'tab': Key.tab,
                    'enter': Key.enter,
                    'backspace': Key.backspace,
                    'delete': Key.delete,
                    'up': Key.up,
                    'down': Key.down,
                    'left': Key.left,
                    'right': Key.right,
                    'page_up': Key.page_up,
                    'page_down': Key.page_down,
                    'home': Key.home,
                    'end': Key.end,
                    'insert': Key.insert,
                    'f1': Key.f1,
                    'f2': Key.f2,
                    'f3': Key.f3,
                    'f4': Key.f4,
                    'f5': Key.f5,
                    'f6': Key.f6,
                    'f7': Key.f7,
                    'f8': Key.f8,
                    'f9': Key.f9,
                    'f10': Key.f10,
                    'f11': Key.f11,
                    'f12': Key.f12
                }
                
                # Handle single character keys
                if len(key) == 1:
                    self.logger.info(f"Using single character key: {key}")
                    key = key  # Keep as string for single characters
                else:
                    # Map special key or keep as is
                    key = special_keys.get(key, key)
                    self.logger.info(f"Mapped special key: {key}")
            
            try:
                if action['press_type'] == 'tap':
                    self.keyboard.press(key)
                    time.sleep(action.get('duration', 0.1))
                    self.keyboard.release(key)
                elif action['press_type'] == 'hold':
                    self.keyboard.press(key)
                    if action.get('duration'):
                        time.sleep(action['duration'])
                        self.keyboard.release(key)
                elif action['press_type'] == 'release':
                    self.keyboard.release(key)
                    
                self.logger.debug(f"Executed keyboard action: {action}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to execute keyboard action {action}: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error executing keyboard action: {e}")
            return False
    
    def _execute_mouse_action(self, action: Dict[str, Any]) -> bool:
        """Execute mouse input."""
        try:
            if action['subtype'] == 'move':
                if action.get('relative', False):
                    # Get current position
                    current_x, current_y = self.mouse.position
                    delta_x, delta_y = action['position']
                    
                    # Apply mouse sensitivity from config
                    game_controls = self.get_game_controls()
                    sensitivity = game_controls.get('mouse', {}).get('sensitivity', 1.0)
                    invert_y = game_controls.get('mouse', {}).get('invert_y', False)
                    
                    # Calculate new position with sensitivity
                    new_x = current_x + (delta_x * sensitivity)
                    new_y = current_y + (delta_y * sensitivity * (-1 if invert_y else 1))
                    
                    # Move mouse
                    self.mouse.position = (new_x, new_y)
                else:
                    self.mouse.position = action['position']
            elif action['subtype'] == 'click':
                if action['press_type'] == 'click':
                    self.mouse.click(action['button'])
                elif action['press_type'] == 'hold':
                    self.mouse.press(action['button'])
                    if action.get('duration'):
                        time.sleep(action['duration'])
                        self.mouse.release(action['button'])
                elif action['press_type'] == 'release':
                    self.mouse.release(action['button'])
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing mouse action: {e}")
            return False
    
    def validate_action(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Validate if an action can be executed in current state."""
        try:
            # Basic format validation
            if not isinstance(action, dict):
                self.logger.error("Action must be a dictionary")
                return False
            
            # Check required fields
            if 'type' not in action:
                self.logger.error("Action missing 'type' field")
                return False
            if 'key' not in action:
                self.logger.error("Action missing 'key' field")
                return False
            if 'duration' not in action:
                self.logger.error("Action missing 'duration' field")
                return False
            
            # Validate type
            valid_types = ['movement', 'action', 'menu', 'hotbar', 'look']
            if action['type'] not in valid_types:
                self.logger.error(f"Invalid action type: {action['type']}")
                return False
            
            # Validate key based on type
            game_controls = self.get_game_controls()
            if action['type'] == 'movement':
                valid_keys = game_controls.get('keyboard', {}).get('movement', {}).keys()
                if action['key'] not in valid_keys:
                    self.logger.error(f"Invalid movement key: {action['key']}")
                    return False
            
            # Validate duration
            try:
                duration = float(action['duration'])
                if duration <= 0:
                    self.logger.error(f"Invalid duration: {duration}")
                    return False
            except (ValueError, TypeError):
                self.logger.error(f"Invalid duration format: {action['duration']}")
                return False
            
            # Only check game state constraints if provided
            if game_state:
                if not self._check_cooldowns(action):
                    self.logger.warning("Action on cooldown")
                    return False
                if not self._check_requirements(action, game_state):
                    self.logger.warning("Action requirements not met")
                    return False
                if not self._check_constraints(action, game_state):
                    self.logger.warning("Action constraints not satisfied")
                    return False
            
            self.logger.info(f"Action validated successfully: {action}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating action: {e}")
            return False
    
    def _check_cooldowns(self, action: Dict[str, Any]) -> bool:
        """Check if action is on cooldown."""
        action_id = action.get('id')
        if action_id in self.active_actions:
            last_time = self.active_actions[action_id]['timestamp']
            cooldown = self.active_actions[action_id]['cooldown']
            return (time.time() - last_time) >= cooldown
        return True
    
    def _check_requirements(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Check if action requirements are met."""
        requirements = action.get('requirements', {})
        
        # Check resource requirements
        for resource, amount in requirements.get('resources', {}).items():
            if game_state.get('resources', {}).get(resource, 0) < amount:
                return False
        
        # Check state requirements
        for state_key, state_value in requirements.get('state', {}).items():
            if game_state.get(state_key) != state_value:
                return False
                
        return True
    
    def _check_constraints(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Check if action constraints are satisfied."""
        constraints = action.get('constraints', {})
        
        # Check position constraints
        if 'position' in constraints:
            current_pos = game_state.get('player', {}).get('position', (0, 0))
            target_pos = constraints['position']
            distance = ((current_pos[0] - target_pos[0]) ** 2 + 
                       (current_pos[1] - target_pos[1]) ** 2) ** 0.5
            if distance > constraints.get('max_distance', float('inf')):
                return False
        
        # Check state constraints
        if 'state' in constraints:
            for state_key, state_value in constraints['state'].items():
                if game_state.get(state_key) != state_value:
                    return False
                    
        return True
    
    def _cleanup_stuck_inputs(self) -> None:
        """Clean up any stuck keyboard or mouse inputs."""
        try:
            game_controls = self.get_game_controls()
            
            # Release all keyboard keys
            for category in game_controls.get('keyboard', {}).values():
                for key in category.values():
                    if isinstance(key, str):
                        self.keyboard.release(key)
                    elif isinstance(key, list):
                        for k in key:
                            if isinstance(k, str):
                                self.keyboard.release(k)
            
            # Release mouse buttons
            self.mouse.release(Button.left)
            self.mouse.release(Button.right)
            self.mouse.release(Button.middle)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up stuck inputs: {e}")
