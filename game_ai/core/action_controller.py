"""Action controller for executing game actions through input simulation."""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

class ActionController:
    """Handles execution of game actions through input simulation."""
    
    def __init__(self, config_path: Optional[Path] = None, game_type: str = "default", vision_system=None, window_manager=None):
        """
        Initialize the action controller.
        
        Args:
            config_path (Path, optional): Path to configuration file
            game_type (str): Type of game to load controls for (e.g., "minecraft", "fps_template")
            vision_system: Reference to VisionSystem instance for movement feedback
            window_manager: Reference to WindowManager for window control
        """
        self.logger = logging.getLogger(__name__)
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.config = self._load_config(config_path)
        self.control_maps = self._load_control_maps()
        self.active_actions: Dict[str, Any] = {}
        self.last_action_time = time.time()
        self.game_type = game_type
        self.vision_system = vision_system
        self.window_manager = window_manager
        self.last_vision_state = None
        self.mouse_locked = False
        
    def lock_mouse(self) -> bool:
        """Lock mouse to game window."""
        if self.window_manager:
            if self.window_manager.focus_game_window():
                self.mouse_locked = True
                return True
        return False
    
    def unlock_mouse(self) -> None:
        """Release mouse lock."""
        self.mouse_locked = False
        if self.window_manager:
            self.window_manager.release_game_window()
    
    def _handle_esc_key(self, key: str) -> bool:
        """Handle ESC key to unlock mouse."""
        if key == 'esc' and self.mouse_locked:
            self.unlock_mouse()
            return True
        return False
    
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
            is_continuous = action.get('continuous', action_config.get('continuous', False))
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
    
    def plot_complex_path(self, waypoints: List[Tuple[float, float, float]], 
                         smoothness: float = 1.0) -> List[Dict[str, Any]]:
        """
        Plot a path through multiple waypoints.
        
        Args:
            waypoints: List of (x, y, z) positions to move through
            smoothness: Path smoothness factor (0.5-2.0, higher = smoother but slower)
            
        Returns:
            List of actions to execute the path
        """
        try:
            if len(waypoints) < 2:
                self.logger.error("Need at least 2 waypoints to plot path")
                return []
            
            actions = []
            
            # Plot path segments between consecutive waypoints
            for i in range(len(waypoints) - 1):
                current = waypoints[i]
                target = waypoints[i + 1]
                
                # Get actions for this segment
                segment_actions = self.plot_path(current, target, smoothness)
                
                # Add small pause between segments for smoother transitions
                if segment_actions and i < len(waypoints) - 2:
                    actions.extend(segment_actions)
                    actions.append({
                        'type': 'movement',
                        'key': 'forward',
                        'duration': 0.1 * smoothness
                    })
                else:
                    actions.extend(segment_actions)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error plotting complex path: {e}")
            return []
    
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
            
            # Release mouse lock
            if self.mouse_locked:
                self.unlock_mouse()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up stuck inputs: {e}")

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
            # Parse direction and optional intensity
            parts = action_input.split('_')
            direction = parts[1]
            intensity = float(parts[2]) if len(parts) > 2 else 1.0
            
            # Pure Minecraft-style mouse movement
            base_speed = 3.0  # Very small base speed for precise control
            
            # Only allow cardinal direction movements
            if direction == 'left':
                delta_x = -base_speed
                delta_y = 0
            elif direction == 'right':
                delta_x = base_speed
                delta_y = 0
            elif direction == 'up':
                delta_x = 0
                delta_y = -base_speed
            elif direction == 'down':
                delta_x = 0
                delta_y = base_speed
            else:
                # Ignore diagonal movements
                return []
                    
            # Basic intensity scaling
            delta_x *= intensity
            delta_y *= intensity
            
            # Single movement action
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
            
            # Handle ESC key specially
            if self._handle_esc_key(key):
                return True
            
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
            if not self.mouse_locked and not self.lock_mouse():
                self.logger.error("Failed to lock mouse to game window")
                return False
                
            if action['subtype'] == 'move':
                if action.get('relative', False):
                    delta_x, delta_y = action['position']
                    
                    # Get mouse settings from config
                    game_controls = self.get_game_controls()
                    sensitivity = game_controls.get('mouse', {}).get('sensitivity', 1.0)
                    invert_y = game_controls.get('mouse', {}).get('invert_y', False)
                    dead_zone = game_controls.get('mouse', {}).get('dead_zone', 0.05)
                    
                    # Apply dead zone
                    if abs(delta_x) < dead_zone:
                        delta_x = 0
                    if abs(delta_y) < dead_zone:
                        delta_y = 0
                    
                    # Apply sensitivity and send relative movement
                    if self.window_manager:
                        self.window_manager.move_mouse_relative(
                            delta_x * sensitivity,
                            delta_y * sensitivity * (-1 if invert_y else 1)
                        )
                    else:
                        # Fallback to direct mouse control
                        current_x, current_y = self.mouse.position
                        new_x = current_x + (delta_x * sensitivity)
                        new_y = current_y + (delta_y * sensitivity * (-1 if invert_y else 1))
                        self.mouse.position = (new_x, new_y)
                    
                    # Small sleep to allow game to process movement
                    time.sleep(0.001)
                else:
                    if self.window_manager:
                        self.window_manager.set_mouse_position(*action['position'])
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
            elif action['type'] == 'look':
                # Only allow cardinal directions
                valid_directions = ['left', 'right', 'up', 'down']
                parts = action['key'].split('_')
                if len(parts) < 2 or not action['key'].startswith('look_'):
                    self.logger.error(f"Invalid look key format: {action['key']}")
                    return False
                    
                direction = parts[1]
                if direction not in valid_directions:
                    self.logger.error(f"Invalid look direction: {direction}")
                    return False
                    
                # Validate intensity if provided
                if len(parts) > 2:
                    try:
                        intensity = float(parts[2])
                        if intensity <= 0 or intensity > 2.0:  # Limit max intensity
                            self.logger.error(f"Invalid look intensity: {intensity}")
                            return False
                    except ValueError:
                        self.logger.error(f"Invalid look intensity format: {action['key']}")
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
    
    def _smooth_start(self, t: float) -> float:
        """Enhanced acceleration curve."""
        return t * t * (3 - 2 * t)  # Cubic bezier
    
    def _smooth_end(self, t: float) -> float:
        """Enhanced deceleration curve."""
        return t * t * (3 - 2 * t)  # Cubic bezier
    
    def _smooth_step(self, t: float) -> float:
        """Enhanced smooth step function."""
        t = t * t * (3 - 2 * t)  # Basic smoothstep
        return (1 - math.cos(t * math.pi)) / 2  # Further smoothing
    
    def _calculate_look_direction(self, screen_x: float, screen_y: float, intensity: float = 1.0) -> str:
        """
        Calculate cardinal look direction based on screen coordinates.
        
        Args:
            screen_x: X coordinate on screen
            screen_y: Y coordinate on screen
            intensity: Movement intensity (0.0-2.0)
            
        Returns:
            Look direction command (e.g., 'look_left_1.0')
        """
        try:
            # Get screen center
            center_x = 400
            center_y = 300
            if self.vision_system and self.last_vision_state:
                frame_shape = self.last_vision_state.get('frame_shape', (600, 800))
                center_y, center_x = frame_shape[0] / 2, frame_shape[1] / 2
            
            # Calculate distances from center
            dx = screen_x - center_x
            dy = screen_y - center_y
            
            # Choose primary direction only (no diagonals)
            if abs(dx) > abs(dy):
                direction = 'right' if dx > 0 else 'left'
            else:
                direction = 'down' if dy > 0 else 'up'
            
            # Simple intensity scaling based on distance
            distance = max(abs(dx), abs(dy))
            max_distance = max(center_x, center_y)
            scaled_intensity = min(2.0, max(0.1, (distance / max_distance) * intensity))
            
            return f'look_{direction}_{scaled_intensity:.1f}'
            
        except Exception as e:
            self.logger.error(f"Error calculating look direction: {e}")
            return 'look_right_1.0'  # Safe default
    
    def _update_vision_state(self) -> Dict[str, Any]:
        """Update and return the current vision state."""
        if self.vision_system:
            frame = self.vision_system.capture_screen()
            self.last_vision_state = self.vision_system.process_frame(frame)
            return self.last_vision_state
        return {}

    def _adjust_movement_from_vision(self, actions: List[Dict[str, Any]], 
                                   vision_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Minimal vision-based movement adjustment."""
        if not vision_state or not actions:
            return actions

        # Only check if we need to look at a target
        analysis = vision_state.get('analysis', {})
        if 'player' in analysis and 'position' in analysis['player']:
            player_pos = analysis['player']['position']
            
            # Find closest target (enemy, resource, etc)
            closest_target = None
            min_distance = float('inf')
            
            for target_type in ['threats', 'resources']:
                targets = analysis.get('environment', {}).get(target_type, [])
                for target in targets:
                    if 'position' in target and 'distance' in target:
                        if target['distance'] < min_distance:
                            min_distance = target['distance']
                            closest_target = target['position']
            
            # If we found a target, adjust look direction
            if closest_target:
                dx = closest_target[0] - player_pos[0]
                dy = closest_target[1] - player_pos[1]
                
                # Add a look action to face target
                if abs(dx) > abs(dy):
                    actions.insert(0, {
                        'type': 'look',
                        'key': f'look_{"right" if dx > 0 else "left"}_1.0',
                        'duration': 0.1,
                        'continuous': True
                    })
                else:
                    actions.insert(0, {
                        'type': 'look',
                        'key': f'look_{"down" if dy > 0 else "up"}_1.0',
                        'duration': 0.1,
                        'continuous': True
                    })
        
        return actions

    def _create_feature_based_adjustments(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create micro-adjustments based on feature tracking."""
        if not matches:
            return []
            
        # Calculate average feature movement
        avg_distance = sum(m['distance'] for m in matches) / len(matches)
        
        # Create subtle look adjustments
        adjustments = []
        if avg_distance > 0.5:  # Only adjust if significant movement
            intensity = min(1.0, avg_distance / 10)
            adjustments.append({
                'type': 'look',
                'key': f'look_right_{intensity}',
                'duration': 0.05,
                'continuous': True
            })
            
        return adjustments

    def plot_path(self, current_pos: Tuple[float, float, float],
                 target_pos: Tuple[float, float, float],
                 smoothness: float = 1.0,
                 look_ahead: float = 2.0) -> List[Dict[str, Any]]:
        """
        Plot an enhanced path from current position to target position with look-ahead.
        
        Args:
            current_pos: Current (x, y, z) position
            target_pos: Target (x, y, z) position
            smoothness: Path smoothness factor (0.5-2.0, higher = smoother but slower)
            look_ahead: How far ahead to look while moving (in blocks)
            
        Returns:
            List of actions to execute the path
        """
        try:
            # Get current vision state
            vision_state = self._update_vision_state()
            
            # Enhanced path planning with look-ahead and vision feedback
            actions = []
            total_distance = math.sqrt(
                (target_pos[0] - current_pos[0]) ** 2 +
                (target_pos[1] - current_pos[1]) ** 2 +
                (target_pos[2] - current_pos[2]) ** 2
            )
            
            # Adjust segment size based on terrain complexity
            terrain_type = vision_state.get('analysis', {}).get('environment', {}).get('terrain_type', 'moderate')
            if terrain_type == 'dense':
                look_ahead *= 0.5  # Smaller segments in complex terrain
            elif terrain_type == 'sparse':
                look_ahead *= 1.5  # Larger segments in open areas
            
            # Break movement into smaller segments for better control
            num_segments = max(1, int(total_distance / look_ahead))
            
            for i in range(num_segments + 1):
                # Calculate intermediate target with look-ahead
                t = min(1.0, (i + 1) / num_segments)
                intermediate_pos = (
                    current_pos[0] + (target_pos[0] - current_pos[0]) * t,
                    current_pos[1] + (target_pos[1] - current_pos[1]) * t,
                    current_pos[2] + (target_pos[2] - current_pos[2]) * t
                )
                
                # Calculate direction to intermediate target
                dx = intermediate_pos[0] - current_pos[0]
                dz = intermediate_pos[2] - current_pos[2]
                dy = intermediate_pos[1] - current_pos[1]
                
                # Enhanced angle calculations
                target_yaw = math.degrees(math.atan2(-dx, dz))
                horizontal_distance = math.sqrt(dx * dx + dz * dz)
                target_pitch = -math.degrees(math.atan2(dy, horizontal_distance))
                
                # Simultaneous look and move for smoother motion
                if abs(dx) > 0.1 or abs(dz) > 0.1 or abs(dy) > 0.1:
                    # Calculate turn intensity based on angle difference
                    turn_intensity = min(2.0, math.sqrt(
                        (target_yaw ** 2 + target_pitch ** 2) / 2025.0))  # Max 45 degrees
                    
                    # Determine primary look direction
                    if abs(target_yaw) > abs(target_pitch):
                        look_dir = 'right' if target_yaw < 0 else 'left'
                    else:
                        look_dir = 'up' if target_pitch < 0 else 'down'
                    
                    # Add smooth look action
                    actions.append({
                        'type': 'look',
                        'key': f'look_{look_dir}_{turn_intensity}',
                        'duration': 0.15 * smoothness,
                        'continuous': True
                    })
                
                # Calculate segment distance and duration
                segment_distance = math.sqrt(
                    (intermediate_pos[0] - current_pos[0]) ** 2 +
                    (intermediate_pos[1] - current_pos[1]) ** 2 +
                    (intermediate_pos[2] - current_pos[2]) ** 2
                )
                
                # Add movement with dynamic duration
                if segment_distance > 0.1:
                    movement_duration = segment_distance * 0.12 * smoothness
                    actions.append({
                        'type': 'movement',
                        'key': 'forward',
                        'duration': movement_duration,
                        'continuous': True
                    })
                
                # Update current position for next segment
                current_pos = intermediate_pos
            
            # Adjust actions based on vision feedback
            adjusted_actions = self._adjust_movement_from_vision(actions, vision_state)
            
            return adjusted_actions
            
        except Exception as e:
            self.logger.error(f"Error plotting path: {e}")
            return []
