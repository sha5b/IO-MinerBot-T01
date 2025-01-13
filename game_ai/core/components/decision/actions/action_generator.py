"""Action generation component."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class ActionGenerator:
    """Generates concrete actions from high-level goals."""
    
    def __init__(self, config: Dict[str, Any], executor: Optional['ActionExecutor'] = None):
        """
        Initialize action generator.
        
        @param {Dict[str, Any]} config - Configuration settings
        @param {ActionExecutor} executor - Optional executor for direct action execution
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.executor = executor
        
        # Action generation settings
        self.settings = {
            'max_sequence_length': config.get('action', {}).get('max_sequence_length', 20),
            'min_duration': config.get('action', {}).get('min_duration', 0.1),
            'max_duration': config.get('action', {}).get('max_duration', 5.0),
            'default_duration': 0.5
        }
        
        # Action mappings for different goals
        self.goal_mappings = {
            'explore': self._generate_explore_actions,
            'gather': self._generate_gather_actions,
            'combat': self._generate_combat_actions,
            'navigate': self._generate_navigation_actions,
            'interact': self._generate_interaction_actions
        }
    
    def generate_actions(self, goal: Dict[str, Any], 
                        game_state: Dict[str, Any],
                        execute: bool = False) -> List[Dict[str, Any]]:
        """
        Generate action sequence for goal.
        
        @param {Dict[str, Any]} goal - Goal to generate actions for
        @param {Dict[str, Any]} game_state - Current game state
        @param {bool} execute - Whether to execute actions immediately if executor available
        @returns {List[Dict[str, Any]]} - Generated action sequence
        """
        try:
            # Get goal type and generator
            goal_type = goal.get('type', 'explore')
            generator = self.goal_mappings.get(goal_type, self._generate_default_actions)
            
            # Generate actions
            actions = generator(goal, game_state)
            
            # Format and validate actions
            actions = self._format_actions(actions)
            
            # Execute if requested and executor available
            if execute and self.executor:
                for action in actions:
                    if not self.executor.execute_action(action):
                        self.logger.warning(f"Failed to execute action: {action}")
                        break
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating actions: {e}")
            return self._get_fallback_actions()
    
    def _generate_explore_actions(self, goal: Dict[str, Any], 
                                game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate exploration action sequence."""
        try:
            actions = []
            
            # Look around pattern
            actions.extend([
                {'type': 'look', 'key': 'look_left', 'duration': 0.2},
                {'type': 'look', 'key': 'look_right', 'duration': 0.2},
                {'type': 'look', 'key': 'look_up', 'duration': 0.2}
            ])
            
            # Move forward
            actions.append(
                {'type': 'movement', 'key': 'forward', 'duration': 1.0}
            )
            
            # Add interaction if target specified
            if 'target' in goal:
                actions.append(
                    {'type': 'action', 'key': 'mouse1', 'duration': 0.5}
                )
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating explore actions: {e}")
            return []
    
    def _generate_gather_actions(self, goal: Dict[str, Any], 
                               game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate resource gathering action sequence."""
        try:
            actions = []
            
            # Approach resource
            actions.append(
                {'type': 'movement', 'key': 'forward', 'duration': 0.3}
            )
            
            # Look at resource
            actions.append(
                {'type': 'look', 'key': 'look_up', 'duration': 0.2}
            )
            
            # Gather action
            actions.append(
                {'type': 'action', 'key': 'mouse1', 'duration': 1.0}
            )
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating gather actions: {e}")
            return []
    
    def _generate_combat_actions(self, goal: Dict[str, Any], 
                               game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate combat action sequence."""
        try:
            actions = []
            
            # Check health for combat style
            health = game_state.get('player', {}).get('health', 100)
            
            if health < 30:
                # Defensive actions
                actions.extend([
                    {'type': 'movement', 'key': 'backward', 'duration': 0.5},
                    {'type': 'action', 'key': 'mouse2', 'duration': 1.0}
                ])
            else:
                # Offensive actions
                actions.extend([
                    {'type': 'movement', 'key': 'forward', 'duration': 0.3},
                    {'type': 'action', 'key': 'mouse1', 'duration': 0.5},
                    {'type': 'movement', 'key': 'backward', 'duration': 0.2}
                ])
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating combat actions: {e}")
            return []
    
    def _generate_navigation_actions(self, goal: Dict[str, Any], 
                                   game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate navigation action sequence."""
        try:
            actions = []
            
            # Get target position
            target = goal.get('target', {})
            current = game_state.get('position', {})
            
            # Calculate direction (simplified)
            if target.get('x', 0) > current.get('x', 0):
                actions.append(
                    {'type': 'movement', 'key': 'right', 'duration': 0.5}
                )
            else:
                actions.append(
                    {'type': 'movement', 'key': 'left', 'duration': 0.5}
                )
            
            # Move forward
            actions.append(
                {'type': 'movement', 'key': 'forward', 'duration': 1.0}
            )
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating navigation actions: {e}")
            return []
    
    def _generate_interaction_actions(self, goal: Dict[str, Any], 
                                    game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate interaction action sequence."""
        try:
            actions = []
            
            # Look at target
            actions.append(
                {'type': 'look', 'key': 'look_up', 'duration': 0.2}
            )
            
            # Approach if needed
            if goal.get('approach', True):
                actions.append(
                    {'type': 'movement', 'key': 'forward', 'duration': 0.3}
                )
            
            # Interact based on type
            if goal.get('interaction_type') == 'use':
                actions.append(
                    {'type': 'action', 'key': 'mouse2', 'duration': 0.5}
                )
            else:
                actions.append(
                    {'type': 'action', 'key': 'mouse1', 'duration': 0.3}
                )
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating interaction actions: {e}")
            return []
    
    def _generate_default_actions(self, goal: Dict[str, Any], 
                                game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default action sequence."""
        return [
            {'type': 'look', 'key': 'look_left', 'duration': 0.2},
            {'type': 'movement', 'key': 'forward', 'duration': 0.5}
        ]
    
    def execute_actions(self, actions: List[Dict[str, Any]]) -> bool:
        """
        Execute a sequence of actions.
        
        @param {List[Dict[str, Any]]} actions - Actions to execute
        @returns {bool} - Success status
        """
        if not self.executor:
            self.logger.error("No executor available for action execution")
            return False
            
        success = True
        for action in actions:
            if not self.executor.execute_action(action):
                self.logger.warning(f"Failed to execute action: {action}")
                success = False
                break
                
        return success
        
    def _format_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format and validate action sequence.
        
        @param {List[Dict[str, Any]]} actions - Actions to format
        @returns {List[Dict[str, Any]]} - Formatted actions
        """
        try:
            formatted = []
            
            for action in actions:
                # Ensure required fields
                if 'type' not in action or 'key' not in action:
                    continue
                
                # Format duration
                duration = min(
                    self.settings['max_duration'],
                    max(
                        self.settings['min_duration'],
                        action.get('duration', self.settings['default_duration'])
                    )
                )
                
                # Create formatted action
                formatted.append({
                    'type': action['type'],
                    'key': action['key'],
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Limit sequence length
            if len(formatted) > self.settings['max_sequence_length']:
                formatted = formatted[:self.settings['max_sequence_length']]
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting actions: {e}")
            return actions
    
    def _get_fallback_actions(self) -> List[Dict[str, Any]]:
        """Get basic fallback action sequence."""
        return self._format_actions([
            {'type': 'look', 'key': 'look_left', 'duration': 0.2},
            {'type': 'movement', 'key': 'forward', 'duration': 0.5}
        ])
    
    def add_goal_mapping(self, goal_type: str, generator_func: callable) -> None:
        """
        Add new goal action mapping.
        
        Args:
            goal_type (str): Type of goal
            generator_func (callable): Function to generate actions
        """
        try:
            self.goal_mappings[goal_type] = generator_func
            self.logger.info(f"Added goal mapping: {goal_type}")
        except Exception as e:
            self.logger.error(f"Error adding goal mapping: {e}")
