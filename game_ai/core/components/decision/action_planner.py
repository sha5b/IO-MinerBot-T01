"""Action planning component for generating executable action sequences."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .actions import (
    ActionGenerator,
    ActionExecutor,
    ActionTemplates,
    ActionValidator,
    ActionOptimizer,
    create_action_system
)

class ActionPlanner:
    """
    Generates concrete action sequences from strategies.
    Delegates to specialized components for generation, validation, and optimization.
    """
    
    def __init__(self, config: Dict[str, Any], action_system: Optional[Dict[str, Any]] = None):
        """
        Initialize action planner.
        
        @param {Dict[str, Any]} config - Configuration settings
        @param {Dict[str, Any]} action_system - Optional pre-configured action system
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize or use provided action system
        if action_system:
            self.action_system = action_system
        else:
            self.action_system = create_action_system(config)
            
        # Extract components
        self.generator = self.action_system['generator']
        self.executor = self.action_system.get('executor')
        self.templates = self.action_system['templates']
        self.validator = self.action_system['validator']
        self.optimizer = self.action_system['optimizer']
        
        # Register default templates
        self._register_default_templates()
        
    def _register_default_templates(self):
        """Register default action templates."""
        templates = {
            'movement': {
                'explore': [
                    {'type': 'look', 'key': 'look_left', 'duration': 0.2},
                    {'type': 'movement', 'key': 'forward', 'duration': 1.0},
                    {'type': 'look', 'key': 'look_right', 'duration': 0.2}
                ],
                'approach': [
                    {'type': 'movement', 'key': 'forward', 'duration': 0.5}
                ],
                'retreat': [
                    {'type': 'movement', 'key': 'backward', 'duration': 0.5},
                    {'type': 'look', 'key': 'look_left', 'duration': 0.2},
                    {'type': 'look', 'key': 'look_right', 'duration': 0.2}
                ]
            },
            'resource': {
                'gather': [
                    {'type': 'action', 'key': 'mouse1', 'duration': 1.0}
                ],
                'collect': [
                    {'type': 'movement', 'key': 'forward', 'duration': 0.3},
                    {'type': 'action', 'key': 'mouse1', 'duration': 0.5}
                ]
            },
            'combat': {
                'attack': [
                    {'type': 'action', 'key': 'mouse1', 'duration': 0.5},
                    {'type': 'movement', 'key': 'backward', 'duration': 0.3}
                ],
                'defend': [
                    {'type': 'action', 'key': 'mouse2', 'duration': 1.0},
                    {'type': 'movement', 'key': 'backward', 'duration': 0.5}
                ]
            },
            'interaction': {
                'use': [
                    {'type': 'action', 'key': 'mouse2', 'duration': 0.5}
                ],
                'activate': [
                    {'type': 'action', 'key': 'mouse1', 'duration': 0.3}
                ]
            }
        }
    
        # Register with templates component
        for category, patterns in templates.items():
            for action_type, sequence in patterns.items():
                self.templates.add_template(category, action_type, sequence)
                
    def plan_actions(self, strategy: Dict[str, Any], game_state: Dict[str, Any],
                    constraints: Dict[str, Any], execute: bool = False) -> List[Dict[str, Any]]:
        """
        Generate action sequence from strategy.
        
        @param {Dict[str, Any]} strategy - Current strategy
        @param {Dict[str, Any]} game_state - Current game state
        @param {Dict[str, Any]} constraints - Action constraints
        @param {bool} execute - Whether to execute actions immediately
        @returns {List[Dict[str, Any]]} - Sequence of executable actions
        """
        try:
            # Get current objective
            objective = self._get_current_objective(strategy)
            if not objective:
                return []
            
            # Get template from objective
            template = self.templates.get_template(
                objective.get('type'),
                objective.get('goal')
            )
            
            # Generate action sequence
            actions = self.generator.generate_actions({
                'type': objective.get('type'),
                'template': template,
                'target': objective.get('target'),
                'constraints': constraints
            }, game_state)
            
            # Validate actions
            actions = [
                action for action in actions
                if self.validator.validate_action(action, game_state)
            ]
            
            # Optimize sequence
            actions = self.optimizer.optimize_sequence(
                actions,
                game_state  # Only pass sequence and game_state
            )
            
            # Execute if requested
            if execute and self.executor:
                self.generator.execute_actions(actions)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error planning actions: {e}")
            return self._get_fallback_sequence()
    
    def _get_current_objective(self, strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get highest priority objective.
        
        @param {Dict[str, Any]} strategy - Current strategy
        @returns {Optional[Dict[str, Any]]} - Current objective or None
        """
        objectives = strategy.get('objectives', [])
        return objectives[0] if objectives else None
    
    
    def _get_fallback_sequence(self) -> List[Dict[str, Any]]:
        """Get basic fallback action sequence."""
        return self.generator.generate_actions({
            'type': 'explore',
            'template': self.templates.get_template('movement', 'explore'),
            'constraints': {}
        }, {})
