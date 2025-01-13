"""
Action system components initialization.

This module initializes and connects the various action system components:
- ActionGenerator: Creates concrete actions from high-level goals
- ActionExecutor: Handles execution and lifecycle of actions
- ActionTemplates: Manages predefined action patterns
- ActionValidator: Validates and constrains actions
- ActionOptimizer: Optimizes action sequences
"""

from typing import Dict, Any, Optional

from ....action_controller import ActionController
from ...memory.state_tracker import StateTracker
from .action_executor import ActionExecutor
from .action_generator import ActionGenerator
from .action_templates import ActionTemplates
from .action_validator import ActionValidator
from .action_optimizer import ActionOptimizer

__all__ = [
    'ActionExecutor',
    'ActionGenerator',
    'ActionTemplates',
    'ActionValidator',
    'ActionOptimizer',
    'create_action_system'
]

def create_action_system(config: Dict[str, Any],
                        state_tracker: Optional[StateTracker] = None,
                        action_controller: Optional[ActionController] = None) -> Dict[str, Any]:
    """
    Create and initialize the action system components.
    
    @param {Dict[str, Any]} config - Configuration for the action system
    @param {StateTracker} state_tracker - Optional state tracker instance
    @param {ActionController} action_controller - Optional action controller instance
    @returns {Dict[str, Any]} - Initialized action system components
    """
    # Create components
    executor = ActionExecutor(state_tracker, action_controller) if state_tracker and action_controller else None
    generator = ActionGenerator(config, executor)
    templates = ActionTemplates()
    validator = ActionValidator()
    optimizer = ActionOptimizer()
    
    return {
        'executor': executor,
        'generator': generator,
        'templates': templates,
        'validator': validator,
        'optimizer': optimizer
    }
