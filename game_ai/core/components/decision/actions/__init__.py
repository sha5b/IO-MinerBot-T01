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
import logging

from ....action_controller import ActionController
from ...memory.state_tracker import StateTracker
from .action_executor import ActionExecutor
from .action_generator import ActionGenerator
from .action_templates import ActionTemplates
from .action_validator import ActionValidator
from .action_optimizer import ActionOptimizer

logger = logging.getLogger(__name__)

__all__ = [
    'ActionExecutor',
    'ActionGenerator',
    'ActionTemplates',
    'ActionValidator',
    'ActionOptimizer',
    'create_action_system'
]

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate action system configuration.
    
    @param {Dict[str, Any]} config - Configuration to validate
    @returns {bool} - Whether configuration is valid
    """
    try:
        # Check action settings
        action_config = config.get('action', {})
        required_action_settings = {
            'max_sequence_length': (int, lambda x: x > 0),
            'min_duration': (float, lambda x: x > 0),
            'max_duration': (float, lambda x: x > 0),
            'timeout_multiplier': (float, lambda x: x > 0)
        }
        
        for setting, (type_, validator) in required_action_settings.items():
            value = action_config.get(setting)
            if value is None:
                logger.error(f"Missing required action setting: {setting}")
                return False
            if not isinstance(value, type_):
                logger.error(f"Invalid type for {setting}: expected {type_}, got {type(value)}")
                return False
            if not validator(value):
                logger.error(f"Invalid value for {setting}: {value}")
                return False
                
        # Validate relationships
        if action_config['min_duration'] >= action_config['max_duration']:
            logger.error("min_duration must be less than max_duration")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        return False

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
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Invalid action system configuration")
        
    # Create components with validated config
    templates = ActionTemplates(config)
    validator = ActionValidator(config)
    optimizer = ActionOptimizer(config)
    executor = ActionExecutor(config, state_tracker, action_controller) if state_tracker and action_controller else None
    generator = ActionGenerator(config, executor)
    
    return {
        'executor': executor,
        'generator': generator,
        'templates': templates,
        'validator': validator,
        'optimizer': optimizer
    }
