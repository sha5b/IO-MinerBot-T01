"""Action template management component."""

from typing import Dict, List, Any
import logging

class ActionTemplates:
    """Manages predefined action patterns and templates."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize action templates.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Action templates for different goals
        self.templates = {
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
        
        # Template categories and types
        self.categories = list(self.templates.keys())
        self.types = {
            category: list(templates.keys())
            for category, templates in self.templates.items()
        }
    
    def get_template(self, category: str, action_type: str) -> List[Dict[str, Any]]:
        """
        Get action template for given category and type.
        
        Args:
            category (str): Action category
            action_type (str): Type of action
            
        Returns:
            list: Action sequence template
        """
        try:
            return self.templates.get(category, {}).get(action_type, [])
        except Exception as e:
            self.logger.error(f"Error getting template: {e}")
            return []
    
    def add_template(self, category: str, action_type: str,
                    template: List[Dict[str, Any]]) -> None:
        """
        Add new action template.
        
        Args:
            category (str): Action category
            action_type (str): Type of action
            template (list): Action sequence template
        """
        try:
            # Validate template format
            if not self._validate_template(template):
                self.logger.error("Invalid template format")
                return
            
            # Add template
            if category not in self.templates:
                self.templates[category] = {}
                self.categories.append(category)
                self.types[category] = []
            
            self.templates[category][action_type] = template
            if action_type not in self.types[category]:
                self.types[category].append(action_type)
            
            self.logger.info(f"Added template: {category}/{action_type}")
            
        except Exception as e:
            self.logger.error(f"Error adding template: {e}")
    
    def remove_template(self, category: str, action_type: str) -> None:
        """
        Remove action template.
        
        Args:
            category (str): Action category
            action_type (str): Type of action
        """
        try:
            if category in self.templates and action_type in self.templates[category]:
                del self.templates[category][action_type]
                self.types[category].remove(action_type)
                
                # Remove category if empty
                if not self.templates[category]:
                    del self.templates[category]
                    self.categories.remove(category)
                    del self.types[category]
                
                self.logger.info(f"Removed template: {category}/{action_type}")
                
        except Exception as e:
            self.logger.error(f"Error removing template: {e}")
    
    def get_categories(self) -> List[str]:
        """
        Get available action categories.
        
        Returns:
            list: Action categories
        """
        return self.categories.copy()
    
    def get_types(self, category: str) -> List[str]:
        """
        Get available action types for category.
        
        Args:
            category (str): Action category
            
        Returns:
            list: Action types
        """
        return self.types.get(category, []).copy()
    
    def _validate_template(self, template: List[Dict[str, Any]]) -> bool:
        """Validate template format."""
        try:
            if not isinstance(template, list):
                return False
            
            for action in template:
                if not isinstance(action, dict):
                    return False
                    
                # Check required fields
                if not all(k in action for k in ['type', 'key', 'duration']):
                    return False
                    
                # Check field types
                if not isinstance(action['type'], str):
                    return False
                if not isinstance(action['key'], str):
                    return False
                if not isinstance(action['duration'], (int, float)):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_default_template(self) -> List[Dict[str, Any]]:
        """Get basic fallback template."""
        return [
            {
                'type': 'look',
                'key': 'look_left',
                'duration': 0.2
            },
            {
                'type': 'movement',
                'key': 'forward',
                'duration': 0.5
            }
        ]
