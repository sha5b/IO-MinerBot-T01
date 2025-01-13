"""
Action Executor Component

Handles the execution of actions, manages their lifecycle, and provides execution status/feedback.
Integrates with StateTracker for updates and ActionController for input execution.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from ...memory.state_tracker import StateTracker
from ....action_controller import ActionController

logger = logging.getLogger(__name__)

class ActionExecutor:
    """
    Handles the execution of actions and manages their lifecycle.
    
    @property {Dict} active_actions - Currently executing actions
    @property {Dict} action_handlers - Registered handlers for different action types
    """
    
    def __init__(self, state_tracker: StateTracker, action_controller: ActionController):
        """
        Initialize the ActionExecutor.
        
        @param {StateTracker} state_tracker - System state tracker for updates
        """
        self.active_actions = {}  # Track currently executing actions
        self.action_handlers = {}  # Map action types to their handlers
        self.state_tracker = state_tracker
        self.action_controller = action_controller
        
        # Register state update callback
        self.state_tracker.register_callback(
            'action_state',
            self._handle_state_update
        )
        
        # Register default handlers
        self._register_default_handlers()
        
    def register_handler(self, action_type: str, handler: Callable):
        """
        Register a handler function for a specific action type.
        
        @param {str} action_type - Type of action this handler manages
        @param {Callable} handler - Function that executes the action
        """
        self.action_handlers[action_type] = handler
        logger.debug(f"Registered handler for action type: {action_type}")
        
    def _handle_state_update(self, state_data: Dict[str, Any]):
        """
        Handle updates from state tracker.
        
        @param {Dict[str, Any]} state_data - Updated state information
        """
        # Update action states based on vision/state feedback
        for action_id, action in self.active_actions.items():
            if self._validate_action_state(action_id, state_data):
                self.complete_action(action_id, True)
            elif self._detect_action_failure(action_id, state_data):
                self.complete_action(action_id, False)
    
    def _register_default_handlers(self):
        """Register default action handlers that use ActionController."""
        self.register_handler('movement', self._handle_movement)
        self.register_handler('look', self._handle_look)
        self.register_handler('action', self._handle_game_action)
        self.register_handler('menu', self._handle_menu_action)
        self.register_handler('hotbar', self._handle_hotbar_action)
    
    def _handle_movement(self, action_data: Dict[str, Any]) -> bool:
        """Handle movement type actions."""
        return self.action_controller.execute_action(action_data)
    
    def _handle_look(self, action_data: Dict[str, Any]) -> bool:
        """Handle look type actions."""
        return self.action_controller.execute_action(action_data)
    
    def _handle_game_action(self, action_data: Dict[str, Any]) -> bool:
        """Handle game actions like mouse clicks."""
        return self.action_controller.execute_action(action_data)
    
    def _handle_menu_action(self, action_data: Dict[str, Any]) -> bool:
        """Handle menu navigation actions."""
        return self.action_controller.execute_action(action_data)
    
    def _handle_hotbar_action(self, action_data: Dict[str, Any]) -> bool:
        """Handle hotbar selection actions."""
        return self.action_controller.execute_action(action_data)
    
    def _validate_action_state(self, action_id: str, state_data: Dict[str, Any]) -> bool:
        """
        Validate action completion through state/vision data.
        
        @param {str} action_id - ID of action to validate
        @param {Dict[str, Any]} state_data - Current state data
        @returns {bool} - Whether action is complete
        """
        action = self.active_actions[action_id]
        action_type = action['data']['type']
        action_key = action['data']['key']
        
        # Use ActionController's validation
        return self.action_controller.validate_action(
            action['data'],
            state_data.get('game_state', {})
        )
            
        return False
    
    def _detect_action_failure(self, action_id: str, state_data: Dict[str, Any]) -> bool:
        """
        Detect if an action has failed based on state data.
        
        @param {str} action_id - ID of action to check
        @param {Dict[str, Any]} state_data - Current state data
        @returns {bool} - Whether action has failed
        """
        action = self.active_actions[action_id]
        start_time = datetime.fromisoformat(action['data']['timestamp'])
        current_time = datetime.now()
        
        # Check for timeout
        if (current_time - start_time).total_seconds() > action['data']['duration'] * 2:
            logger.warning(f"Action {action_id} timed out")
            return True
            
        return False
    
    
    def execute_action(self, action_data: Dict[str, Any]) -> bool:
        """
        Begin execution of a new action.
        
        @param {Dict[str, Any]} action_data - Action parameters and data including:
            type: Action type (movement, look, action)
            key: Specific command
            duration: Time to complete
            timestamp: Start time
        @returns {bool} - Success status of action initiation
        """
        # Generate action ID from data
        action_id = f"{action_data['type']}_{action_data['key']}_{action_data['timestamp']}"
        
        if action_id in self.active_actions:
            logger.warning(f"Action {action_id} is already executing")
            return False
            
        if action_data['type'] not in self.action_handlers:
            logger.error(f"No handler registered for action type: {action_data['type']}")
            return False
            
        try:
            handler = self.action_handlers[action_data['type']]
            self.active_actions[action_id] = {
                'data': action_data,
                'status': 'running',
                'handler': handler
            }
            # Update state tracker
            self.state_tracker.update_state('action_state', {
                'active_action': action_id,
                'action_data': action_data,
                'status': 'running'
            })
            
            # Execute handler
            handler(action_data)
            logger.info(f"Started execution of action {action_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute action {action_id}: {str(e)}")
            return False
            
    def update_action(self, action_id: str) -> Optional[str]:
        """
        Update the status of an executing action.
        
        @param {str} action_id - ID of action to update
        @returns {Optional[str]} - Current status of the action or None if not found
        """
        if action_id not in self.active_actions:
            return None
            
        action = self.active_actions[action_id]
        try:
            # Call handler's update method if it exists
            if hasattr(action['handler'], 'update'):
                action['handler'].update(action['data'])
            return action['status']
        except Exception as e:
            logger.error(f"Error updating action {action_id}: {str(e)}")
            action['status'] = 'error'
            return 'error'
            
    def complete_action(self, action_id: str, success: bool = True):
        """
        Mark an action as completed and clean up.
        
        @param {str} action_id - ID of action to complete
        @param {bool} success - Whether the action completed successfully
        """
        if action_id in self.active_actions:
            action = self.active_actions[action_id]
            action['status'] = 'completed' if success else 'failed'
            
            # Call cleanup method if it exists
            try:
                if hasattr(action['handler'], 'cleanup'):
                    action['handler'].cleanup(action['data'])
            except Exception as e:
                logger.error(f"Error during action cleanup {action_id}: {str(e)}")
                
            del self.active_actions[action_id]
            logger.info(f"Completed action {action_id} with status: {'success' if success else 'failed'}")
            
    def cancel_action(self, action_id: str):
        """
        Cancel an executing action.
        
        @param {str} action_id - ID of action to cancel
        """
        if action_id in self.active_actions:
            action = self.active_actions[action_id]
            try:
                if hasattr(action['handler'], 'cancel'):
                    action['handler'].cancel(action['data'])
            except Exception as e:
                logger.error(f"Error cancelling action {action_id}: {str(e)}")
            finally:
                del self.active_actions[action_id]
                logger.info(f"Cancelled action {action_id}")
                
    def get_action_status(self, action_id: str) -> Optional[str]:
        """
        Get the current status of an action.
        
        @param {str} action_id - ID of action to check
        @returns {Optional[str]} - Current status or None if not found
        """
        if action_id in self.active_actions:
            return self.active_actions[action_id]['status']
        return None
        
    def cleanup(self):
        """Clean up all active actions."""
        action_ids = list(self.active_actions.keys())
        for action_id in action_ids:
            self.cancel_action(action_id)
