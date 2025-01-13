"""Integration tests for the action system components."""

import unittest
from unittest.mock import MagicMock, patch
import logging
from datetime import datetime

from ..core.components.decision.actions import (
    ActionGenerator,
    ActionExecutor,
    ActionTemplates,
    ActionValidator,
    ActionOptimizer,
    create_action_system
)
from ..core.components.memory.state_tracker import StateTracker
from ..core.action_controller import ActionController
from ..core.components.decision.action_planner import ActionPlanner

class TestActionSystem(unittest.TestCase):
    """Test suite for action system integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'action': {
                'max_sequence_length': 5,
                'min_duration': 0.1,
                'max_duration': 2.0
            }
        }
        
        # Mock dependencies
        self.state_tracker = MagicMock(spec=StateTracker)
        self.action_controller = MagicMock(spec=ActionController)
        
        # Create action system
        self.action_system = create_action_system(
            self.config,
            self.state_tracker,
            self.action_controller
        )
        
        # Create planner
        self.planner = ActionPlanner(self.config, self.action_system)
        
    def test_action_system_creation(self):
        """Test action system components are properly created and connected."""
        self.assertIsNotNone(self.action_system['executor'])
        self.assertIsNotNone(self.action_system['generator'])
        self.assertIsNotNone(self.action_system['templates'])
        self.assertIsNotNone(self.action_system['validator'])
        self.assertIsNotNone(self.action_system['optimizer'])
        
    def test_template_registration(self):
        """Test action templates are properly registered."""
        templates = self.action_system['templates']
        
        # Check default templates are registered
        self.assertIsNotNone(templates.get_template('movement', 'explore'))
        self.assertIsNotNone(templates.get_template('combat', 'attack'))
        self.assertIsNotNone(templates.get_template('resource', 'gather'))
        
    def test_action_generation(self):
        """Test action generation from strategy."""
        strategy = {
            'objectives': [{
                'type': 'exploration',
                'goal': 'explore',
                'target': {'x': 100, 'y': 100}
            }]
        }
        
        game_state = {
            'player': {'position': {'x': 0, 'y': 0}},
            'environment': {'visibility': 1.0}
        }
        
        constraints = {
            'movement_constraints': {
                'speed': 100,
                'terrain': 'normal'
            }
        }
        
        actions = self.planner.plan_actions(strategy, game_state, constraints)
        
        self.assertTrue(len(actions) > 0)
        self.assertEqual(actions[0]['type'], 'look')
        
    def test_action_validation(self):
        """Test action validation during planning."""
        # Mock validator to reject certain actions
        self.action_system['validator'].validate_action = MagicMock(
            side_effect=lambda a, s: a['type'] != 'invalid'
        )
        
        strategy = {
            'objectives': [{
                'type': 'movement',
                'goal': 'explore'
            }]
        }
        
        # Add invalid action to template
        self.action_system['templates'].add_template(
            'movement',
            'explore',
            [
                {'type': 'invalid', 'key': 'test', 'duration': 0.1},
                {'type': 'look', 'key': 'look_left', 'duration': 0.2}
            ]
        )
        
        actions = self.planner.plan_actions(strategy, {}, {})
        
        # Invalid action should be filtered out
        self.assertTrue(all(a['type'] != 'invalid' for a in actions))
        
    def test_action_optimization(self):
        """Test action sequence optimization."""
        strategy = {
            'objectives': [{
                'type': 'movement',
                'goal': 'explore'
            }]
        }
        
        # Add redundant actions to template
        self.action_system['templates'].add_template(
            'movement',
            'explore',
            [
                {'type': 'look', 'key': 'look_left', 'duration': 0.2},
                {'type': 'look', 'key': 'look_left', 'duration': 0.2},
                {'type': 'movement', 'key': 'forward', 'duration': 0.5}
            ]
        )
        
        actions = self.planner.plan_actions(strategy, {}, {})
        
        # Redundant actions should be merged
        self.assertTrue(len(actions) < 3)
        
    def test_action_execution(self):
        """Test action execution through executor."""
        strategy = {
            'objectives': [{
                'type': 'movement',
                'goal': 'explore'
            }]
        }
        
        # Execute actions
        self.planner.plan_actions(strategy, {}, {}, execute=True)
        
        # Verify action controller was called
        self.action_controller.execute_action.assert_called()
        
    def test_error_handling(self):
        """Test error handling and fallback sequences."""
        # Mock generator to raise error
        self.action_system['generator'].generate_actions = MagicMock(
            side_effect=Exception("Test error")
        )
        
        strategy = {
            'objectives': [{
                'type': 'movement',
                'goal': 'explore'
            }]
        }
        
        # Should return fallback sequence
        actions = self.planner.plan_actions(strategy, {}, {})
        
        self.assertTrue(len(actions) > 0)
        self.assertEqual(actions[0]['type'], 'look')
        
    def test_state_tracking(self):
        """Test state tracking during action execution."""
        strategy = {
            'objectives': [{
                'type': 'movement',
                'goal': 'explore'
            }]
        }
        
        self.planner.plan_actions(strategy, {}, {}, execute=True)
        
        # Verify state tracker was updated
        self.state_tracker.update_state.assert_called_with(
            'action_state',
            unittest.mock.ANY  # Any dict with status info
        )
