"""Decision engine for AI strategic planning and action selection."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np

from .components.decision import (
    ObjectiveManager,
    StrategyPlanner,
    SituationAnalyzer,
    ActionPlanner,
    ReactiveController
)

class DecisionEngine:
    """Handles all decision making and action planning."""
    
    def __init__(self, config_path: Optional[Path] = None, ollama_interface=None):
        """
        Initialize the decision engine.
        
        Args:
            config_path (Path, optional): Path to configuration file
            ollama_interface: Interface to LLM service
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.ollama = ollama_interface
        
        # Initialize components with structured config
        self.objective_manager = ObjectiveManager(self.config)
        self.strategy_planner = StrategyPlanner(self.config, self.objective_manager)
        self.situation_analyzer = SituationAnalyzer(self.config)
        
        # Create action config with defaults if needed
        action_config = self.config.get('action', {})
        if not action_config:
            self.logger.warning("No action config found, using defaults")
            action_config = {
                'max_sequence_length': 20,
                'min_duration': 0.1,
                'max_duration': 5.0,
                'timeout_multiplier': 2.0
            }
            self.config['action'] = action_config
            
        self.action_planner = ActionPlanner(self.config)
        self.reactive_controller = ReactiveController(self.config)
        
        # State tracking
        self.current_strategy: Dict[str, Any] = {}
        self.action_queue: List[Dict[str, Any]] = []
        self.last_state: Dict[str, Any] = {}
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load decision engine configuration."""
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
                    
                    # Extract and validate action config
                    action_config = full_config.get('action', {})
                    required_fields = [
                        'max_sequence_length',
                        'min_duration',
                        'max_duration',
                        'timeout_multiplier'
                    ]
                    
                    # Check required fields
                    for field in required_fields:
                        if field not in action_config:
                            self.logger.warning(f"Missing required action setting: {field}")
                    
                    # Structure config for components
                    return {
                        'action': action_config,
                        'controls': full_config.get('controls', {}),
                        'memory': full_config.get('memory', {}),
                        'vision': full_config.get('vision', {})
                    }
            return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def strategic_planning(self, game_state: Dict[str, Any], 
                         memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Develop long-term strategic goals based on game state and memory.
        
        Args:
            game_state (dict): Current game state
            memory (list): Historical memory data
            
        Returns:
            dict: Strategic plan
        """
        try:
            # First check for immediate reactions
            sanitized_state = self._sanitize_state(game_state)
            reaction = self.reactive_controller.process_state_changes(sanitized_state)
            if reaction:
                self.logger.info("Generated reactive response")
                return {
                    'type': 'reactive',
                    'strategy': reaction,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze current situation
            situation = self.situation_analyzer.analyze_situation(sanitized_state)
            
            # Check if current strategy is still valid
            if self.current_strategy:
                if self._should_update_strategy(situation):
                    self.logger.info("Current strategy needs update")
                else:
                    return self.current_strategy
            
            # Develop new strategy
            strategy = self.strategy_planner.develop_strategy(
                sanitized_state,
                memory,
                self.ollama
            )
            
            # Update objective tracking
            self._update_objective_tracking(strategy, situation)
            
            self.current_strategy = strategy
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error in strategic planning: {e}")
            return self._get_fallback_strategy()
    
    def tactical_planning(self, strategy: Dict[str, Any], 
                         game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Develop medium-term tactical plans to achieve strategic goals.
        
        Args:
            strategy (dict): Current strategy
            game_state (dict): Current game state
            
        Returns:
            list: Tactical action sequence
        """
        try:
            # Get constraints
            constraints = self._identify_constraints(game_state)
            
            # Generate action sequence
            actions = self.action_planner.plan_actions(
                strategy,
                game_state,
                constraints
            )
            
            # Store in action queue
            self.action_queue = actions
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error in tactical planning: {e}")
            return []
    
    def reactive_response(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate immediate reactive actions based on sudden changes.
        
        Args:
            game_state (dict): Current game state
            
        Returns:
            dict: Reactive response or None
        """
        return self.reactive_controller.process_state_changes(
            self._sanitize_state(game_state)
        )
    
    def evaluate_situation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current game situation.
        
        Args:
            game_state (dict): Current game state
            
        Returns:
            dict: Situation analysis
        """
        return self.situation_analyzer.analyze_situation(game_state)
    
    def prioritize_actions(self, actions: List[Dict[str, Any]], 
                         game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prioritize and order actions based on current state.
        
        Args:
            actions (list): List of actions to prioritize
            game_state (dict): Current game state
            
        Returns:
            list: Prioritized actions
        """
        try:
            # Get current situation
            situation = self.situation_analyzer.analyze_situation(game_state)
            
            # Prioritize objectives first
            objectives = self.objective_manager.prioritize_objectives(
                self.current_strategy.get('objectives', []),
                situation
            )
            
            # Update strategy with prioritized objectives
            self.current_strategy['objectives'] = objectives
            
            # Generate and prioritize actions
            return self.action_planner.plan_actions(
                self.current_strategy,
                game_state,
                self._identify_constraints(game_state)
            )
            
        except Exception as e:
            self.logger.error(f"Error prioritizing actions: {e}")
            return actions
    
    def _identify_constraints(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify current constraints on actions."""
        return {
            'health_limit': game_state.get('player', {}).get('health', 100),
            'resource_limits': game_state.get('inventory', {}).get('capacity', 100),
            'movement_constraints': self._get_movement_constraints(game_state)
        }
    
    def _get_movement_constraints(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get current movement constraints."""
        return {
            'terrain': game_state.get('environment', {}).get('terrain_type', 'normal'),
            'speed': game_state.get('player', {}).get('movement_speed', 100),
            'obstacles': game_state.get('environment', {}).get('obstacles', [])
        }
    
    def _get_fallback_strategy(self) -> Dict[str, Any]:
        """Get basic fallback strategy."""
        return {
            'objectives': [
                self.objective_manager.create_objective(
                    objective_type='survival',
                    goal='survive',
                    priority='high'
                )
            ],
            'constraints': {},
            'focus': 'survival',
            'contingencies': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def _should_update_strategy(self, situation: Dict[str, Any]) -> bool:
        """
        Determine if current strategy needs updating.
        
        Args:
            situation (dict): Current situation analysis
            
        Returns:
            bool: Whether strategy should be updated
        """
        if not self.current_strategy:
            return True
            
        try:
            # Check if current objectives are still valid
            current_objectives = self.current_strategy.get('objectives', [])
            if not current_objectives:
                return True
            
            # Check if highest priority objective is still relevant
            top_objective = current_objectives[0]
            obj_type = top_objective.get('type')
            
            # Check critical conditions
            if obj_type == 'survival':
                health = situation.get('player_status', {}).get('health', {}).get('value', 100)
                if health < 30:  # Critical health takes precedence
                    return True
                    
            elif obj_type == 'combat':
                threats = situation.get('threats', [])
                immediate_threats = [t for t in threats if t.get('distance', float('inf')) < 20]
                if immediate_threats:  # Immediate threats take precedence
                    return True
                    
            elif obj_type == 'resource':
                if not situation.get('resources', {}).get('current', {}):
                    return True  # No resources requires strategy update
                    
            # Check if objective completion criteria met
            if self._is_objective_complete(top_objective, situation):
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking strategy update: {e}")
            return True
    
    def _update_objective_tracking(self, strategy: Dict[str, Any], 
                                 situation: Dict[str, Any]) -> None:
        """
        Update objective tracking and completion status.
        
        Args:
            strategy (dict): Current strategy
            situation (dict): Current situation analysis
        """
        try:
            # Update existing objectives
            for obj in self.objective_manager.get_active_objectives():
                if self._is_objective_complete(obj, situation):
                    self.objective_manager.update_objective_status(
                        obj['id'],
                        'completed',
                        'Objective completion criteria met'
                    )
            
            # Add new objectives
            for obj in strategy.get('objectives', []):
                if obj not in self.objective_manager.get_active_objectives():
                    self.objective_manager.create_objective(
                        obj['type'],
                        obj['goal'],
                        obj.get('priority', 'medium'),
                        obj.get('details', '')
                    )
                    
        except Exception as e:
            self.logger.error(f"Error updating objective tracking: {e}")
    
    def _is_objective_complete(self, objective: Dict[str, Any], 
                             situation: Dict[str, Any]) -> bool:
        """
        Check if objective completion criteria are met.
        
        Args:
            objective (dict): Objective to check
            situation (dict): Current situation analysis
            
        Returns:
            bool: Whether objective is complete
        """
        try:
            obj_type = objective.get('type')
            obj_goal = objective.get('goal')
            
            if obj_type == 'survival' and obj_goal == 'heal':
                health = situation.get('player_status', {}).get('health', {}).get('value', 100)
                return health > 70  # Health restored
                
            elif obj_type == 'combat' and obj_goal == 'evade_threat':
                threats = situation.get('threats', [])
                return not any(t.get('distance', float('inf')) < 30 for t in threats)
                
            elif obj_type == 'resource' and obj_goal == 'gather_basic_resources':
                resources = situation.get('resources', {}).get('current', {})
                return bool(resources)  # Basic resources gathered
                
            elif obj_type == 'exploration' and obj_goal == 'explore_area':
                explored = situation.get('environment', {}).get('explored_ratio', 0)
                return explored > 0.7  # Area sufficiently explored
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking objective completion: {e}")
            return False
    
    def _get_current_objective(self) -> Optional[Dict[str, Any]]:
        """Get current highest priority objective."""
        active_objectives = self.objective_manager.get_active_objectives()
        if not active_objectives:
            return None
        return active_objectives[0]  # Objectives are kept sorted by priority
    
    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-JSON serializable objects from state."""
        sanitized = {}
        for key, value in state.items():
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_state(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        self._sanitize_state(item) if isinstance(item, dict) else item
                        for item in value
                        if not hasattr(item, 'dtype')  # Skip numpy arrays
                    ]
                else:
                    sanitized[key] = value
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                sanitized[key] = value.tolist()
        return sanitized
    
    def add_objective_type(self, type_name: str, base_priority: float) -> None:
        """
        Add new objective type.
        
        Args:
            type_name (str): Name of objective type
            base_priority (float): Base priority score
        """
        self.objective_manager.add_objective_type(type_name, base_priority)
    
    def add_action_template(self, category: str, action_type: str, 
                          template: Dict[str, Any]) -> None:
        """
        Add new action template.
        
        Args:
            category (str): Action category
            action_type (str): Type of action
            template (dict): Action template
        """
        self.action_planner.add_action_template(category, action_type, template)
    
    def add_response_template(self, name: str, template: Dict[str, Any]) -> None:
        """
        Add new reactive response template.
        
        Args:
            name (str): Template name
            template (dict): Response template
        """
        self.reactive_controller.add_response_template(name, template)
    
    def update_thresholds(self, component: str, 
                         new_thresholds: Dict[str, Any]) -> None:
        """
        Update component thresholds.
        
        Args:
            component (str): Component to update ('reactive', 'situation', etc.)
            new_thresholds (dict): New threshold values
        """
        if component == 'reactive':
            self.reactive_controller.update_thresholds(new_thresholds)
        elif component == 'situation':
            self.situation_analyzer.update_thresholds(new_thresholds)
