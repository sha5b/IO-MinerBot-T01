"""Decision engine for AI strategic planning and action selection."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

class DecisionEngine:
    """Handles all decision making and action planning."""
    
    def __init__(self, config_path: Optional[Path] = None, ollama_interface=None):
        """Initialize the decision engine."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.current_strategy: Dict[str, Any] = {}
        self.action_queue: List[Dict[str, Any]] = []
        self.last_state: Dict[str, Any] = {}
        self.ollama = ollama_interface
        self.game_rules = self._load_game_rules()
        self.control_maps = self._load_control_maps()
        self.game_type = "minecraft"  # Default to minecraft controls
        
    def _load_game_rules(self) -> Dict[str, Any]:
        """Load game rules configuration."""
        try:
            with open('game_ai/config/game_rules.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading game rules: {e}")
            return {}
            
    def _load_control_maps(self) -> Dict[str, Any]:
        """Load control mappings configuration."""
        try:
            with open('game_ai/config/control_maps.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading control maps: {e}")
            return {}
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load decision engine configuration."""
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def strategic_planning(self, game_state: Dict[str, Any], memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Develop long-term strategic goals based on game state and memory."""
        try:
            # Extract environment data from game state
            environment_data = game_state.get('analysis', {}).get('environment', {})
            if not environment_data:
                self.logger.warning("No environment data in game state")
                environment_data = {
                    'terrain_type': 'unknown',
                    'terrain_analysis': {},
                    'threats': self.game_rules.get('behaviors', {}).get('combat', {}).get('hostile_mobs', []),
                    'resources': self.game_rules.get('behaviors', {}).get('resource_gathering', {}).get('initial_resources', []),
                    'passive_mobs': self.game_rules.get('behaviors', {}).get('combat', {}).get('passive_mobs', [])
                }
            
            # Extract player data from game state
            player_data = game_state.get('player', {})
            if not player_data:
                self.logger.warning("No player data in game state")
                player_data = {
                    'health': self.game_rules.get('priorities', {}).get('health', 100),
                    'status_effects': [],
                    'equipment': {
                        'tools': self.game_rules.get('behaviors', {}).get('crafting', {}).get('basic_tools', []),
                        'items': self.game_rules.get('behaviors', {}).get('crafting', {}).get('essential_items', [])
                    }
                }
                game_state['player'] = player_data
            
            # Analyze current situation
            situation = {
                'player_status': self._evaluate_player_status(game_state),
                'threats': environment_data.get('threats', []),
                'opportunities': self._identify_opportunities(environment_data),
                'resources': self._evaluate_resources(game_state)
            }
            
            # Start with basic survival objectives from game rules
            objectives = []
            for primary_objective in self.game_rules.get('objectives', {}).get('primary', []):
                objectives.append({
                    'type': 'minecraft',
                    'priority': 'high',
                    'goal': primary_objective,
                    'details': 'Initial survival task'
                })
            
            if self.ollama:
                # Use LLM for strategic planning
                system_prompt = """You are an AI playing Minecraft. Your goal is to survive and thrive.
Current objectives are: get wood, make tools, find food, build shelter, and mine resources.
Analyze the game state and decide what to do next. Be specific and actionable."""
                
                game_state_str = f"""
Player status: {situation['player_status']}
Resources: {situation['resources']}
Environment: {environment_data}
Current objective: {self._get_current_objective()}
"""
                
                try:
                    response = self.ollama.generate(system_prompt + "\n\nGame State:\n" + game_state_str)
                    self.logger.info(f"LLM Strategic Decision: {response}")
                    
                    # Add LLM suggestions to objectives
                    llm_objectives = self._parse_llm_objectives(response)
                    objectives.extend(llm_objectives)
                except Exception as e:
                    self.logger.error(f"Error in LLM planning: {e}")
                    # Continue with basic objectives if LLM fails
            
            # Create strategic plan
            strategy = {
                'objectives': objectives,
                'priorities': self._prioritize_objectives(objectives, situation),
                'constraints': self._identify_constraints(game_state),
                'timestamp': datetime.now().isoformat()
            }
            
            self.current_strategy = strategy
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error in strategic planning: {e}")
            # Return a basic fallback strategy
            return {
                'objectives': [{
                    'type': 'minecraft',
                    'priority': 'high',
                    'goal': 'get_wood',
                    'details': 'Basic survival task'
                }],
                'priorities': [],
                'constraints': {},
                'timestamp': datetime.now().isoformat()
            }
            
    def _get_current_objective(self) -> str:
        """Get the current primary objective based on game rules."""
        try:
            primary_objectives = self.game_rules.get('objectives', {}).get('primary', [])
            if not self.current_strategy.get('objectives'):
                return primary_objectives[0] if primary_objectives else "survive"
            return self.current_strategy['objectives'][0]['goal']
        except Exception:
            return "survive"
            
    def _parse_llm_objectives(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured objectives."""
        objectives = []
        try:
            # Basic parsing of LLM response
            response_lines = llm_response.strip().split('\n')
            for line in response_lines:
                if ':' in line:
                    action, details = line.split(':', 1)
                    objectives.append({
                        'type': 'minecraft',
                        'priority': 'high',
                        'goal': action.strip().lower(),
                        'details': details.strip()
                    })
            
            if not objectives:
                # Fallback if parsing fails
                objectives.append({
                    'type': 'minecraft',
                    'priority': 'high',
                    'goal': llm_response.strip().lower(),
                    'details': ''
                })
        except Exception as e:
            self.logger.error(f"Error parsing LLM objectives: {e}")
            objectives.append({
                'type': 'minecraft',
                'priority': 'high',
                'goal': 'survive',
                'details': ''
            })
        
        return objectives
    
    def tactical_planning(self, strategy: Dict[str, Any], game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Develop medium-term tactical plans to achieve strategic goals."""
        try:
            if not strategy or not strategy.get('objectives'):
                self.logger.warning("No valid strategy or objectives")
                return []
                
            # Get current objective and environment
            current_objective = self._get_current_objective()
            environment = game_state.get('analysis', {}).get('environment', {})
            
            # Get nearest resource based on objective
            nearest_resource = None
            if environment.get('resources'):
                for resource in environment['resources']:
                    if (current_objective == "get_wood" and resource['type'] == 'tree') or \
                       (current_objective == "find_food" and resource['type'] in ['cow', 'pig', 'sheep']):
                        if not nearest_resource or resource['distance'] < nearest_resource['distance']:
                            nearest_resource = resource
            
            basic_actions = []
            
            # If we found a relevant resource, move towards it
            if nearest_resource:
                # Calculate direction to resource
                frame_width = game_state.get('frame_shape', (0, 0, 0))[1]
                resource_x = nearest_resource['position'][0]
                center_x = frame_width / 2
                
                # Look and move towards resource
                if abs(resource_x - center_x) > 50:  # If not centered
                    if resource_x < center_x:
                        basic_actions.extend([
                            {
                                'type': 'look',
                                'key': 'look_left',
                                'duration': 0.2
                            },
                            {
                                'type': 'movement',
                                'key': 'left',
                                'duration': 0.3
                            }
                        ])
                    else:
                        basic_actions.extend([
                            {
                                'type': 'look',
                                'key': 'look_right',
                                'duration': 0.2
                            },
                            {
                                'type': 'movement',
                                'key': 'right',
                                'duration': 0.3
                            }
                        ])
                
                # Look around, move forward and interact
                basic_actions.extend([
                    {
                        'type': 'look',
                        'key': 'look_up',
                        'duration': 0.2
                    },
                    {
                        'type': 'look',
                        'key': 'look_down',
                        'duration': 0.2
                    },
                    {
                        'type': 'movement',
                        'key': 'forward',
                        'duration': 1.0
                    },
                    {
                        'type': 'action',
                        'key': 'mouse1',
                        'duration': 2.0
                    }
                ])
            else:
                # Exploration pattern with looking around and jumping if no resource found
                basic_actions = [
                {
                    'type': 'look',
                    'key': 'look_left',
                    'duration': 0.3
                },
                {
                    'type': 'movement',
                    'key': 'forward',
                    'duration': 0.5
                },
                {
                    'type': 'movement',
                    'key': 'jump',
                    'duration': 0.2
                },
                {
                    'type': 'look',
                    'key': 'look_right',
                    'duration': 0.3
                },
                {
                    'type': 'movement',
                    'key': 'right',
                    'duration': 0.3
                },
                {
                    'type': 'look',
                    'key': 'look_up',
                    'duration': 0.2
                },
                {
                    'type': 'movement',
                    'key': 'forward',
                    'duration': 0.5
                },
                {
                    'type': 'movement',
                    'key': 'jump',
                    'duration': 0.2
                },
                {
                    'type': 'look',
                    'key': 'look_down',
                    'duration': 0.2
                }
            ]
            
            if self.ollama:
                # Use LLM for additional tactical planning
                system_prompt = """You are controlling a Minecraft character.
Available actions:
- Movement: move forward/backward/left/right, jump (space)
- Looking: look left/right/up/down to change view direction
- Actions: break blocks (left click), place blocks (right click)

What specific actions should be taken to achieve the current objective?
Remember to look around to find resources and navigate. Use natural language like:
'look left', 'look up', 'move forward', 'break block', 'jump', etc."""
                
                situation_str = f"""
Current objective: {current_objective}
Player status: {game_state.get('player', {})}
Environment: {game_state.get('environment', {})}
"""
                
                response = self.ollama.generate(system_prompt + "\n\nSituation:\n" + situation_str)
                self.logger.info(f"LLM Tactical Decision: {response}")
                
                # Convert LLM response to actions
                llm_actions = self._convert_llm_to_actions(response)
                basic_actions.extend(llm_actions)
            
            self.logger.info(f"Generated actions: {basic_actions}")
            return basic_actions
            
        except Exception as e:
            self.logger.error(f"Error in tactical planning: {e}")
            return []
            
    def _convert_llm_to_actions(self, llm_response: str) -> List[Dict[str, Any]]:
        """Convert LLM response to concrete game actions."""
        actions = []
        try:
            # Try to parse as JSON first
            try:
                action_data = json.loads(llm_response)
                if isinstance(action_data, dict):
                    # Single action in JSON format
                    if self._validate_action_format(action_data):
                        actions.append(action_data)
                elif isinstance(action_data, list):
                    # Multiple actions in JSON format
                    for action in action_data:
                        if self._validate_action_format(action):
                            actions.append(action)
            except json.JSONDecodeError:
                # Fallback to text parsing if not JSON
                self.logger.warning(f"Failed to parse LLM response as JSON: {llm_response}")
                response_lines = llm_response.strip().split('\n')
                for line in response_lines:
                    line = line.strip().lower()
                    action = self._parse_action_line(line)
                    if action:
                        actions.append(action)
        
        except Exception as e:
            self.logger.error(f"Error converting LLM response to actions: {e}")
            # Fallback to basic survival action
            actions.append({'type': 'movement', 'key': 'w', 'duration': 1.0})
        
        return actions

    def _validate_action_format(self, action: Dict[str, Any]) -> bool:
        """Validate action format matches expected schema."""
        try:
            # Required fields
            if not all(k in action for k in ['type', 'key', 'duration']):
                self.logger.warning(f"Action missing required fields: {action}")
                return False
            
            # Type validation
            valid_types = ['movement', 'action', 'look']
            if action['type'] not in valid_types:
                self.logger.warning(f"Invalid action type: {action['type']}. Must be one of: {valid_types}")
                return False
            
            # Key validation based on type
            if action['type'] == 'movement':
                game_controls = self.control_maps.get('games', {}).get(self.game_type, {})
                valid_movement_keys = game_controls.get('keyboard', {}).get('movement', {}).keys()
                if not valid_movement_keys:
                    valid_movement_keys = self.control_maps.get('games', {}).get('default', {}).get('keyboard', {}).get('movement', {}).keys()
                if action['key'] not in valid_movement_keys:
                    self.logger.warning(f"Invalid movement key: {action['key']}. Must be one of: {valid_movement_keys}")
                    return False
            elif action['type'] == 'action':
                valid_action_keys = ['mouse1', 'mouse2']
                if action['key'] not in valid_action_keys:
                    self.logger.warning(f"Invalid action key: {action['key']}. Must be one of: {valid_action_keys}")
                    return False
            elif action['type'] == 'look':
                valid_look_keys = ['look_left', 'look_right', 'look_up', 'look_down']
                if action['key'] not in valid_look_keys:
                    self.logger.warning(f"Invalid look key: {action['key']}. Must be one of: {valid_look_keys}")
                    return False
            
            # Duration validation
            if not isinstance(action['duration'], (int, float)) or action['duration'] <= 0:
                self.logger.warning(f"Invalid duration: {action['duration']}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating action format: {e}")
            return False

    def _parse_action_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line of text into an action if possible."""
        try:
            # Look mappings
            if 'look' in line:
                if 'left' in line:
                    return {'type': 'look', 'key': 'look_left', 'duration': 0.2}
                elif 'right' in line:
                    return {'type': 'look', 'key': 'look_right', 'duration': 0.2}
                elif 'up' in line:
                    return {'type': 'look', 'key': 'look_up', 'duration': 0.2}
                elif 'down' in line:
                    return {'type': 'look', 'key': 'look_down', 'duration': 0.2}
            
            # Movement mappings
            if any(move in line for move in ['walk', 'move', 'go']):
                if 'forward' in line:
                    return {'type': 'movement', 'key': 'forward', 'duration': 1.0}
                elif 'back' in line:
                    return {'type': 'movement', 'key': 'backward', 'duration': 1.0}
                elif 'left' in line and 'look' not in line:  # Avoid conflict with look_left
                    return {'type': 'movement', 'key': 'left', 'duration': 1.0}
                elif 'right' in line and 'look' not in line:  # Avoid conflict with look_right
                    return {'type': 'movement', 'key': 'right', 'duration': 1.0}
            
            # Action mappings
            if 'break' in line or 'mine' in line:
                return {'type': 'action', 'key': 'mouse1', 'duration': 2.0}
            elif 'place' in line:
                return {'type': 'action', 'key': 'mouse2', 'duration': 0.5}
            elif 'jump' in line:
                return {'type': 'movement', 'key': 'jump', 'duration': 0.5}
            
            return None
        except Exception as e:
            self.logger.error(f"Error parsing action line: {e}")
            return None
    
    def reactive_response(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate immediate reactive actions based on sudden changes."""
        try:
            # Detect significant changes
            changes = self._detect_state_changes(game_state)
            
            # Check for threats or opportunities
            if self._requires_immediate_response(changes):
                # Generate reactive action
                action = self._generate_reactive_action(changes, game_state)
                return action
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in reactive response: {e}")
            return None
    
    def evaluate_situation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current game situation."""
        try:
            return {
                'player_status': self._evaluate_player_status(game_state),
                'threats': self._identify_threats(game_state),
                'opportunities': self._identify_opportunities(game_state),
                'resources': self._evaluate_resources(game_state)
            }
        except Exception as e:
            self.logger.error(f"Error evaluating situation: {e}")
            return {}
    
    def prioritize_actions(self, actions: List[Dict[str, Any]], game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize and order actions based on current state."""
        try:
            # Score each action
            scored_actions = [
                (action, self._calculate_action_priority(action, game_state))
                for action in actions
            ]
            
            # Sort by priority score
            sorted_actions = [
                action for action, _ in sorted(
                    scored_actions,
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            
            return sorted_actions
            
        except Exception as e:
            self.logger.error(f"Error prioritizing actions: {e}")
            return actions
    
    def _determine_objectives(self, situation: Dict[str, Any], memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine appropriate objectives based on situation."""
        objectives = []
        
        # Survival objectives
        if situation['player_status'].get('health', 100) < 50:
            objectives.append({
                'type': 'survival',
                'priority': 'high',
                'goal': 'restore_health'
            })
            
        # Resource objectives
        if situation['resources'].get('low_resources', []):
            objectives.append({
                'type': 'gathering',
                'priority': 'medium',
                'goal': 'gather_resources',
                'targets': situation['resources']['low_resources']
            })
            
        # Strategic objectives
        if situation['opportunities']:
            objectives.append({
                'type': 'strategic',
                'priority': 'medium',
                'goal': 'exploit_opportunity',
                'targets': situation['opportunities']
            })
            
        return objectives
    
    def _prioritize_objectives(self, objectives: List[Dict[str, Any]], situation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize objectives based on situation."""
        priority_scores = {
            'survival': 100,
            'threat': 90,
            'gathering': 70,
            'strategic': 50
        }
        
        # Score and sort objectives
        scored_objectives = [
            (obj, priority_scores.get(obj['type'], 0))
            for obj in objectives
        ]
        
        return [obj for obj, _ in sorted(scored_objectives, key=lambda x: x[1], reverse=True)]
    
    def _identify_constraints(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify current constraints on actions."""
        return {
            'health_limit': game_state.get('player', {}).get('health', 100),
            'resource_limits': game_state.get('inventory', {}).get('capacity', 100),
            'movement_constraints': self._check_movement_constraints(game_state)
        }
    
    def _plan_objective_actions(
        self,
        objective: Dict[str, Any],
        game_state: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan sequence of actions to achieve an objective."""
        actions = []
        
        if objective['type'] == 'survival':
            actions.extend(self._plan_survival_actions(objective, game_state))
        elif objective['type'] == 'gathering':
            actions.extend(self._plan_gathering_actions(objective, game_state, constraints))
        elif objective['type'] == 'strategic':
            actions.extend(self._plan_strategic_actions(objective, game_state))
            
        return actions
    
    def _optimize_action_sequence(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize sequence of actions for efficiency."""
        # Implement action sequence optimization
        return actions
    
    def _detect_state_changes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant changes from last state."""
        changes = {}
        
        if self.last_state:
            # Compare health
            old_health = self.last_state.get('player', {}).get('health', 100)
            new_health = game_state.get('player', {}).get('health', 100)
            if abs(new_health - old_health) > 10:
                changes['health_change'] = new_health - old_health
                
            # Compare threats
            old_threats = set(self.last_state.get('threats', []))
            new_threats = set(game_state.get('threats', []))
            changes['new_threats'] = list(new_threats - old_threats)
            changes['removed_threats'] = list(old_threats - new_threats)
            
        self.last_state = game_state
        return changes
    
    def _requires_immediate_response(self, changes: Dict[str, Any]) -> bool:
        """Determine if changes require immediate response."""
        return (
            changes.get('health_change', 0) < -20 or
            changes.get('new_threats', []) or
            changes.get('immediate_danger', False)
        )
    
    def _generate_reactive_action(self, changes: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate reactive action."""
        if changes.get('health_change', 0) < -20:
            return {
                'type': 'heal',
                'priority': 'immediate',
                'target': 'self'
            }
        elif changes.get('new_threats', []):
            return {
                'type': 'evade',
                'priority': 'immediate',
                'target': changes['new_threats'][0]
            }
        return {}
    
    def _calculate_action_priority(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> float:
        """Calculate priority score for an action."""
        base_priority = {
            'immediate': 100.0,
            'high': 80.0,
            'medium': 60.0,
            'low': 40.0
        }.get(action.get('priority', 'low'), 20.0)
        
        # Apply situational modifiers
        modifiers = self._calculate_priority_modifiers(action, game_state)
        
        return base_priority * modifiers
    
    def _calculate_priority_modifiers(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> float:
        """Calculate situational priority modifiers."""
        modifier = 1.0
        
        # Health-based modifier
        health = game_state.get('player', {}).get('health', 100)
        if health < 50 and action['type'] == 'heal':
            modifier *= 1.5
            
        # Threat-based modifier
        if game_state.get('threats', []) and action['type'] in ['evade', 'combat']:
            modifier *= 1.3
            
        return modifier
    
    def _check_movement_constraints(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check current movement constraints."""
        return {
            'terrain_restrictions': game_state.get('environment', {}).get('terrain_type', 'normal'),
            'movement_speed': game_state.get('player', {}).get('movement_speed', 100),
            'obstacles': game_state.get('environment', {}).get('obstacles', [])
        }
    
    def _evaluate_player_status(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current player status."""
        player_data = game_state.get('player', {})
        return {
            'health': player_data.get('health', self.game_rules.get('priorities', {}).get('health', 100)),
            'status_effects': player_data.get('status', []),
            'equipment': player_data.get('equipment', {
                'tools': self.game_rules.get('behaviors', {}).get('crafting', {}).get('basic_tools', []),
                'items': self.game_rules.get('behaviors', {}).get('crafting', {}).get('essential_items', [])
            })
        }
    
    def _identify_threats(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current threats."""
        return game_state.get('environment', {}).get('threats', [])
    
    def _identify_opportunities(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current opportunities."""
        return game_state.get('environment', {}).get('opportunities', [])
    
    def _evaluate_resources(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current resource status."""
        inventory = game_state.get('player', {}).get('inventory', {})
        initial_resources = self.game_rules.get('behaviors', {}).get('resource_gathering', {}).get('initial_resources', [])
        return {
            'current_resources': inventory.get('items', {resource: 0 for resource in initial_resources}),
            'capacity': inventory.get('capacity', self.game_rules.get('constraints', {}).get('inventory_slots', 36)),
            'low_resources': initial_resources
        }
