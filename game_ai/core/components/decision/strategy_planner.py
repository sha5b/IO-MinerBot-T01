"""Strategy planning component for developing long-term game plans."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import time

class StrategyPlanner:
    """Handles long-term strategic planning."""
    
    def __init__(self, config: Dict[str, Any], objective_manager):
        """
        Initialize strategy planner.
        
        Args:
            config (dict): Configuration settings
            objective_manager: Reference to objective manager
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.objective_manager = objective_manager
        
        # Strategy settings
        self.planning_horizon = config.get('strategy', {}).get('planning_horizon', 60)
        self.max_strategies = config.get('strategy', {}).get('max_strategies', 5)
        self.adaptation_rate = config.get('strategy', {}).get('adaptation_rate', 0.5)
        self.llm_timeout = config.get('strategy', {}).get('llm_timeout', 5.0)  # 5 second timeout
        
        # Default strategies for different situations
        self.default_strategies = {
            'exploration': {
                'objectives': [
                    {'type': 'exploration', 'goal': 'explore_area', 'priority': 'medium'}
                ],
                'focus': 'exploration'
            },
            'survival': {
                'objectives': [
                    {'type': 'survival', 'goal': 'find_resources', 'priority': 'high'}
                ],
                'focus': 'survival'
            },
            'combat': {
                'objectives': [
                    {'type': 'combat', 'goal': 'evade_threat', 'priority': 'high'}
                ],
                'focus': 'combat'
            }
        }
    
    def develop_strategy(self, game_state: Dict[str, Any], 
                        memory: List[Dict[str, Any]], 
                        llm=None) -> Dict[str, Any]:
        """
        Develop strategic plan based on current state and memory.
        
        Args:
            game_state (dict): Current game state
            memory (list): Historical memory data
            llm: Optional LLM interface
            
        Returns:
            dict: Strategic plan
        """
        try:
            # Generate primary strategy using rule-based system
            strategy = self._generate_rule_based_strategy(game_state, memory)
            
            # Optionally enhance with LLM if available
            if llm:
                try:
                    llm_strategy = self._generate_llm_strategy(game_state, memory, llm)
                    if llm_strategy:
                        # Merge LLM insights with rule-based strategy
                        strategy['objectives'].extend(llm_strategy.get('objectives', []))
                except Exception as e:
                    self.logger.error(f"Error generating LLM strategy: {e}")
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error developing strategy: {e}")
            return self._get_fallback_strategy()
    
    def _generate_llm_strategy(self, game_state: Dict[str, Any], 
                             memory: List[Dict[str, Any]], 
                             llm) -> Optional[Dict[str, Any]]:
        """Generate strategy using LLM."""
        try:
            # Prepare prompt
            prompt = self._create_strategy_prompt(game_state, memory)
            
            # Set timeout for LLM response
            start_time = time.time()
            response = None
            
            # Try to get LLM response with timeout
            while time.time() - start_time < self.llm_timeout:
                try:
                    response = llm.generate(prompt)
                    break
                except Exception as e:
                    self.logger.warning(f"LLM attempt failed: {e}")
                    time.sleep(0.1)
            
            if not response:
                self.logger.warning("LLM response timed out")
                return None
            
            # Parse response into strategy
            strategy = self._parse_llm_response(response)
            if not strategy:
                return None
            
            # Validate and format strategy
            return self._format_strategy(strategy)
            
        except Exception as e:
            self.logger.error(f"Error in LLM strategy generation: {e}")
            return None
    
    def _create_strategy_prompt(self, game_state: Dict[str, Any], 
                              memory: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM strategy generation."""
        # Extract relevant state information
        player_status = game_state.get('player', {})
        resources = game_state.get('inventory', {})
        environment = game_state.get('environment', {})
        
        # Format prompt
        prompt = """You are an AI playing a game. Analyze the situation and suggest strategic objectives.

Current Situation:
Player Status: {player}
Resources: {resources}
Environment: {environment}

What strategic objectives should be pursued? Provide specific, actionable goals.""".format(
            player=player_status,
            resources=resources,
            environment=environment
        )
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into strategy structure."""
        try:
            # Try parsing as JSON first
            try:
                strategy = json.loads(response)
                if isinstance(strategy, dict):
                    return strategy
            except json.JSONDecodeError:
                pass
            
            # Parse text response
            objectives = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    goal, details = line.split(':', 1)
                    objectives.append({
                        'type': 'generated',
                        'goal': goal.strip().lower(),
                        'details': details.strip(),
                        'priority': 'medium'
                    })
            
            if objectives:
                return {
                    'objectives': objectives,
                    'focus': objectives[0]['goal'],
                    'source': 'llm'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _generate_rule_based_strategy(self, game_state: Dict[str, Any], 
                                    memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate strategy using rule-based system."""
        objectives = []
        focus = 'exploration'  # Default focus
        
        # Check health status
        health = game_state.get('player', {}).get('health', 100)
        if health < 30:
            objectives.append({
                'type': 'survival',
                'goal': 'heal',
                'priority': 'critical',
                'details': 'Find food or healing items'
            })
            focus = 'survival'
        elif health < 70:
            objectives.append({
                'type': 'survival',
                'goal': 'heal',
                'priority': 'high',
                'details': 'Restore health when convenient'
            })
        
        # Check threats
        threats = game_state.get('environment', {}).get('threats', [])
        if threats:
            immediate_threats = [t for t in threats if t.get('distance', float('inf')) < 20]
            if immediate_threats:
                objectives.append({
                    'type': 'combat',
                    'goal': 'evade_threat',
                    'priority': 'critical',
                    'details': f'Evade {len(immediate_threats)} immediate threats'
                })
                focus = 'combat'
        
        # Check resources
        inventory = game_state.get('inventory', {}).get('items', {})
        if not inventory:
            objectives.append({
                'type': 'resource',
                'goal': 'gather_basic_resources',
                'priority': 'high',
                'details': 'Gather wood and stone'
            })
            focus = 'resource'
        
        # Check opportunities
        opportunities = game_state.get('environment', {}).get('opportunities', [])
        if opportunities:
            nearby_resources = [o for o in opportunities if o.get('distance', float('inf')) < 30]
            if nearby_resources:
                objectives.append({
                    'type': 'resource',
                    'goal': 'gather_resources',
                    'priority': 'medium',
                    'details': f'Gather {len(nearby_resources)} nearby resources'
                })
        
        # Add exploration if no critical objectives
        if not objectives or all(o['priority'] not in ['critical', 'high'] for o in objectives):
            objectives.append({
                'type': 'exploration',
                'goal': 'explore_area',
                'priority': 'medium',
                'details': 'Explore surroundings for resources and opportunities'
            })
        
        return {
            'objectives': objectives,
            'focus': focus,
            'timestamp': datetime.now().isoformat(),
            'source': 'rule_based'
        }
    
    def _needs_survival_focus(self, game_state: Dict[str, Any]) -> bool:
        """Check if survival needs attention."""
        health = game_state.get('player', {}).get('health', 100)
        resources = game_state.get('inventory', {}).get('items', {})
        
        return health < 50 or not resources
    
    def _needs_combat_focus(self, game_state: Dict[str, Any]) -> bool:
        """Check if combat needs attention."""
        threats = game_state.get('environment', {}).get('threats', [])
        return bool(threats)
    
    def _format_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Format and validate strategy structure."""
        formatted = {
            'objectives': [],
            'focus': strategy.get('focus', 'survival'),
            'timestamp': datetime.now().isoformat(),
            'source': strategy.get('source', 'unknown')
        }
        
        # Format objectives
        for obj in strategy.get('objectives', []):
            formatted['objectives'].append({
                'type': obj.get('type', 'generated'),
                'goal': obj.get('goal', 'survive'),
                'priority': obj.get('priority', 'medium'),
                'details': obj.get('details', '')
            })
        
        return formatted
    
    def _get_fallback_strategy(self) -> Dict[str, Any]:
        """Get basic fallback strategy."""
        return {
            'objectives': [
                {
                    'type': 'survival',
                    'goal': 'survive',
                    'priority': 'high',
                    'details': 'Basic survival objective'
                }
            ],
            'focus': 'survival',
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
