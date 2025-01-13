"""Objective management component for handling AI goals and priorities."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class ObjectiveManager:
    """Handles creation, tracking, and prioritization of AI objectives."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize objective manager.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Objective settings
        self.max_objectives = config.get('objectives', {}).get('max_objectives', 10)
        self.min_priority = config.get('objectives', {}).get('min_priority', 0.0)
        self.max_priority = config.get('objectives', {}).get('max_priority', 1.0)
        
        # Objective type definitions
        self.objective_types = {
            'survival': {
                'base_priority': 1.0,
                'description': 'Basic survival objectives',
                'goals': ['find_food', 'find_shelter', 'heal']
            },
            'exploration': {
                'base_priority': 0.6,
                'description': 'World exploration objectives',
                'goals': ['explore_area', 'find_resources', 'map_terrain']
            },
            'combat': {
                'base_priority': 0.8,
                'description': 'Combat-related objectives',
                'goals': ['engage_enemy', 'evade_threat', 'find_weapons']
            },
            'crafting': {
                'base_priority': 0.5,
                'description': 'Item crafting objectives',
                'goals': ['gather_materials', 'craft_item', 'upgrade_equipment']
            },
            'building': {
                'base_priority': 0.4,
                'description': 'Construction objectives',
                'goals': ['build_shelter', 'fortify_position', 'create_storage']
            }
        }
        
        # Active objectives tracking
        self.active_objectives: List[Dict[str, Any]] = []
        self.completed_objectives: List[Dict[str, Any]] = []
    
    def create_objective(self, objective_type: str, goal: str, 
                        priority: str = 'medium', details: str = "") -> Dict[str, Any]:
        """
        Create a new objective.
        
        Args:
            objective_type (str): Type of objective
            goal (str): Specific goal
            priority (str): Priority level ('low', 'medium', 'high')
            details (str): Additional objective details
            
        Returns:
            dict: Created objective
        """
        try:
            # Validate objective type
            if objective_type not in self.objective_types:
                self.logger.warning(f"Unknown objective type: {objective_type}")
                objective_type = 'survival'  # Default to survival
            
            # Get base priority for type
            base_priority = self.objective_types[objective_type]['base_priority']
            
            # Convert priority string to multiplier
            priority_multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5
            }
            priority_mult = priority_multipliers.get(priority, 1.0)
            
            # Calculate final priority
            final_priority = min(
                self.max_priority,
                max(self.min_priority, base_priority * priority_mult)
            )
            
            # Create objective
            objective = {
                'type': objective_type,
                'goal': goal,
                'priority': final_priority,
                'details': details,
                'status': 'active',
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            return objective
            
        except Exception as e:
            self.logger.error(f"Error creating objective: {e}")
            return self._get_fallback_objective()
    
    def prioritize_objectives(self, objectives: List[Dict[str, Any]], 
                            situation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prioritize objectives based on current situation.
        
        Args:
            objectives (list): List of objectives to prioritize
            situation (dict): Current situation analysis
            
        Returns:
            list: Prioritized objectives
        """
        try:
            # Calculate priority scores
            scored_objectives = []
            for obj in objectives:
                score = self._calculate_priority_score(obj, situation)
                scored_objectives.append((obj, score))
            
            # Sort by priority score
            sorted_objectives = [
                obj for obj, _ in sorted(
                    scored_objectives,
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            
            # Limit number of objectives
            return sorted_objectives[:self.max_objectives]
            
        except Exception as e:
            self.logger.error(f"Error prioritizing objectives: {e}")
            return objectives
    
    def _calculate_priority_score(self, objective: Dict[str, Any], 
                                situation: Dict[str, Any]) -> float:
        """Calculate priority score for objective."""
        try:
            base_score = objective.get('priority', 0.5)
            
            # Apply situational modifiers
            modifiers = self._get_situational_modifiers(objective, situation)
            
            # Calculate final score
            score = base_score
            for modifier in modifiers:
                score *= modifier
            
            return min(self.max_priority, max(self.min_priority, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating priority score: {e}")
            return 0.5
    
    def _get_situational_modifiers(self, objective: Dict[str, Any], 
                                 situation: Dict[str, Any]) -> List[float]:
        """Get priority modifiers based on situation."""
        modifiers = []
        
        try:
            obj_type = objective.get('type')
            
            # Health-based modifiers
            if obj_type == 'survival':
                health = situation.get('player_status', {}).get('health', {}).get('value', 100)
                if health < 30:
                    modifiers.append(2.0)  # High priority when health is low
                elif health < 70:
                    modifiers.append(1.5)
            
            # Threat-based modifiers
            if obj_type == 'combat':
                threats = situation.get('threats', [])
                if threats:
                    modifiers.append(1.5)  # Higher priority when threats present
            
            # Resource-based modifiers
            if obj_type in ['crafting', 'building']:
                resources = situation.get('resources', {}).get('current', {})
                if not resources:
                    modifiers.append(0.5)  # Lower priority when no resources
            
            # Exploration modifiers
            if obj_type == 'exploration':
                explored = situation.get('environment', {}).get('explored_ratio', 0.5)
                if explored < 0.3:
                    modifiers.append(1.2)  # Higher priority in unexplored areas
            
        except Exception as e:
            self.logger.error(f"Error getting situational modifiers: {e}")
        
        return modifiers if modifiers else [1.0]
    
    def update_objective_status(self, objective_id: str, 
                              status: str, 
                              details: Optional[str] = None) -> None:
        """
        Update status of an objective.
        
        Args:
            objective_id (str): Objective identifier
            status (str): New status ('active', 'completed', 'failed')
            details (str, optional): Additional status details
        """
        try:
            for obj in self.active_objectives:
                if obj.get('id') == objective_id:
                    obj['status'] = status
                    obj['last_updated'] = datetime.now().isoformat()
                    if details:
                        obj['status_details'] = details
                    
                    # Move to completed list if done
                    if status == 'completed':
                        self.completed_objectives.append(obj)
                        self.active_objectives.remove(obj)
                    
                    break
                    
        except Exception as e:
            self.logger.error(f"Error updating objective status: {e}")
    
    def add_objective_type(self, type_name: str, base_priority: float,
                          description: str = "", goals: Optional[List[str]] = None) -> None:
        """
        Add new objective type.
        
        Args:
            type_name (str): Name of objective type
            base_priority (float): Base priority score
            description (str): Type description
            goals (list, optional): List of possible goals
        """
        try:
            self.objective_types[type_name] = {
                'base_priority': min(self.max_priority, max(self.min_priority, base_priority)),
                'description': description,
                'goals': goals or []
            }
            self.logger.info(f"Added new objective type: {type_name}")
            
        except Exception as e:
            self.logger.error(f"Error adding objective type: {e}")
    
    def get_active_objectives(self) -> List[Dict[str, Any]]:
        """
        Get list of active objectives.
        
        Returns:
            list: Active objectives
        """
        return self.active_objectives.copy()
    
    def get_completed_objectives(self) -> List[Dict[str, Any]]:
        """
        Get list of completed objectives.
        
        Returns:
            list: Completed objectives
        """
        return self.completed_objectives.copy()
    
    def _get_fallback_objective(self) -> Dict[str, Any]:
        """Get basic fallback objective."""
        return {
            'type': 'survival',
            'goal': 'survive',
            'priority': 1.0,
            'details': 'Basic survival objective',
            'status': 'active',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
