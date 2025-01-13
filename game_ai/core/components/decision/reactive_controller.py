"""Reactive control component for immediate responses to game state changes."""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

class ReactiveController:
    """Handles immediate responses to game state changes."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reactive controller.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.last_state: Optional[Dict[str, Any]] = None
        self.last_response_time = datetime.now()
        
        # Response templates for different situations
        self.response_templates = {
            'damage_taken': {
                'priority': 'immediate',
                'actions': [
                    {'type': 'movement', 'key': 'backward', 'duration': 0.5},
                    {'type': 'look', 'key': 'look_up', 'duration': 0.2}
                ]
            },
            'threat_detected': {
                'priority': 'high',
                'actions': [
                    {'type': 'look', 'key': 'look_left', 'duration': 0.2},
                    {'type': 'look', 'key': 'look_right', 'duration': 0.2}
                ]
            },
            'resource_spotted': {
                'priority': 'medium',
                'actions': [
                    {'type': 'look', 'key': 'look_up', 'duration': 0.2},
                    {'type': 'movement', 'key': 'forward', 'duration': 0.5}
                ]
            }
        }
        
        # Thresholds for reactive triggers
        self.thresholds = {
            'health_change': -10,  # Significant health drop
            'threat_distance': 20,  # Close threat detection
            'opportunity_distance': 10,  # Nearby resource detection
            'reaction_cooldown': 0.5  # Minimum time between reactions
        }
    
    def process_state_changes(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process changes in game state and generate immediate response if needed.
        
        Args:
            game_state (dict): Current game state
            
        Returns:
            dict: Reactive response or None if no response needed
        """
        try:
            # Check if enough time has passed since last response
            if not self._can_respond():
                return None
            
            # Detect changes from last state
            changes = self._detect_changes(game_state)
            
            # Check if changes require immediate response
            if self._requires_response(changes):
                response = self._generate_response(changes, game_state)
                
                # Update last response time
                self.last_response_time = datetime.now()
                
                return response
            
            # Update last state
            self.last_state = self._sanitize_state(game_state)
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing state changes: {e}")
            return None
    
    def _detect_changes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant changes from last state."""
        if not self.last_state:
            return {}
            
        changes = {
            'health_change': self._detect_health_change(game_state),
            'new_threats': self._detect_new_threats(game_state),
            'new_opportunities': self._detect_new_opportunities(game_state),
            'environment_changes': self._detect_environment_changes(game_state)
        }
        
        return changes
    
    def _detect_health_change(self, game_state: Dict[str, Any]) -> float:
        """Calculate health change from last state."""
        if not self.last_state:
            return 0.0
            
        current_health = game_state.get('player', {}).get('health', 100)
        last_health = self.last_state.get('player', {}).get('health', 100)
        
        return current_health - last_health
    
    def _detect_new_threats(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect new threats that weren't in last state."""
        if not self.last_state:
            return []
            
        current_threats = set(
            threat['id'] for threat in 
            game_state.get('environment', {}).get('threats', [])
        )
        last_threats = set(
            threat['id'] for threat in 
            self.last_state.get('environment', {}).get('threats', [])
        )
        
        new_threat_ids = current_threats - last_threats
        
        return [
            threat for threat in 
            game_state.get('environment', {}).get('threats', [])
            if threat['id'] in new_threat_ids
        ]
    
    def _detect_new_opportunities(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect new opportunities that weren't in last state."""
        if not self.last_state:
            return []
            
        current_opps = set(
            opp['id'] for opp in 
            game_state.get('environment', {}).get('opportunities', [])
        )
        last_opps = set(
            opp['id'] for opp in 
            self.last_state.get('environment', {}).get('opportunities', [])
        )
        
        new_opp_ids = current_opps - last_opps
        
        return [
            opp for opp in 
            game_state.get('environment', {}).get('opportunities', [])
            if opp['id'] in new_opp_ids
        ]
    
    def _detect_environment_changes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant environment changes."""
        if not self.last_state:
            return {}
            
        return {
            'terrain': self._detect_terrain_changes(game_state),
            'visibility': self._detect_visibility_changes(game_state),
            'obstacles': self._detect_obstacle_changes(game_state)
        }
    
    def _detect_terrain_changes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in terrain."""
        if not self.last_state:
            return {}
            
        current_terrain = game_state.get('environment', {}).get('terrain', {})
        last_terrain = self.last_state.get('environment', {}).get('terrain', {})
        
        changes = {}
        
        # Compare terrain types
        if current_terrain.get('type') != last_terrain.get('type'):
            changes['type'] = {
                'old': last_terrain.get('type'),
                'new': current_terrain.get('type')
            }
        
        # Compare elevation
        if current_terrain.get('elevation') != last_terrain.get('elevation'):
            changes['elevation'] = {
                'old': last_terrain.get('elevation'),
                'new': current_terrain.get('elevation')
            }
        
        return changes
    
    def _detect_visibility_changes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in visibility conditions."""
        if not self.last_state:
            return {}
            
        current_vis = game_state.get('environment', {}).get('visibility', {})
        last_vis = self.last_state.get('environment', {}).get('visibility', {})
        
        changes = {}
        
        # Compare visibility level
        if current_vis.get('level') != last_vis.get('level'):
            changes['level'] = {
                'old': last_vis.get('level'),
                'new': current_vis.get('level')
            }
        
        # Compare lighting
        if current_vis.get('lighting') != last_vis.get('lighting'):
            changes['lighting'] = {
                'old': last_vis.get('lighting'),
                'new': current_vis.get('lighting')
            }
        
        return changes
    
    def _detect_obstacle_changes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in obstacles."""
        if not self.last_state:
            return {}
            
        current_obs = game_state.get('environment', {}).get('obstacles', [])
        last_obs = self.last_state.get('environment', {}).get('obstacles', [])
        
        return {
            'new': [obs for obs in current_obs if obs not in last_obs],
            'removed': [obs for obs in last_obs if obs not in current_obs]
        }
    
    def _requires_response(self, changes: Dict[str, Any]) -> bool:
        """Determine if changes require immediate response."""
        # Check health changes
        if changes.get('health_change', 0) < self.thresholds['health_change']:
            return True
            
        # Check new threats
        for threat in changes.get('new_threats', []):
            if threat.get('distance', float('inf')) < self.thresholds['threat_distance']:
                return True
                
        # Check new opportunities
        for opp in changes.get('new_opportunities', []):
            if opp.get('distance', float('inf')) < self.thresholds['opportunity_distance']:
                return True
                
        return False
    
    def _generate_response(self, changes: Dict[str, Any], 
                         current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate response to changes."""
        response_type = self._determine_response_type(changes)
        template = self.response_templates.get(response_type, {})
        
        response = {
            'type': response_type,
            'priority': template.get('priority', 'medium'),
            'actions': self._customize_response_actions(
                template.get('actions', []),
                changes,
                current_state
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _determine_response_type(self, changes: Dict[str, Any]) -> str:
        """Determine appropriate response type based on changes."""
        if changes.get('health_change', 0) < self.thresholds['health_change']:
            return 'damage_taken'
            
        if changes.get('new_threats', []):
            return 'threat_detected'
            
        if changes.get('new_opportunities', []):
            return 'resource_spotted'
            
        return 'generic'
    
    def _customize_response_actions(self, base_actions: List[Dict[str, Any]],
                                  changes: Dict[str, Any],
                                  current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Customize response actions based on specific situation."""
        customized = []
        
        for action in base_actions:
            # Deep copy the action
            new_action = {**action}
            
            # Customize based on situation
            if action['type'] == 'movement':
                new_action['duration'] = self._adjust_movement_duration(
                    action['duration'],
                    changes,
                    current_state
                )
            elif action['type'] == 'look':
                new_action = self._adjust_look_direction(
                    new_action,
                    changes,
                    current_state
                )
            
            customized.append(new_action)
        
        return customized
    
    def _can_respond(self) -> bool:
        """Check if enough time has passed since last response."""
        time_since_last = (datetime.now() - self.last_response_time).total_seconds()
        return time_since_last >= self.thresholds['reaction_cooldown']
    
    def _adjust_movement_duration(self, base_duration: float,
                                changes: Dict[str, Any],
                                current_state: Dict[str, Any]) -> float:
        """Adjust movement duration based on situation."""
        # Increase duration if moving away from threat
        if changes.get('new_threats', []):
            return base_duration * 1.5
            
        # Decrease duration if moving toward opportunity
        if changes.get('new_opportunities', []):
            return base_duration * 0.8
            
        return base_duration
    
    def _adjust_look_direction(self, action: Dict[str, Any],
                             changes: Dict[str, Any],
                             current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust look direction based on situation."""
        # If there's a specific threat/opportunity location, adjust look direction
        if changes.get('new_threats', []):
            threat = changes['new_threats'][0]
            return self._create_look_action_for_target(threat.get('position'))
            
        if changes.get('new_opportunities', []):
            opp = changes['new_opportunities'][0]
            return self._create_look_action_for_target(opp.get('position'))
            
        return action
    
    def _create_look_action_for_target(self, target_pos: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Create appropriate look action for target position."""
        if not target_pos:
            return {
                'type': 'look',
                'key': 'look_up',
                'duration': 0.2
            }
            
        # This would calculate appropriate look direction based on target position
        # For now, return a simple look action
        return {
            'type': 'look',
            'key': 'look_up',
            'duration': 0.2
        }
    
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
        return sanitized
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update reactive thresholds."""
        self.thresholds.update(new_thresholds)
        self.logger.info("Updated reactive thresholds")
    
    def add_response_template(self, name: str, template: Dict[str, Any]) -> None:
        """Add new response template."""
        self.response_templates[name] = template
        self.logger.info(f"Added new response template: {name}")
