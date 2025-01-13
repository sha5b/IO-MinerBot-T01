"""Situation analysis component for evaluating game state."""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class SituationAnalyzer:
    """Analyzes game state to inform decision making."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize situation analyzer.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Analysis settings
        self.thresholds = {
            'threat_distance': 20.0,     # Distance to consider threat immediate
            'opportunity_distance': 10.0, # Distance to consider resource reachable
            'low_health': 30,            # Health threshold for danger
            'low_resources': 25,         # Resource threshold for scarcity
            'safe_distance': 30.0,       # Distance considered safe from threats
            'visibility_threshold': 0.6   # Minimum visibility for normal operation
        }
        
        # Risk level weights
        self.risk_weights = {
            'health': 0.3,
            'threats': 0.3,
            'resources': 0.2,
            'environment': 0.2
        }
    
    def analyze_situation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current game situation.
        
        Args:
            game_state (dict): Current game state
            
        Returns:
            dict: Situation analysis
        """
        try:
            # Analyze different aspects
            player_status = self._analyze_player_status(game_state)
            threats = self._analyze_threats(game_state)
            opportunities = self._analyze_opportunities(game_state)
            resources = self._analyze_resources(game_state)
            environment = self._analyze_environment(game_state)
            
            # Calculate overall risk level
            risk_level = self._calculate_risk_level(
                player_status,
                threats,
                resources,
                environment
            )
            
            # Compile analysis
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'player_status': player_status,
                'threats': threats,
                'opportunities': opportunities,
                'resources': resources,
                'environment': environment,
                'risk_level': risk_level
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing situation: {e}")
            return self._get_fallback_analysis()
    
    def _analyze_player_status(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player status."""
        try:
            player = game_state.get('player', {})
            
            # Get basic stats
            health = player.get('health', 100)
            status_effects = player.get('status', [])
            equipment = player.get('equipment', {})
            
            # Evaluate health status
            health_status = 'critical' if health < 30 else 'low' if health < 70 else 'good'
            
            # Check for negative status effects
            negative_effects = [
                effect for effect in status_effects
                if effect.get('type') in ['poison', 'weakness', 'slowness']
            ]
            
            return {
                'health': {
                    'value': health,
                    'status': health_status
                },
                'status_effects': status_effects,
                'negative_effects': negative_effects,
                'equipment': equipment,
                'combat_ready': health > 50 and not negative_effects
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing player status: {e}")
            return {'health': {'value': 100, 'status': 'unknown'}}
    
    def _analyze_threats(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze threats in environment."""
        try:
            environment = game_state.get('environment', {})
            threats = environment.get('threats', [])
            
            analyzed_threats = []
            for threat in threats:
                # Calculate threat level
                base_threat = threat.get('threat_level', 0.5)
                distance = threat.get('distance', float('inf'))
                
                # Distance-based threat scaling
                if distance < self.thresholds['threat_distance']:
                    threat_level = base_threat * (
                        1 + (self.thresholds['threat_distance'] - distance) / 
                        self.thresholds['threat_distance']
                    )
                else:
                    threat_level = base_threat * (
                        self.thresholds['threat_distance'] / distance
                    )
                
                analyzed_threats.append({
                    'type': threat.get('type', 'unknown'),
                    'distance': distance,
                    'threat_level': min(1.0, threat_level),
                    'position': threat.get('position'),
                    'immediate': distance < self.thresholds['threat_distance']
                })
            
            # Sort by threat level
            return sorted(
                analyzed_threats,
                key=lambda x: x['threat_level'],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing threats: {e}")
            return []
    
    def _analyze_opportunities(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze opportunities in environment."""
        try:
            environment = game_state.get('environment', {})
            opportunities = environment.get('opportunities', [])
            
            analyzed_opps = []
            for opp in opportunities:
                # Calculate opportunity value
                base_value = opp.get('value', 0.5)
                distance = opp.get('distance', float('inf'))
                
                # Distance-based value scaling
                if distance < self.thresholds['opportunity_distance']:
                    value = base_value
                else:
                    value = base_value * (
                        self.thresholds['opportunity_distance'] / distance
                    )
                
                analyzed_opps.append({
                    'type': opp.get('type', 'unknown'),
                    'distance': distance,
                    'value': min(1.0, value),
                    'position': opp.get('position'),
                    'reachable': distance < self.thresholds['opportunity_distance']
                })
            
            # Sort by value
            return sorted(
                analyzed_opps,
                key=lambda x: x['value'],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing opportunities: {e}")
            return []
    
    def _analyze_resources(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource status."""
        try:
            inventory = game_state.get('inventory', {})
            
            # Get current resources
            current = inventory.get('items', {})
            capacity = inventory.get('capacity', 100)
            
            # Check for low resources
            low_resources = []
            for item, count in current.items():
                threshold = inventory.get('thresholds', {}).get(item, 25)
                if count < threshold:
                    low_resources.append(item)
            
            return {
                'current': current,
                'capacity': capacity,
                'low_resources': low_resources,
                'full': sum(current.values()) >= capacity,
                'empty': not current
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing resources: {e}")
            return {'current': {}, 'capacity': 100, 'low_resources': []}
    
    def _analyze_environment(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environment conditions."""
        try:
            environment = game_state.get('environment', {})
            
            # Get environment properties
            terrain = environment.get('terrain', {})
            visibility = environment.get('visibility', 1.0)
            obstacles = environment.get('obstacles', [])
            
            # Calculate exploration status
            explored = environment.get('explored_chunks', 0)
            total_chunks = environment.get('total_chunks', 100)
            explored_ratio = explored / total_chunks if total_chunks > 0 else 0
            
            return {
                'terrain_type': terrain.get('type', 'unknown'),
                'visibility': visibility,
                'visibility_status': 'low' if visibility < self.thresholds['visibility_threshold'] else 'good',
                'obstacles': len(obstacles),
                'explored_ratio': explored_ratio,
                'safe_zone': self._is_safe_zone(environment)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing environment: {e}")
            return {'terrain_type': 'unknown', 'visibility': 1.0}
    
    def _calculate_risk_level(self, player_status: Dict[str, Any],
                            threats: List[Dict[str, Any]],
                            resources: Dict[str, Any],
                            environment: Dict[str, Any]) -> float:
        """Calculate overall risk level."""
        try:
            # Health risk
            health = player_status.get('health', {}).get('value', 100)
            health_risk = 1.0 - (health / 100)
            
            # Threat risk
            threat_risk = max(
                [threat['threat_level'] for threat in threats],
                default=0.0
            )
            
            # Resource risk
            resource_count = len(resources.get('current', {}))
            resource_risk = 1.0 if resource_count == 0 else (
                0.5 if resources.get('low_resources') else 0.0
            )
            
            # Environment risk
            visibility = environment.get('visibility', 1.0)
            env_risk = 1.0 - visibility
            
            # Calculate weighted risk
            risk_level = (
                self.risk_weights['health'] * health_risk +
                self.risk_weights['threats'] * threat_risk +
                self.risk_weights['resources'] * resource_risk +
                self.risk_weights['environment'] * env_risk
            )
            
            return min(1.0, max(0.0, risk_level))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {e}")
            return 0.5
    
    def _is_safe_zone(self, environment: Dict[str, Any]) -> bool:
        """Determine if current location is safe."""
        try:
            threats = environment.get('threats', [])
            
            # Check if any threats are within safe distance
            for threat in threats:
                if threat.get('distance', float('inf')) < self.thresholds['safe_distance']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking safe zone: {e}")
            return True
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update analysis thresholds.
        
        Args:
            new_thresholds (dict): New threshold values
        """
        self.thresholds.update(new_thresholds)
        self.logger.info("Updated analysis thresholds")
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Get basic fallback analysis."""
        return {
            'timestamp': datetime.now().isoformat(),
            'player_status': {
                'health': {'value': 100, 'status': 'unknown'},
                'combat_ready': True
            },
            'threats': [],
            'opportunities': [],
            'resources': {'current': {}, 'capacity': 100, 'low_resources': []},
            'environment': {'terrain_type': 'unknown', 'visibility': 1.0},
            'risk_level': 0.0
        }
