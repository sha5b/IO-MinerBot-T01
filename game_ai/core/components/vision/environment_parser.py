"""
Advanced environment parser that creates meaningful tactical summaries for LLM context
"""

def parse_environment(raw_data):
    """
    Transform raw environment data into a high-level tactical assessment
    
    @param {dict} raw_data - Raw environment data from vision system
    @return {dict} Strategic environment summary
    """
    if not raw_data:
        return {}

    env = raw_data.get('environment', {})
    player = raw_data.get('player', {})

    # Analyze tactical situation
    threats = env.get('threats', [])
    resources = env.get('resources', [])
    terrain_type = env.get('terrain_type', 'unknown')
    
    # Assess threat levels and positions
    threat_assessment = _assess_threats(threats)
    
    # Evaluate resource availability
    resource_assessment = _assess_resources(resources)
    
    # Analyze terrain advantages/disadvantages
    terrain_assessment = _assess_terrain(terrain_type, env.get('terrain_analysis', {}))
    
    # Evaluate player's tactical position
    position_assessment = _assess_position(player, threats, resources, terrain_type)

    return {
        'summary': _generate_situation_summary(
            threat_assessment,
            resource_assessment,
            terrain_assessment,
            position_assessment
        ),
        'tactical_details': {
            'threat_level': threat_assessment['level'],
            'resource_availability': resource_assessment['availability'],
            'terrain_advantage': terrain_assessment['advantage'],
            'position_quality': position_assessment['quality']
        },
        'tactical_rating': _calculate_tactical_rating(
            threat_assessment,
            resource_assessment,
            terrain_assessment,
            position_assessment
        )
    }

def _assess_threats(threats):
    """Analyze threat distribution and severity"""
    immediate = sum(1 for t in threats if t.get('distance', 0) <= 100)
    close = sum(1 for t in threats if 100 < t.get('distance', 0) <= 300)
    distant = sum(1 for t in threats if t.get('distance', 0) > 300)
    
    level = 'low'
    if immediate > 2 or (immediate > 0 and close > 2):
        level = 'high'
    elif immediate > 0 or close > 1:
        level = 'medium'
        
    return {
        'level': level,
        'distribution': {
            'immediate': immediate,
            'close': close,
            'distant': distant
        }
    }

def _assess_resources(resources):
    """Evaluate resource accessibility and distribution"""
    nearby = sum(1 for r in resources if r.get('distance', 0) <= 150)
    reachable = sum(1 for r in resources if 150 < r.get('distance', 0) <= 300)
    
    availability = 'scarce'
    if nearby > 2 or (nearby > 0 and reachable > 2):
        availability = 'abundant'
    elif nearby > 0 or reachable > 1:
        availability = 'moderate'
        
    return {
        'availability': availability,
        'distribution': {
            'nearby': nearby,
            'reachable': reachable
        }
    }

def _assess_terrain(terrain_type, terrain_analysis):
    """Analyze terrain tactical implications"""
    advantage = 'neutral'
    
    if terrain_type == 'dense':
        advantage = 'strong'  # Good cover and maneuverability
    elif terrain_type == 'sparse':
        advantage = 'weak'    # Exposed position
    
    return {
        'advantage': advantage,
        'type': terrain_type
    }

def _assess_position(player, threats, resources, terrain_type):
    """Evaluate tactical quality of current position"""
    if not player:
        return {'quality': 'unknown'}
        
    # Count nearby threats and resources
    nearby_threats = sum(1 for t in threats if t.get('distance', 0) <= 200)
    nearby_resources = sum(1 for r in resources if r.get('distance', 0) <= 200)
    
    quality = 'neutral'
    
    # Position is good if near resources with few threats
    if nearby_resources > nearby_threats and terrain_type != 'sparse':
        quality = 'strong'
    # Position is poor if surrounded by threats or exposed
    elif nearby_threats > 1 or (nearby_threats > 0 and terrain_type == 'sparse'):
        quality = 'weak'
        
    return {'quality': quality}

def _generate_situation_summary(threats, resources, terrain, position):
    """Generate a concise tactical situation summary"""
    summary_parts = []
    
    # Threat assessment
    if threats['level'] == 'high':
        summary_parts.append("High threat environment")
    elif threats['level'] == 'medium':
        summary_parts.append("Moderate threat presence")
    else:
        summary_parts.append("Low threat area")
    
    # Resource assessment
    if resources['availability'] == 'abundant':
        summary_parts.append("resource-rich")
    elif resources['availability'] == 'moderate':
        summary_parts.append("adequate resources")
    else:
        summary_parts.append("resource-scarce")
    
    # Terrain and position
    if terrain['advantage'] == 'strong' and position['quality'] == 'strong':
        summary_parts.append("in defensible position")
    elif terrain['advantage'] == 'weak' or position['quality'] == 'weak':
        summary_parts.append("exposed position")
        
    return " with ".join(summary_parts)

def _calculate_tactical_rating(threats, resources, terrain, position):
    """
    Calculate overall tactical situation rating from 0 (critical) to 10 (optimal)
    """
    rating = 5  # Start neutral
    
    # Threat impact (-4 to 0)
    if threats['level'] == 'high':
        rating -= 4
    elif threats['level'] == 'medium':
        rating -= 2
    
    # Resource bonus (0 to +2)
    if resources['availability'] == 'abundant':
        rating += 2
    elif resources['availability'] == 'moderate':
        rating += 1
    
    # Terrain advantage (-1 to +1)
    if terrain['advantage'] == 'strong':
        rating += 1
    elif terrain['advantage'] == 'weak':
        rating -= 1
    
    # Position quality (-2 to +2)
    if position['quality'] == 'strong':
        rating += 2
    elif position['quality'] == 'weak':
        rating -= 2
    
    return max(0, min(10, rating))
