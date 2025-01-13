# Decision Engine Documentation

## Overview
The Decision Engine is the strategic brain of the AI system, responsible for analyzing game states, planning actions, and making decisions. It combines rule-based decision making with LLM-powered strategic planning to create intelligent and adaptive behavior.

## Features
- Strategic planning
- Tactical action planning
- Reactive response system
- LLM integration
- Priority-based decision making
- Resource management
- Threat assessment
- Opportunity identification

## Core Components

### 1. Strategic Planning
- Long-term goal setting
- Resource prioritization
- Survival strategy development
- LLM-enhanced decision making
- Objective management

### 2. Tactical Planning
- Action sequence generation
- Resource gathering planning
- Movement optimization
- Combat tactics
- Environment navigation

### 3. Reactive System
- Immediate threat response
- Health management
- Opportunity exploitation
- Dynamic priority adjustment

## Configuration
The system uses multiple configuration files:

### Game Rules (game_rules.json)
```json
{
    "objectives": {
        "primary": [
            "get_wood",
            "make_tools",
            "find_food",
            "build_shelter"
        ]
    }
}
```

### Control Maps (control_maps.json)
```json
{
    "movement": {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "jump": "space"
    },
    "actions": {
        "attack": "mouse1",
        "use": "mouse2"
    }
}
```

## Extension Points

### 1. Custom Strategic Planning
Add new strategic capabilities:
1. Create new strategy evaluation methods
2. Implement custom objective types
3. Add specialized resource management
4. Develop new planning algorithms

### 2. Enhanced Tactical Planning
Extend tactical capabilities:
1. Add new action types
2. Implement specialized movement patterns
3. Create custom combat strategies
4. Develop resource gathering patterns

### 3. Reactive System Enhancement
Improve reactive capabilities:
1. Add new threat responses
2. Implement opportunity detection
3. Create emergency protocols
4. Develop status effect handling

### 4. LLM Integration
Extend LLM capabilities:
1. Customize prompts
2. Add new decision contexts
3. Implement response parsing
4. Create specialized behaviors

## Dependencies
- Python's json module
- pathlib for file operations
- logging system
- datetime for temporal operations
- Ollama interface for LLM integration

## Usage Example

```python
from game_ai.core.decision_engine import DecisionEngine

# Initialize the engine
engine = DecisionEngine(config_path="config/decision_config.json")

# Generate strategic plan
strategy = engine.strategic_planning(game_state, memory)

# Generate tactical actions
actions = engine.tactical_planning(strategy, game_state)

# Get reactive response
reaction = engine.reactive_response(game_state)
```

## Decision Making Process

### 1. Strategic Level
```python
strategy = {
    'objectives': [
        {
            'type': 'minecraft',
            'priority': 'high',
            'goal': 'get_wood',
            'details': 'Initial survival task'
        }
    ],
    'priorities': [],
    'constraints': {},
    'timestamp': '2024-01-01T12:00:00'
}
```

### 2. Tactical Level
```python
actions = [
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
]
```

### 3. Reactive Level
```python
reaction = {
    'type': 'evade',
    'priority': 'immediate',
    'target': 'zombie'
}
```

## Best Practices
1. Regular strategy updates
2. Efficient action sequencing
3. Proper error handling
4. Performance monitoring
5. Resource optimization

## Performance Considerations
- Action queue management
- Decision tree optimization
- Memory usage efficiency
- LLM response caching
- State change detection

## Error Handling
The system implements comprehensive error handling for:
- Configuration loading
- Strategy generation
- Action planning
- LLM integration
- State processing

## Integration Points

### 1. Vision System Integration
- Process visual input
- Detect objects and threats
- Analyze environment
- Track resources

### 2. Memory System Integration
- Access historical data
- Store decision outcomes
- Track resource states
- Maintain objective progress

### 3. Action Controller Integration
- Execute planned actions
- Handle movement commands
- Manage action timing
- Process feedback

### 4. LLM Integration
- Strategic planning
- Tactical decisions
- Natural language processing
- Behavior adaptation

## Future Enhancements
1. Advanced decision trees
2. Machine learning integration
3. Pattern recognition
4. Predictive planning
5. Multi-objective optimization
6. Dynamic strategy adjustment
7. Enhanced threat assessment
8. Improved resource management

## Action Types

### Movement Actions
```python
{
    'type': 'movement',
    'key': 'forward',  # forward, backward, left, right, space
    'duration': 1.0
}
```

### Interaction Actions
```python
{
    'type': 'action',
    'key': 'mouse1',  # mouse1, mouse2
    'duration': 2.0
}
```

## Priority System
- Immediate: Threat response, critical health
- High: Essential resources, primary objectives
- Medium: Secondary objectives, opportunities
- Low: Optional tasks, exploration

## State Evaluation
The engine continuously evaluates:
- Player status
- Resource levels
- Environmental threats
- Opportunities
- Constraints
- Progress towards objectives
