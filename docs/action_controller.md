# Action Controller Documentation

## Overview
The Action Controller is responsible for translating high-level game actions into low-level input commands and executing them through input simulation. It provides a robust system for mapping game actions to keyboard and mouse inputs while handling timing, validation, and error cases.

## Features
- Input simulation for keyboard and mouse
- Configurable control mappings
- Game-specific control schemes
- Action validation
- Input timing management
- Error handling and recovery
- Stuck input cleanup
- Cooldown management

## Core Components

### 1. Input Controllers
- Keyboard controller (pynput)
- Mouse controller (pynput)
- Input timing management
- Action queuing

### 2. Control Mapping
- Game-specific control schemes
- Configurable key bindings
- Action type definitions
- Input sequence generation

### 3. Action Validation
- Format validation
- State-based validation
- Cooldown checking
- Requirement verification
- Constraint checking

## Configuration
The system uses multiple configuration files:

### Control Maps (control_maps.json)
```json
{
    "games": {
        "minecraft": {
            "keyboard": {
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
        }
    },
    "action_types": {
        "movement": {
            "continuous": true,
            "default_duration": 0.1
        }
    }
}
```

### Action Controller Config
```json
{
    "controls": {
        "input_delay": 0.05,
        "action_timeout": 5.0
    }
}
```

## Extension Points

### 1. Custom Game Controls
Add support for new games:
1. Create game-specific control scheme
2. Define key mappings
3. Add action types
4. Configure input timing

### 2. Enhanced Action Types
Extend action capabilities:
1. Add new action categories
2. Define custom input sequences
3. Create specialized mappings
4. Implement timing patterns

### 3. Input Validation
Add validation rules:
1. Create new validation checks
2. Add state requirements
3. Implement constraints
4. Define cooldown rules

### 4. Input Processing
Enhance input handling:
1. Add new input types
2. Create custom modifiers
3. Implement input combinations
4. Add macro support

## Dependencies
- pynput for input simulation
- Python's json module
- pathlib for file operations
- logging system
- time module for timing control

## Usage Example

```python
from game_ai.core.action_controller import ActionController

# Initialize controller
controller = ActionController(config_path="config/controls_config.json", game_type="minecraft")

# Define action
action = {
    'type': 'movement',
    'key': 'forward',
    'duration': 1.0
}

# Execute action
success = controller.execute_action(action)
```

## Action Types

### Movement Actions
```python
{
    'type': 'movement',
    'key': 'forward',  # forward, backward, left, right, jump
    'duration': 1.0
}
```

### Mouse Movement Actions
```python
{
    'type': 'mouse',
    'subtype': 'move',
    'position': [x, y],  # Absolute screen coordinates
    'duration': 0.1
}

# Or relative movement
{
    'type': 'mouse',
    'subtype': 'move_relative',
    'offset': [dx, dy],  # Relative movement in pixels
    'duration': 0.1
}
```

### Interaction Actions
```python
{
    'type': 'action',
    'key': 'mouse1',  # mouse1, mouse2
    'duration': 0.5
}
```

### Menu Actions
```python
{
    'type': 'menu',
    'key': 'inventory',
    'duration': 0.1
}
```

### Hotbar Actions
```python
{
    'type': 'hotbar',
    'key': 'slot1',
    'duration': 0.1
}
```

## Best Practices
1. Proper input timing
2. Error handling
3. Input cleanup
4. State validation
5. Resource management

## Performance Considerations
- Input delay management
- Action queuing efficiency
- Resource usage monitoring
- Timing accuracy
- Input cleanup

## Error Handling
The system implements comprehensive error handling for:
- Invalid actions
- Input failures
- State validation
- Timing issues
- Resource constraints

## Input Simulation
The controller supports:
- Keyboard key press/release
- Mouse button clicks
- Mouse movement (absolute and relative)
- Mouse movement smoothing
- Movement interpolation
- Input combinations
- Timed sequences
- Cursor position tracking

### Special Keys Support
```python
special_keys = {
    'space': Key.space,
    'shift': Key.shift,
    'ctrl': Key.ctrl,
    'alt': Key.alt,
    'esc': Key.esc,
    # ... and more
}
```

## Validation System

### Format Validation
- Required fields checking
- Type validation
- Value range checking
- Duration validation

### State Validation
- Resource requirements
- Position constraints
- State requirements
- Cooldown checking

### Constraint Checking
- Distance constraints
- State constraints
- Timing constraints
- Resource constraints

## Future Enhancements
1. Macro recording
2. Complex input sequences
3. Advanced timing control
4. Input optimization
5. State prediction
6. Enhanced validation
7. Performance monitoring
8. Custom input devices
9. Advanced mouse movement patterns
10. Path prediction
11. Movement smoothing algorithms
12. Acceleration/deceleration control

## Integration Points

### 1. Decision Engine Integration
- Receive action commands
- Execute planned sequences
- Provide feedback
- Handle timing

### 2. Vision System Integration
- State validation
- Position checking
- Object interaction
- Timing coordination

### 3. Memory System Integration
- Action history
- State tracking
- Performance metrics
- Error logging

## Safety Features
1. Input cleanup on exit
2. Stuck key detection
3. Emergency stop
4. Error recovery
5. State validation
