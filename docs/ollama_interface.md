# Ollama Interface Documentation

## Overview
The Ollama Interface provides a robust connection to the Ollama LLM service, enabling AI-powered decision making and strategic planning. It implements a singleton pattern with thread-safe streaming responses and comprehensive error handling.

## Features
- Singleton instance management
- Streaming response handling
- Thread-safe operations
- Configurable model settings
- Conversation history management
- Context formatting
- Response parsing
- Error handling
- Prompt management

## Core Components

### 1. Connection Management
- Singleton instance control
- Model initialization
- Connection verification
- Streaming thread management

### 2. Response Handling
- Asynchronous streaming
- Response queuing
- Partial response accumulation
- Response parsing
- Error recovery

### 3. Context Management
- Game state formatting
- Query type handling
- System prompts
- Conversation history

## Configuration
The system is configurable through a JSON configuration file:

```json
{
    "ollama": {
        "model": "llama2",
        "host": "http://localhost:11434",
        "context_window": 4096,
        "temperature": 0.7
    }
}
```

## Extension Points

### 1. Custom Prompts
Add new prompt types:
1. Create prompt templates
2. Define query formats
3. Add system prompts
4. Implement response parsing

### 2. Enhanced Response Processing
Extend response handling:
1. Add new parsing methods
2. Implement custom formatters
3. Create specialized handlers
4. Add validation rules

### 3. Context Management
Improve context handling:
1. Add new context types
2. Implement memory integration
3. Create context pruning strategies
4. Add state formatting rules

### 4. Stream Processing
Enhance streaming capabilities:
1. Add new stream handlers
2. Implement custom processors
3. Create specialized filters
4. Add monitoring systems

## Dependencies
- requests for HTTP communication
- json for data parsing
- threading for concurrent operations
- queue for response management
- datetime for timestamps
- Custom logger implementation

## Usage Example

```python
from game_ai.core.ollama_interface import OllamaInterface

# Initialize interface
ollama = OllamaInterface(config_path="config/ollama_config.json")

# Query model with streaming response
game_state = {...}  # Your game state
for response in ollama.query_model(game_state, "strategic"):
    if response['type'] == 'partial':
        print("Partial response:", response['content'])
    elif response['type'] == 'complete':
        print("Final response:", response['content'])

# Direct generation
response = ollama.generate("What should I do next?")
```

## Query Types

### Strategic Query
```python
response = ollama.query_model(game_state, "strategic")
# Focuses on long-term planning and resource management
```

### Tactical Query
```python
response = ollama.query_model(game_state, "tactical")
# Focuses on immediate action planning and execution
```

### Analysis Query
```python
response = ollama.query_model(game_state, "analysis")
# Focuses on analyzing current game state and patterns
```

## Best Practices
1. Proper error handling
2. Resource cleanup
3. Context management
4. Response validation
5. Stream processing

## Performance Considerations
- Response streaming efficiency
- Memory usage management
- Thread synchronization
- Queue management
- Context window limits

## Error Handling
The system implements comprehensive error handling for:
- Connection failures
- Response parsing errors
- Stream interruptions
- Configuration issues
- Thread management

## Response Processing

### Structured Response Format
```python
{
    'type': 'complete',
    'content': {
        'raw_response': 'Original response text',
        'structured_data': {
            'recommendations': [],
            'actions': [],
            'analysis': {}
        },
        'timestamp': '2024-01-01T12:00:00'
    }
}
```

### Streaming Response Format
```python
{
    'type': 'partial',
    'content': 'Chunk of response',
    'accumulated': 'Full accumulated response so far'
}
```

## Future Enhancements
1. Advanced caching
2. Response optimization
3. Context compression
4. Enhanced streaming
5. Pattern recognition
6. Performance monitoring
7. Custom model support
8. Advanced error recovery

## Integration Points

### 1. Decision Engine Integration
- Strategic planning
- Action generation
- State analysis
- Pattern recognition

### 2. Memory System Integration
- Context history
- Response caching
- State tracking
- Pattern learning

### 3. Vision System Integration
- State analysis
- Object recognition
- Environment understanding
- Action validation

## Safety Features
1. Connection timeout handling
2. Response validation
3. Error recovery
4. Resource cleanup
5. Thread safety

## Prompt System

### System Prompts
```python
prompts = {
    'strategic': """
Focus on long-term planning and resource management.
Consider objectives, threats, and opportunities.
Provide a prioritized list of strategic goals.""",
    
    'tactical': """
Focus on immediate action planning and execution.
Consider current threats and opportunities.
Provide specific, actionable steps.""",
    
    'analysis': """
Focus on analyzing the current game state.
Identify patterns, risks, and opportunities.
Provide detailed insights and recommendations."""
}
```

## Thread Management
- Background streaming thread
- Thread-safe queue operations
- Resource cleanup on shutdown
- Error recovery mechanisms
- State synchronization
