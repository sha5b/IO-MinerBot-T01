# Memory Management System Documentation

## Overview
The Memory Management System is responsible for storing, organizing, and retrieving game state information across different time scales. It implements a three-tier memory architecture: current state, short-term memory, and long-term memory.

## Features
- Three-tier memory architecture
- Persistent storage
- Configurable memory limits
- Query-based memory retrieval
- Automatic cleanup
- JSON-based storage format
- Time-based organization

## Core Components

### 1. Memory Tiers

#### Current State
- Holds the most recent game state
- Instantly accessible
- Persisted to disk for recovery
- Located in `memory/current_state/latest.json`

#### Short-Term Memory
- Recent state history
- Configurable size limit
- In-memory access for speed
- Persisted to `memory/short_term/memory.json`

#### Long-Term Memory
- Historical state archive
- Date-based organization
- Disk-based storage
- Automatic cleanup
- Located in `memory/long_term/{date}.json`

## Configuration
The system is configurable through a JSON configuration file:

```json
{
    "memory": {
        "short_term_limit": 100,
        "long_term_limit": 1000,
        "cleanup_interval": 3600
    }
}
```

## Extension Points

### 1. Custom Memory Tiers
Add new memory tiers by:
1. Creating a new storage class
2. Implementing the storage interface
3. Adding configuration options
4. Integrating with the main MemoryManager

### 2. Enhanced Query System
Extend the query capabilities by:
1. Adding new query parameters
2. Implementing custom matching logic
3. Creating specialized search functions

### 3. Memory Analytics
Add analytics capabilities:
1. Implement memory usage tracking
2. Add pattern recognition
3. Create memory optimization algorithms

### 4. Storage Plugins
Create custom storage backends:
1. Implement the storage interface
2. Add new persistence methods
3. Create custom cleanup strategies

## Dependencies
- Python's built-in json module
- pathlib for file operations
- datetime for temporal operations
- logging system

## Usage Example

```python
from game_ai.core.memory_manager import MemoryManager

# Initialize the system
memory = MemoryManager(config_path="config/memory_config.json")

# Store current state
current_state = {
    "player": {"position": [100, 200]},
    "inventory": {"wood": 5, "stone": 3}
}
memory.update_current_state(current_state)

# Query memory
query = {
    "time_range": [start_time, end_time],
    "limit": 10
}
relevant_memories = memory.get_relevant_memory(query)
```

## Best Practices
1. Regular cleanup scheduling
2. Efficient query design
3. Proper error handling
4. Memory limit optimization
5. Regular persistence checks

## Performance Considerations
- Memory usage monitoring
- Storage space management
- Query optimization
- Cleanup timing
- File I/O efficiency

## Error Handling
The system implements comprehensive error handling for:
- File operations
- Memory limits
- Query processing
- Configuration loading
- Data persistence

## Directory Structure
```
memory/
├── current_state/
│   └── latest.json
├── short_term/
│   └── memory.json
└── long_term/
    ├── 2024-01-01.json
    ├── 2024-01-02.json
    └── ...
```

## Future Enhancements
1. Memory compression
2. Advanced querying capabilities
3. Memory indexing
4. Pattern recognition
5. Memory optimization
6. Database integration
7. Memory analytics
8. Custom storage backends

## Memory Query System
The query system supports:
- Time range filtering
- State parameter matching
- Result limiting
- Custom query parameters

### Query Examples
```python
# Time-based query
query = {
    "time_range": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    "limit": 5
}

# State parameter query
query = {
    "player.position": [100, 200],
    "inventory.wood": 5,
    "limit": 10
}
```

## Memory Cleanup
- Automatic cleanup based on configured interval
- Removes oldest long-term memories first
- Configurable retention policies
- Maintains system performance

## Integration Points
1. Vision System Integration
   - Store visual analysis results
   - Track object detection history
   - Maintain environment state

2. Decision Engine Integration
   - Access historical decisions
   - Store action outcomes
   - Track performance metrics

3. Action Controller Integration
   - Store action history
   - Track command sequences
   - Maintain state changes
