# Terminus-001: Modular Game AI System

## Overview
Terminus-001 is a Python-based modular game AI system designed to provide intelligent automation for games. Built using a virtual environment (venv) for isolation and dependency management, the system focuses on modularity and easy extension of each component. It combines computer vision (powered by YOLO), memory management, decision making, and action execution to create an autonomous agent.

### Initial Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Steps
1. **YOLO Training Setup**
   The first major component to implement is the vision system's YOLO model for Minecraft object detection. This involves:
   - Setting up automated gameplay recording
   - Implementing Ollama-based classification for training data
   - Training the YOLO model on Minecraft objects
   - Continuous improvement through bot-driven gameplay
   
   See [YOLO Training Documentation](docs/yolo_training.md) for detailed implementation.

2. **Vision System Integration**
   After training the YOLO model:
   - Integrate model with real-time screen capture
   - Implement object tracking and state management
   - Set up performance monitoring

## Core Systems

### 1. Vision System
The Vision System handles all visual input processing and analysis. It provides:
- Screen capture capabilities
- Object detection using YOLO
- Environment analysis
- UI element detection
- Real-time visualization

[Documentation](docs/vision_system.md)

### 2. Memory System
The Memory System manages state information across different time scales:
- Current state tracking
- Short-term memory
- Long-term memory
- Query-based retrieval
- Persistent storage

[Documentation](docs/memory_system.md)

### 3. Decision Engine
The Decision Engine handles strategic planning and action selection:
- Strategic planning
- Tactical planning
- Reactive responses
- LLM integration
- Priority-based decisions

[Documentation](docs/decision_engine.md)

### 4. Action Controller
The Action Controller executes game actions through input simulation:
- Input simulation
- Control mapping
- Action validation
- Timing management
- Error handling

[Documentation](docs/action_controller.md)

### 5. Ollama Interface
The Ollama Interface provides LLM capabilities:
- LLM communication
- Streaming responses
- Context management
- Response parsing
- Thread safety

[Documentation](docs/ollama_interface.md)

## Building the System Modularly

### Step 1: Basic Setup
1. Create project structure:
```
game_ai/
├── __init__.py
├── config/
│   ├── control_maps.json
│   ├── game_rules.json
│   └── system_config.json
├── core/
│   ├── __init__.py
│   ├── vision_system.py
│   ├── memory_manager.py
│   ├── decision_engine.py
│   ├── action_controller.py
│   └── ollama_interface.py
└── core/components/
    ├── vision/
    ├── memory/
    └── decision/
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: System Implementation Order

1. **Vision System First**
   - Implement basic screen capture
   - Add object detection
   - Develop environment analysis
   - Add UI detection

2. **Memory System Second**
   - Set up storage structure
   - Implement state tracking
   - Add query capabilities
   - Enable persistence

3. **Decision Engine Third**
   - Create basic decision framework
   - Add strategic planning
   - Implement tactical planning
   - Add reactive responses

4. **Action Controller Fourth**
   - Set up input simulation
   - Implement control mapping
   - Add action validation
   - Enable timing management

5. **Ollama Interface Last**
   - Set up LLM connection
   - Add streaming support
   - Implement context management
   - Enable response parsing

### Step 3: Integration

1. **Vision → Memory**
   - Pass visual data to memory
   - Store state information
   - Track object history

2. **Memory → Decision**
   - Provide state context
   - Enable pattern recognition
   - Support planning

3. **Decision → Action**
   - Convert decisions to actions
   - Handle timing
   - Manage execution

4. **Ollama → Decision**
   - Enhance planning
   - Improve analysis
   - Enable adaptation

## Extension Points

Each system provides clear extension points for adding new capabilities:

### Vision Extensions
- Custom object detection
- New analysis methods
- Enhanced UI detection
- Specialized visualizations

### Memory Extensions
- New storage types
- Custom query methods
- Enhanced persistence
- Pattern recognition

### Decision Extensions
- New planning strategies
- Custom objectives
- Enhanced reactions
- Specialized behaviors

### Action Extensions
- New input methods
- Custom controls
- Enhanced validation
- Macro support

### Ollama Extensions
- Custom prompts
- New response types
- Enhanced parsing
- Specialized behaviors

## Configuration

Each system is configured through JSON files in the config directory:

```json
{
    "vision": {
        "model_path": "models/yolov8n.pt",
        "confidence": 0.25
    },
    "memory": {
        "short_term_limit": 100,
        "long_term_limit": 1000
    },
    "decision": {
        "planning_interval": 5.0,
        "reaction_threshold": 0.8
    },
    "controls": {
        "input_delay": 0.05,
        "action_timeout": 5.0
    },
    "ollama": {
        "model": "llama2",
        "temperature": 0.7
    }
}
```

## Usage

1. Basic setup:
```python
from game_ai.core import VisionSystem, MemoryManager, DecisionEngine, ActionController

# Initialize systems
vision = VisionSystem()
memory = MemoryManager()
decision = DecisionEngine()
action = ActionController()

# Main loop
while True:
    # Get visual input
    game_state = vision.process_frame(vision.capture_screen())
    
    # Update memory
    memory.update_current_state(game_state)
    
    # Make decisions
    strategy = decision.strategic_planning(game_state, memory.get_relevant_memory({}))
    actions = decision.tactical_planning(strategy, game_state)
    
    # Execute actions
    for action_item in actions:
        action.execute_action(action_item)
```

## Best Practices

1. **Modular Development**
   - Develop each system independently
   - Use clear interfaces
   - Maintain separation of concerns
   - Enable easy testing

2. **Error Handling**
   - Implement comprehensive error handling
   - Use logging effectively
   - Enable recovery mechanisms
   - Maintain system stability

3. **Performance**
   - Monitor resource usage
   - Optimize critical paths
   - Handle timing properly
   - Manage memory efficiently

4. **Extension**
   - Follow extension patterns
   - Document new features
   - Maintain compatibility
   - Enable configuration

## Vision System Training Pipeline

### 1. Environment Setup
1. **Python Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Minecraft Setup**
   - Install Minecraft Java Edition
   - Set up development environment
   - Configure screen resolution for consistent capture

### 2. Data Collection System
1. **Automated Gameplay Recording**
   - Implement Minecraft bot control system
   - Record gameplay sessions with screen captures
   - Store frame sequences with timestamps

2. **Data Annotation Pipeline**
   - Use Ollama for initial object classification
   - Store classifications in structured format
   - Enable manual verification interface

### 3. YOLO Training Process
1. **Dataset Preparation**
   - Convert Minecraft screenshots to YOLO format
   - Generate label files from Ollama classifications
   - Split data into train/val/test sets

2. **Model Configuration**
   - Start with YOLOv8n base model
   - Customize for Minecraft objects
   - Configure hyperparameters

3. **Training Loop**
   ```python
   # Training pipeline pseudocode
   def training_pipeline():
       # Initialize systems
       minecraft_bot = MinecraftBot()
       ollama = OllamaInterface()
       
       while training_active:
           # Collect gameplay data
           frames = minecraft_bot.play_and_record()
           
           # Get Ollama classifications
           labels = ollama.classify_frames(frames)
           
           # Prepare YOLO dataset
           dataset = prepare_yolo_dataset(frames, labels)
           
           # Train YOLO model
           train_yolo(dataset, model_config)
           
           # Validate and adjust
           evaluate_model()
   ```

### 4. Continuous Improvement
1. **Feedback Loop**
   - Monitor model performance
   - Collect misclassification examples
   - Retrain with expanded dataset

2. **Integration Testing**
   - Test in various Minecraft environments
   - Validate real-time performance
   - Optimize inference speed

3. **Performance Metrics**
   - Track detection accuracy
   - Measure inference time
   - Monitor resource usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add documentation
5. Submit a pull request

## License

MIT License - See LICENSE file for details
