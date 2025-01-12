# Terminus-001: Autonomous Game AI System

A sophisticated AI system designed to play games autonomously through computer vision, machine learning, and large language models. The system uses screen analysis and input simulation to interact with games, making decisions based on real-time visual input and strategic planning.

## Core Components

### Vision System
- Screen capture and analysis using OpenCV
- Object detection with YOLOv8
- Real-time environment analysis
- UI element detection

### Memory Manager
- State persistence and tracking
- Short-term and long-term memory management
- Efficient memory organization and cleanup
- JSON-based storage system

### Decision Engine
- Strategic long-term planning
- Tactical decision making
- Reactive responses
- Priority-based action selection

### Action Controller
- Input simulation (keyboard and mouse)
- Action mapping and validation
- Error handling and recovery
- Complex input combinations

### Ollama Interface
- Streaming LLM communication
- Context-aware prompting
- Response parsing and structuring
- Conversation history management

### Logging System
- Comprehensive logging with rotation
- Component-specific logging levels
- Performance metrics tracking
- State change monitoring

## Project Structure

```
game_ai/
├── config/
│   ├── system_config.json     # System configuration
│   ├── game_rules.json       # Game-specific rules
│   └── control_maps.json     # Input mappings
├── core/
│   ├── vision_system.py      # Screen capture and analysis
│   ├── memory_manager.py     # State and memory handling
│   ├── decision_engine.py    # AI decision making
│   ├── action_controller.py  # Game control
│   ├── ollama_interface.py   # LLM communication
│   └── logger.py            # Logging system
└── memory/                   # JSON-based storage
    ├── current_state/       # Current game state
    ├── short_term/         # Recent events/states
    └── long_term/          # Historical data
```

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- YOLOv8
- MSS (screen capture)
- Pynput (input simulation)
- Ollama (LLM integration)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio opencv-python ultralytics mss pynput ollama
```

3. Configure Ollama:
- Install Ollama from [ollama.ai](https://ollama.ai)
- Pull the required model:
```bash
ollama pull llama2
```

## Configuration

### System Configuration
Edit `config/system_config.json` to configure:
- Vision system parameters
- Memory limits
- Ollama settings
- Input delays

### Game Rules
Edit `config/game_rules.json` to define:
- Game objectives
- Priority levels
- Constraints
- Behavior patterns

### Control Mappings
Edit `config/control_maps.json` to configure:
- Keyboard mappings
- Mouse settings
- Input combinations

## Features

- **Real-time Vision Processing**: Analyzes game screen in real-time using computer vision and object detection
- **Adaptive Decision Making**: Uses LLM for strategic planning and tactical decisions
- **Memory Management**: Maintains short and long-term memory for improved decision making
- **Input Simulation**: Precise keyboard and mouse control for game interaction
- **Streaming LLM Integration**: Efficient communication with Ollama for continuous feedback
- **Comprehensive Logging**: Detailed logging system for debugging and analysis

## Error Handling

The system includes robust error handling for:
- Vision system failures
- Memory corruption
- Decision timeouts
- Action execution failures
- LLM communication issues

## Cross-Platform Support

- Uses pathlib for platform-agnostic path handling
- Platform-independent input simulation
- Configurable for different operating systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- YOLOv8 for object detection
- Ollama for LLM integration
- OpenCV for computer vision
- PyTorch for machine learning capabilities
