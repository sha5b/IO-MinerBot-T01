# Vision System Documentation

## Overview
The Vision System is a core component responsible for capturing and analyzing the game screen. It provides real-time visual information processing, object detection, and environment analysis capabilities.

## Features
- Screen capture from specific windows or monitors
- YOLO-based object detection
- Environment analysis (terrain, lighting, obstacles)
- UI element detection
- Real-time visualization (optional)
- Player detection and tracking
- Modular design for easy extension

## Core Components

### 1. Screen Capture
- Supports both window-specific and monitor capture
- Automatic fallback mechanisms
- Configurable monitor selection

### 2. Object Detection
- YOLO-based detection system ([Training Documentation](yolo_training.md))
- Configurable confidence thresholds
- Support for multiple object classes
- Extensible for custom models
- Automated training pipeline with Ollama integration
- Continuous learning through bot-driven gameplay

### 3. Environment Analysis
- Terrain type detection
- Lighting condition analysis
- Obstacle detection
- Time of day estimation
- Resource identification

### 4. UI Analysis
- UI element detection
- Inventory tracking
- Interface element positioning

## Configuration
The system is highly configurable through a JSON configuration file:

```json
{
    "vision": {
        "monitor": 1,
        "window_title": null,
        "confidence_threshold": 0.5,
        "yolo": {
            "model_path": "models/yolov8n.pt",
            "confidence": 0.25,
            "iou": 0.45
        }
    }
}
```

## Extension Points

### 1. Custom Object Detection
To add new object detection capabilities:
1. Train or obtain a YOLO model for your specific needs
2. Update the model path in configuration
3. Add new object classes to the detection logic

### 2. Enhanced Environment Analysis
Add new environment analysis features by:
1. Creating new analysis methods in `_analyze_environment`
2. Adding new color ranges for terrain detection
3. Implementing custom feature extractors

### 3. UI Element Detection
Extend UI analysis by:
1. Adding new UI element types
2. Implementing custom detection algorithms
3. Enhancing the inventory tracking system

### 4. Vision Plugins
The system supports plugin-style extensions:
1. Create a new class that inherits from base analyzers
2. Implement the required interface methods
3. Register the plugin with the main VisionSystem

## Dependencies
- OpenCV (cv2)
- NumPy
- MSS (screen capture)
- Ultralytics YOLO
- Logging system

## Usage Example

```python
from game_ai.core.vision_system import VisionSystem

# Initialize the system
vision = VisionSystem(config_path="config/vision_config.json")

# Capture and analyze frame
frame = vision.capture_screen()
game_state = vision.process_frame(frame)

# Access analysis results
player_info = game_state['analysis']['player']
environment = game_state['analysis']['environment']
detected_objects = game_state['analysis']['objects']
```

## Best Practices
1. Regular model updates for improved detection
2. Calibrate confidence thresholds for your use case
3. Implement error handling for robust operation
4. Monitor system performance and adjust accordingly
5. Use visualization during development and debugging

## Performance Considerations
- Frame capture rate optimization
- YOLO model selection (speed vs accuracy)
- Efficient image processing pipeline
- Memory management for continuous operation

## Error Handling
The system implements comprehensive error handling:
- Capture failures
- Model loading issues
- Processing errors
- Configuration problems

## Future Enhancements
1. Multi-model support for specialized detection
2. Advanced terrain analysis
3. Dynamic confidence adjustment
4. Performance optimization
5. Enhanced visualization options
6. Integration with machine learning frameworks
7. Real-time performance metrics
8. Custom plugin system
