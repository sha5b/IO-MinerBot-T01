# YOLO Training System for Minecraft

## Overview
This document outlines the process of training a YOLO model for Minecraft object detection using a combination of automated gameplay, Ollama-based classification, and continuous learning.

## Training Pipeline

### 1. Data Collection System

#### Automated Gameplay Recording
```python
class GameplayRecorder:
    def __init__(self):
        self.minecraft_bot = MinecraftBot()
        self.frame_buffer = []
        
    def record_session(self, duration):
        """Record gameplay session with screen captures"""
        while len(self.frame_buffer) < duration:
            frame = self.minecraft_bot.get_screen()
            self.frame_buffer.append({
                'frame': frame,
                'timestamp': time.time(),
                'bot_state': self.minecraft_bot.get_state()
            })
```

#### Ollama Classification
```python
class DataAnnotator:
    def __init__(self):
        self.ollama = OllamaInterface()
        
    def classify_frame(self, frame):
        """Get Ollama classifications for frame"""
        prompt = self.create_classification_prompt(frame)
        return self.ollama.classify(
            prompt,
            format='json',
            schema={
                'objects': [{
                    'class': 'string',
                    'confidence': 'float',
                    'bbox': 'array'
                }]
            }
        )
```

### 2. Dataset Preparation

#### YOLO Format Conversion
```python
def prepare_yolo_dataset(frames, annotations):
    """Convert frames and annotations to YOLO format"""
    for frame, annotation in zip(frames, annotations):
        # Save image
        image_path = f"training_data/images/{frame['timestamp']}.jpg"
        cv2.imwrite(image_path, frame['frame'])
        
        # Create label file
        label_path = f"training_data/labels/{frame['timestamp']}.txt"
        with open(label_path, 'w') as f:
            for obj in annotation['objects']:
                # Convert bbox to YOLO format (normalized)
                yolo_bbox = convert_to_yolo_format(obj['bbox'])
                # Write class_id x_center y_center width height
                f.write(f"{class_map[obj['class']]} {' '.join(map(str, yolo_bbox))}\n")
```

### 3. Training Configuration

#### Model Setup
```yaml
# yolo_config.yaml
path: training_data  # Dataset root directory
train: images/train  # Train images
val: images/val      # Validation images

# Classes
names:
  0: player
  1: zombie
  2: skeleton
  3: creeper
  4: tree
  5: grass
  6: stone
  7: dirt
  8: water
  9: lava

# Training parameters
epochs: 100
batch_size: 16
imgsz: 640
```

### 4. Training Loop

```python
from ultralytics import YOLO

def train_yolo():
    # Load base model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='yolo_config.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='minecraft_detector'
    )
    
    # Validate
    metrics = model.val()
    print(f"Validation mAP: {metrics.box.map}")
```

### 5. Continuous Learning Pipeline

```python
class ContinuousTraining:
    def __init__(self):
        self.recorder = GameplayRecorder()
        self.annotator = DataAnnotator()
        self.model = YOLO('models/best.pt')
        
    def training_loop(self):
        while True:
            # Collect new gameplay data
            frames = self.recorder.record_session(duration=1000)
            
            # Get Ollama classifications
            annotations = [
                self.annotator.classify_frame(frame)
                for frame in frames
            ]
            
            # Prepare new training data
            prepare_yolo_dataset(frames, annotations)
            
            # Fine-tune model
            self.model.train(
                data='yolo_config.yaml',
                epochs=10,  # Shorter epochs for fine-tuning
                imgsz=640,
                batch=16,
                name='minecraft_detector_continued'
            )
            
            # Evaluate and save if improved
            metrics = self.model.val()
            if metrics.box.map > self.best_map:
                self.model.save('models/best.pt')
                self.best_map = metrics.box.map
```

## Performance Monitoring

### Metrics Tracking
- Mean Average Precision (mAP)
- Inference speed (FPS)
- False positive/negative rates per class
- Resource utilization

### Visualization
```python
def visualize_detections(frame, detections):
    """Visualize YOLO detections on frame"""
    annotated_frame = frame.copy()
    for det in detections:
        # Draw bounding box
        cv2.rectangle(
            annotated_frame,
            (int(det.bbox[0]), int(det.bbox[1])),
            (int(det.bbox[2]), int(det.bbox[3])),
            (0, 255, 0),
            2
        )
        # Add label
        cv2.putText(
            annotated_frame,
            f"{det.class_name} {det.confidence:.2f}",
            (int(det.bbox[0]), int(det.bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return annotated_frame
```

## Integration with Main System

### Usage in Vision System
```python
class VisionSystem:
    def __init__(self):
        self.model = YOLO('models/best.pt')
        
    def process_frame(self, frame):
        # Run inference
        results = self.model(frame)
        
        # Process detections
        detections = []
        for r in results:
            for det in r.boxes.data:
                detections.append({
                    'class': r.names[int(det[5])],
                    'confidence': float(det[4]),
                    'bbox': det[:4].tolist()
                })
        
        return detections
```

## Best Practices

1. **Data Quality**
   - Ensure diverse gameplay scenarios
   - Balance class distributions
   - Verify Ollama classifications
   - Clean and validate annotations

2. **Training Process**
   - Start with small epochs for testing
   - Monitor validation metrics
   - Use early stopping
   - Save checkpoints regularly

3. **Performance Optimization**
   - Balance model size vs accuracy
   - Optimize inference pipeline
   - Use appropriate batch sizes
   - Consider hardware constraints

4. **Integration**
   - Graceful fallback mechanisms
   - Error handling
   - Performance monitoring
   - Version control for models
