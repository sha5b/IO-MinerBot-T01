"""Main entry point for the Terminus-001 Game AI System."""

import sys
import time
import json
import requests
import subprocess
from pathlib import Path
import logging
from datetime import datetime, timedelta
from game_ai.core.logger import GameAILogger
from game_ai import (
    VisionSystem,
    MemoryManager,
    DecisionEngine,
    ActionController,
    OllamaInterface
)

def check_ollama() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False
    return False

def start_ollama() -> bool:
    """Attempt to start Ollama service."""
    try:
        # Try to start Ollama based on OS
        if sys.platform == "win32":
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["ollama", "serve"])
        
        # Wait for Ollama to start
        for _ in range(30):  # 30 second timeout
            if check_ollama():
                return True
            time.sleep(1)
            
    except Exception as e:
        print(f"Error starting Ollama: {e}")
    return False

def check_model(model_name: str = "llama2") -> bool:
    """Check if required model is available."""
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name}
        )
        return response.status_code == 200
    except:
        return False

def pull_model(model_name: str = "llama2") -> bool:
    """Pull required model if not available."""
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")
        return False

def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    try:
        import torch
        import cv2
        import ultralytics
        import mss
        import pynput
        import win32gui  # Required for window capture
        import win32con
        import win32ui
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        if "win32" in str(e):
            print("Please install pywin32: pip install pywin32")
        return False

def download_yolo_model(logger):
    """Download YOLOv8 model if not present."""
    try:
        from ultralytics import YOLO
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "yolov8n.pt"
        if not model_path.exists():
            logger.info("Downloading YOLOv8 model...")
            model = YOLO("yolov8n.pt")  # This will download the model
            model.save(str(model_path))
            logger.info("YOLOv8 model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading YOLO model: {e}")
        return False

def select_game_type(logger: logging.Logger) -> str:
    """
    Allow user to select game type.
    
    Args:
        logger (logging.Logger): Logger instance
        
    Returns:
        str: Selected game type
    """
    try:
        # Load control maps to get available games
        with open('game_ai/config/control_maps.json', 'r') as f:
            control_maps = json.load(f)
            
        available_games = list(control_maps.get('games', {}).keys())
        if not available_games:
            logger.error("No game types found in control maps")
            return "default"
            
        # Print available games
        print("\nAvailable game types:")
        for i, game in enumerate(available_games):
            print(f"{i + 1}. {game}")
            
        # Get user selection
        while True:
            try:
                choice = input("\nSelect game type number: ")
                choice = int(choice)
                
                if 1 <= choice <= len(available_games):
                    selected_game = available_games[choice - 1]
                    logger.info(f"Selected game type: {selected_game}")
                    return selected_game
                    
                print("Invalid selection. Please try again.")
                
            except ValueError:
                print("Please enter a valid number")
                
    except Exception as e:
        logger.error(f"Error during game type selection: {e}")
        return "default"

def initialize_components(logger: GameAILogger) -> tuple:
    """Initialize all system components."""
    try:
        # Download YOLO model if needed
        if not download_yolo_model(logger):
            raise RuntimeError("Failed to download YOLO model")
            
        # Select game type
        main_logger = logger.get_component_logger("main")
        game_type = select_game_type(main_logger)
            
        # Initialize components
        ollama = OllamaInterface()  # Initialize Ollama first
        vision = VisionSystem()
        memory = MemoryManager()
        decision = DecisionEngine(ollama_interface=ollama)  # Pass Ollama to DecisionEngine
        action = ActionController(game_type=game_type)  # Pass game type to ActionController
        
        logger.info("All components initialized successfully")
        return vision, memory, decision, action, ollama
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

def select_window(vision: VisionSystem, logger: logging.Logger) -> bool:
    """
    Allow user to select which window to capture.
    
    Args:
        vision (VisionSystem): Initialized vision system
        logger (logging.Logger): Logger instance
        
    Returns:
        bool: True if window was selected successfully
    """
    try:
        # Get list of available windows
        windows = vision.get_available_windows()
        if not windows:
            logger.error("No available windows found")
            return False
            
        # Print available windows
        print("\nAvailable windows:")
        for i, title in enumerate(windows):
            print(f"{i + 1}. {title}")
        print("0. Use full monitor capture")
        
        # Get user selection
        while True:
            try:
                choice = input("\nSelect window number (or 0 for monitor capture): ")
                choice = int(choice)
                
                if choice == 0:
                    # Use default monitor capture
                    vision.set_capture_monitor(1)  # Default to primary monitor
                    logger.info("Using monitor capture mode")
                    return True
                    
                if 1 <= choice <= len(windows):
                    # Set selected window
                    selected_title = windows[choice - 1]
                    if vision.set_capture_window(selected_title):
                        logger.info(f"Selected window: {selected_title}")
                        return True
                    else:
                        logger.error(f"Failed to set window: {selected_title}")
                        return False
                        
                print("Invalid selection. Please try again.")
                
            except ValueError:
                print("Please enter a valid number")
                
    except Exception as e:
        logger.error(f"Error during window selection: {e}")
        return False

def main():
    """Main entry point for the system."""
    # Initialize logging
    logger = GameAILogger()
    main_logger = logger.get_component_logger("main")
    main_logger.info("Starting Terminus-001 Game AI System")
    
    # Check dependencies
    main_logger.info("Checking dependencies...")
    if not check_dependencies():
        main_logger.error("Missing required dependencies")
        return
    
    # Check Ollama status
    main_logger.info("Checking Ollama service...")
    if not check_ollama():
        main_logger.info("Ollama not running, attempting to start...")
        if not start_ollama():
            main_logger.error("Failed to start Ollama service")
            return
    
    # Check model availability
    main_logger.info("Checking model availability...")
    if not check_model():
        main_logger.info("Model not found, pulling...")
        if not pull_model():
            main_logger.error("Failed to pull required model")
            return
    
    try:
        # Initialize components
        main_logger.info("Initializing system components...")
        vision, memory, decision, action, ollama = initialize_components(logger)
        
        # Create required directories
        main_logger.info("Setting up directory structure...")
        for dir_path in ["logs", "memory/current_state", "memory/short_term", "memory/long_term"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        main_logger.info("System initialization complete")
        
        # Select window for capture
        main_logger.info("Setting up window capture...")
        if not select_window(vision, main_logger):
            main_logger.error("Failed to set up window capture")
            return
            
        # Main game interaction loop
        main_logger.info("Starting main interaction loop...")
        last_strategy_time = datetime.now()
        strategy_interval = timedelta(seconds=5)  # Update strategy every 5 seconds
        
        # Initialize first strategy with retries
        max_retries = 3
        retry_count = 0
        strategy = None
        current_objective = "Initializing..."
        
        while retry_count < max_retries:
            try:
                frame = vision.capture_screen()
                if frame is None:
                    main_logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.5)
                    retry_count += 1
                    continue
                
                state = vision.process_frame(frame, current_objective)
                memory.update_current_state(state)
                memory_query = {
                    'limit': 10,
                    'time_range': [datetime.now() - timedelta(minutes=5), datetime.now()]
                }
                strategy = decision.strategic_planning(state, memory.get_relevant_memory(memory_query))
                if strategy and strategy.get('objectives'):
                    current_objective = decision._get_current_objective()
                    main_logger.info(f"Initial Objective: {current_objective}")
                    break
                else:
                    main_logger.warning("Failed to generate valid strategy, retrying...")
                    time.sleep(0.5)
                    retry_count += 1
            except Exception as e:
                main_logger.error(f"Error during initialization: {e}")
                time.sleep(0.5)
                retry_count += 1
        
        if not strategy or not strategy.get('objectives'):
            main_logger.error("Failed to initialize strategy after retries")
            return
        
        while True:
            try:
                try:
                    # Capture and process current game state
                    frame = vision.capture_screen()
                    if frame is None:
                        main_logger.warning("Failed to capture frame, retrying...")
                        time.sleep(0.1)
                        continue
                        
                    state = vision.process_frame(frame, current_objective)
                    memory.update_current_state(state)
                    
                    current_time = datetime.now()
                    
                    # Update strategy periodically
                    if current_time - last_strategy_time > strategy_interval:
                        memory_query = {
                            'limit': 10,
                            'time_range': [current_time - timedelta(minutes=5), current_time]
                        }
                        strategy = decision.strategic_planning(state, memory.get_relevant_memory(memory_query))
                        last_strategy_time = current_time
                        current_objective = decision._get_current_objective()
                        main_logger.info(f"Current Objective: {current_objective}")
                    
                    # Check if we have a valid strategy
                    if not strategy:
                        main_logger.warning("No valid strategy, replanning...")
                        continue
                    
                    # Get and execute actions
                    actions = decision.tactical_planning(strategy, state)
                    if not actions:
                        main_logger.debug("No actions to execute")
                        time.sleep(0.1)
                        continue
                    
                    # Execute planned actions
                    for action_item in actions:
                        if not action.execute_action(action_item):
                            main_logger.warning(f"Failed to execute action: {action_item}")
                            continue
                        # Brief pause between actions
                        time.sleep(float(action_item.get('duration', 0.1)))
                    
                    # Brief pause to prevent excessive CPU usage
                    time.sleep(0.05)
                    
                except Exception as e:
                    main_logger.error(f"Error in game loop iteration: {e}")
                    time.sleep(0.1)  # Brief pause on error before retrying
                
            except Exception as e:
                main_logger.error(f"Error in game loop: {e}")
                time.sleep(1)  # Longer pause on error
        
    except KeyboardInterrupt:
        main_logger.info("Shutting down gracefully...")
    except Exception as e:
        main_logger.error(f"Critical error: {e}")
    finally:
        # Cleanup
        try:
            vision.cleanup()  # Clean up visualization
            ollama.cleanup()
            logger.shutdown()
        except Exception as e:
            main_logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
