"""Test script to verify all components of the Terminus-001 Game AI System."""

import time
from pathlib import Path
from game_ai.core.logger import GameAILogger
from game_ai import (
    VisionSystem,
    MemoryManager,
    DecisionEngine,
    ActionController,
    OllamaInterface
)

def test_vision_system(logger):
    """Test vision system functionality."""
    try:
        logger.info("Testing Vision System...")
        vision = VisionSystem()
        
        # Test screen capture
        frame = vision.capture_screen()
        if frame is not None:
            logger.info("✓ Screen capture successful")
        else:
            logger.error("✗ Screen capture failed")
            return False
        
        # Test frame processing
        state = vision.process_frame(frame)
        if state is not None:
            logger.info("✓ Frame processing successful")
            logger.info(f"Detected objects: {len(state.get('objects', []))}")
        else:
            logger.error("✗ Frame processing failed")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Vision System test failed: {e}")
        return False

def test_memory_manager(logger):
    """Test memory management functionality."""
    try:
        logger.info("Testing Memory Manager...")
        memory = MemoryManager()
        
        # Test state storage
        test_state = {
            'player': {'health': 100, 'position': (0, 0)},
            'environment': {'objects': [], 'threats': []}
        }
        
        memory.update_current_state(test_state)
        logger.info("✓ State storage successful")
        
        # Test memory retrieval
        query = {'limit': 1}
        memories = memory.get_relevant_memory(query)
        if memories is not None:
            logger.info("✓ Memory retrieval successful")
        else:
            logger.error("✗ Memory retrieval failed")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Memory Manager test failed: {e}")
        return False

def test_decision_engine(logger):
    """Test decision engine functionality."""
    try:
        logger.info("Testing Decision Engine...")
        engine = DecisionEngine()
        
        # Test strategic planning
        test_state = {
            'player': {'health': 100, 'position': (0, 0)},
            'environment': {'threats': [], 'resources': []}
        }
        
        strategy = engine.strategic_planning(test_state, [])
        if strategy:
            logger.info("✓ Strategic planning successful")
            logger.info(f"Generated strategy: {strategy}")
        else:
            logger.error("✗ Strategic planning failed")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Decision Engine test failed: {e}")
        return False

def test_action_controller(logger):
    """Test action controller functionality."""
    try:
        logger.info("Testing Action Controller...")
        controller = ActionController()
        
        # Test action mapping
        test_action = {
            'type': 'movement',
            'direction': 'forward',
            'duration': 0.1
        }
        
        mapped_actions = controller.map_action(test_action)
        if mapped_actions:
            logger.info("✓ Action mapping successful")
        else:
            logger.error("✗ Action mapping failed")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Action Controller test failed: {e}")
        return False

def test_ollama_interface(logger):
    """Test Ollama interface functionality."""
    try:
        logger.info("Testing Ollama Interface...")
        ollama = OllamaInterface()
        
        # Test model query
        test_state = {
            'player': {'health': 100},
            'environment': {'threats': []}
        }
        
        logger.info("Testing LLM query (this may take a few seconds)...")
        response_generator = ollama.query_model(test_state, 'analysis')
        
        received_response = False
        for response in response_generator:
            if response['type'] == 'complete':
                received_response = True
                logger.info("✓ LLM query successful")
                logger.info(f"Sample response: {response['content'][:100]}...")
                break
        
        if not received_response:
            logger.error("✗ LLM query failed")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Ollama Interface test failed: {e}")
        return False

def main():
    """Run system component tests."""
    # Initialize logging
    logger = GameAILogger().get_component_logger("test")
    logger.info("Starting system tests...")
    
    # Create required directories
    for dir_path in ["logs", "memory/current_state", "memory/short_term", "memory/long_term"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tests = [
        ("Vision System", test_vision_system),
        ("Memory Manager", test_memory_manager),
        ("Decision Engine", test_decision_engine),
        ("Action Controller", test_action_controller),
        ("Ollama Interface", test_ollama_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} Testing {test_name} {'='*20}")
        try:
            success = test_func(logger)
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            results.append((test_name, False))
        time.sleep(1)  # Brief pause between tests
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary:")
    logger.info("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:.<30} {status}")
        if not success:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
