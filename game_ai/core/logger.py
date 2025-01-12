"""Centralized logging system for the Game AI."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json
import queue
from logging.handlers import RotatingFileHandler
import threading
from typing import Dict, Any

class GameAILogger:
    """Centralized logging system with console and file output."""
    
    _instance = None
    _lock = threading.Lock()
    _log_queue = queue.Queue()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the logging system.
        
        Args:
            config_path (Path, optional): Path to configuration file
        """
        # Skip if already initialized
        if GameAILogger._initialized:
            return
            
        with GameAILogger._lock:
            if GameAILogger._initialized:
                return
                
            self.config = self._load_config(config_path)
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            
            # Create base logger
            self.logger = logging.getLogger("GameAI")
            self.logger.setLevel(logging.DEBUG)
            
            # Ensure logger has error method
            self.error = self.logger.error
            self.info = self.logger.info
            self.warning = self.logger.warning
            self.debug = self.logger.debug
            
            # Create formatters
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.get('console_level', logging.INFO))
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            log_file = self.log_dir / f"game_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),  # 10MB default
                backupCount=self.config.get('backup_count', 5)
            )
            file_handler.setLevel(self.config.get('file_level', logging.DEBUG))
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Start log processing thread
            self._start_log_processor()
            
            GameAILogger._initialized = True
            
            self.logger.info("Logging system initialized")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load logging configuration."""
        default_config = {
            'console_level': logging.INFO,
            'file_level': logging.DEBUG,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'component_levels': {
                'vision': logging.INFO,
                'memory': logging.INFO,
                'decision': logging.INFO,
                'action': logging.INFO,
                'ollama': logging.INFO
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Convert string level names to logging constants
                    if 'console_level' in loaded_config:
                        loaded_config['console_level'] = getattr(logging, loaded_config['console_level'].upper())
                    if 'file_level' in loaded_config:
                        loaded_config['file_level'] = getattr(logging, loaded_config['file_level'].upper())
                    if 'component_levels' in loaded_config:
                        for component, level in loaded_config['component_levels'].items():
                            loaded_config['component_levels'][component] = getattr(logging, level.upper())
                    return {**default_config, **loaded_config}
            except Exception as e:
                print(f"Error loading logging config: {e}")
                return default_config
        return default_config
    
    def _start_log_processor(self) -> None:
        """Start the background log processing thread."""
        def process_logs():
            while True:
                try:
                    record = self._log_queue.get()
                    if record is None:  # Shutdown signal
                        break
                    self.logger.handle(record)
                except Exception as e:
                    print(f"Error processing log record: {e}")
                    
        self.processor_thread = threading.Thread(target=process_logs, daemon=True)
        self.processor_thread.start()
    
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """
        Get a logger for a specific component.
        
        Args:
            component_name (str): Name of the component
            
        Returns:
            logging.Logger: Logger instance for the component
        """
        logger = logging.getLogger(f"GameAI.{component_name}")
        level = self.config.get('component_levels', {}).get(component_name, logging.INFO)
        logger.setLevel(level)
        return logger
    
    def log_state_change(self, component: str, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> None:
        """
        Log state changes with detailed diff.
        
        Args:
            component (str): Component name
            old_state (dict): Previous state
            new_state (dict): New state
        """
        logger = self.get_component_logger(component)
        
        try:
            # Find differences
            changes = self._diff_states(old_state, new_state)
            if changes:
                logger.debug(f"State changes in {component}:")
                for key, (old_val, new_val) in changes.items():
                    logger.debug(f"  {key}: {old_val} -> {new_val}")
        except Exception as e:
            logger.error(f"Error logging state change: {e}")
    
    def log_performance(self, component: str, operation: str, duration: float) -> None:
        """
        Log performance metrics.
        
        Args:
            component (str): Component name
            operation (str): Operation being measured
            duration (float): Duration in seconds
        """
        logger = self.get_component_logger(component)
        logger.debug(f"Performance - {operation}: {duration:.3f}s")
    
    def log_error(self, component: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error with context.
        
        Args:
            component (str): Component name
            error (Exception): The error that occurred
            context (dict, optional): Additional context information
        """
        logger = self.get_component_logger(component)
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        logger.error(f"Error in {component}: {json.dumps(error_info, indent=2)}")
    
    def _diff_states(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, tuple]:
        """Compare old and new states to find differences."""
        changes = {}
        
        # Check all keys in new state
        for key in new_state:
            if key not in old_state:
                changes[key] = (None, new_state[key])
            elif old_state[key] != new_state[key]:
                changes[key] = (old_state[key], new_state[key])
        
        # Check for removed keys
        for key in old_state:
            if key not in new_state:
                changes[key] = (old_state[key], None)
        
        return changes
    
    def shutdown(self) -> None:
        """Cleanup and shutdown logging system."""
        try:
            # Signal processor thread to stop
            self._log_queue.put(None)
            
            # Wait for processor thread to finish
            if hasattr(self, 'processor_thread'):
                self.processor_thread.join(timeout=1.0)
            
            # Close all handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
                
        except Exception as e:
            print(f"Error during logger shutdown: {e}")
