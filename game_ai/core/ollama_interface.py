"""Interface for communicating with Ollama LLM service."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Union
import requests
from datetime import datetime
import threading
import queue
import time
from .logger import GameAILogger

class OllamaInterface:
    """Manages communication with Ollama LLM service."""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    _response_queue = queue.Queue()
    _stream_thread = None
    _keep_alive = True
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the Ollama interface.
        
        Args:
            config_path (Path, optional): Path to configuration file
        """
        if OllamaInterface._initialized:
            return
            
        with OllamaInterface._lock:
            if OllamaInterface._initialized:
                return
                
            self.logger = GameAILogger().get_component_logger("ollama")
            self.config = self._load_config(config_path)
            self.base_url = self.config.get('host', 'http://localhost:11434')
            self.model = self.config.get('model', 'llama2')
            self.context_window = self.config.get('context_window', 4096)
            self.conversation_history: List[Dict[str, Any]] = []
            
            # Initialize model connection
            self._initialize_model()
            
            OllamaInterface._initialized = True
            self.logger.info("Ollama interface initialized")
    
    def cleanup(self) -> None:
        """Cleanup resources and shutdown streaming thread."""
        try:
            self._keep_alive = False
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5.0)
            self.logger.info("Ollama interface cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """
        Load Ollama configuration.
        
        Args:
            config_path (Path): Path to config file
            
        Returns:
            dict: Configuration settings
        """
        default_config = {
            'model': 'llama2',
            'host': 'http://localhost:11434',
            'context_window': 4096,
            'temperature': 0.7
        }
        
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f).get('ollama', {})
                    return {**default_config, **loaded_config}
            return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return default_config
    
    def _initialize_model(self) -> None:
        """Initialize and verify model connection."""
        try:
            # Check if model is available
            url = f"{self.base_url}/api/show"
            response = requests.post(url, json={'name': self.model})
            response.raise_for_status()
            
            # Start streaming thread
            self._start_stream_thread()
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    def _start_stream_thread(self) -> None:
        """Start background thread for handling streaming responses."""
        def stream_processor():
            while self._keep_alive:
                try:
                    # Check for any pending requests
                    if not hasattr(self, '_current_request'):
                        time.sleep(0.1)
                        continue
                        
                    request = self._current_request
                    url = f"{self.base_url}/api/generate"
                    
                    with requests.post(url, json=request, stream=True) as response:
                        response.raise_for_status()
                        
                        accumulated_response = ""
                        for line in response.iter_lines():
                            if not line:
                                continue
                                
                            try:
                                chunk = json.loads(line)
                                response_part = chunk.get('response', '')
                                accumulated_response += response_part
                                
                                # Put partial response in queue
                                self._response_queue.put({
                                    'type': 'partial',
                                    'content': response_part,
                                    'accumulated': accumulated_response
                                })
                                
                                if chunk.get('done', False):
                                    # Put final response in queue
                                    self._response_queue.put({
                                        'type': 'complete',
                                        'content': accumulated_response
                                    })
                                    break
                                    
                            except json.JSONDecodeError:
                                self.logger.warning(f"Invalid JSON in stream: {line}")
                                
                    delattr(self, '_current_request')
                    
                except Exception as e:
                    self.logger.error(f"Error in stream processor: {e}")
                    time.sleep(1)  # Prevent rapid retries on error
                    
        self._stream_thread = threading.Thread(target=stream_processor, daemon=True)
        self._stream_thread.start()
    
    def query_model(self, game_state: Dict[str, Any], query_type: str) -> Generator[Dict[str, Any], None, None]:
        """
        Send game state to LLM and get streaming response.
        
        Args:
            game_state (dict): Current game state
            query_type (str): Type of query to make
            
        Yields:
            dict: Model response chunks
        """
        try:
            # Format context for the model
            context = self.format_context(game_state, query_type)
            
            # Prepare request
            self._current_request = {
                'model': self.model,
                'prompt': context,
                'temperature': self.config.get('temperature', 0.7),
                'max_tokens': self.context_window,
                'stream': True
            }
            
            # Process streaming response
            accumulated_response = ""
            while True:
                try:
                    response_chunk = self._response_queue.get(timeout=30)
                    
                    if response_chunk['type'] == 'partial':
                        accumulated_response = response_chunk['accumulated']
                        yield {
                            'type': 'partial',
                            'content': response_chunk['content'],
                            'accumulated': accumulated_response
                        }
                    elif response_chunk['type'] == 'complete':
                        # Process final response
                        result = self.process_response(response_chunk['content'])
                        
                        # Update conversation history
                        self._update_history(context, result)
                        
                        yield {
                            'type': 'complete',
                            'content': result
                        }
                        break
                        
                except queue.Empty:
                    self.logger.error("Timeout waiting for model response")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            yield {
                'type': 'error',
                'content': str(e)
            }
    
    def process_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process and parse model response.
        
        Args:
            response (dict): Raw model response
            
        Returns:
            dict: Processed response
        """
        try:
            # Parse structured data
            try:
                # Try to parse as JSON first
                structured_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fall back to text parsing if not JSON
                structured_data = self._parse_text_response(response_text)
            
            return {
                'raw_response': response_text,
                'structured_data': structured_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return {
                'raw_response': str(response),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def format_context(self, game_state: Dict[str, Any], query_type: str) -> str:
        """
        Format game state and query for model input.
        
        Args:
            game_state (dict): Current game state
            query_type (str): Type of query
            
        Returns:
            str: Formatted context string
        """
        try:
            # Build system prompt
            system_prompt = self._get_system_prompt(query_type)
            
            # Format game state
            state_str = json.dumps(game_state, indent=2)
            
            # Format query
            query = self._format_query(query_type, game_state)
            
            # Combine components
            context = f"{system_prompt}\n\nGame State:\n{state_str}\n\nQuery:\n{query}"
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error formatting context: {e}")
            return ""
    
    def update_prompts(self, new_prompts: Dict[str, str]) -> None:
        """
        Update system prompts.
        
        Args:
            new_prompts (dict): New prompt templates
        """
        try:
            if not hasattr(self, 'prompt_templates'):
                self.prompt_templates = {}
                
            self.prompt_templates.update(new_prompts)
            
        except Exception as e:
            self.logger.error(f"Error updating prompts: {e}")
    
    def _get_system_prompt(self, query_type: str) -> str:
        """Get appropriate system prompt for query type."""
        base_prompt = """You are an AI game agent assistant. Analyze the game state and provide strategic decisions.
Your responses should be structured and actionable."""
        
        prompts = {
            'strategic': base_prompt + """
Focus on long-term planning and resource management.
Consider objectives, threats, and opportunities.
Provide a prioritized list of strategic goals.""",
            
            'tactical': base_prompt + """
Focus on immediate action planning and execution.
Consider current threats and opportunities.
Provide specific, actionable steps.""",
            
            'analysis': base_prompt + """
Focus on analyzing the current game state.
Identify patterns, risks, and opportunities.
Provide detailed insights and recommendations."""
        }
        
        return prompts.get(query_type, base_prompt)
    
    def _format_query(self, query_type: str, game_state: Dict[str, Any]) -> str:
        """Format specific query based on type."""
        queries = {
            'strategic': "What are the optimal long-term strategic goals given the current game state?",
            'tactical': "What immediate actions should be taken in response to the current situation?",
            'analysis': "What are the key insights and patterns in the current game state?"
        }
        
        return queries.get(query_type, "Analyze the current game state and provide recommendations.")
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse unstructured text response into structured data."""
        structured_data = {
            'recommendations': [],
            'actions': [],
            'analysis': {}
        }
        
        try:
            # Split into sections
            sections = text.split('\n\n')
            
            for section in sections:
                if section.startswith('Recommendations:'):
                    structured_data['recommendations'] = [
                        rec.strip('- ') for rec in section.split('\n')[1:]
                        if rec.strip('- ')
                    ]
                elif section.startswith('Actions:'):
                    structured_data['actions'] = [
                        action.strip('- ') for action in section.split('\n')[1:]
                        if action.strip('- ')
                    ]
                elif section.startswith('Analysis:'):
                    analysis_lines = section.split('\n')[1:]
                    for line in analysis_lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            structured_data['analysis'][key.strip()] = value.strip()
                            
        except Exception as e:
            self.logger.error(f"Error parsing text response: {e}")
            
        return structured_data
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt (str): Input prompt for the model
            
        Returns:
            str: Model's response text
        """
        try:
            url = f"{self.base_url}/api/generate"
            self.logger.info(f"Sending prompt to LLM: {prompt}")
            
            response = requests.post(url, json={
                'model': self.model,
                'prompt': prompt,
                'temperature': self.config.get('temperature', 0.7),
                'stream': False
            })
            response.raise_for_status()
            result = response.json()
            response_text = result.get('response', '')
            
            self.logger.info(f"Raw LLM Response: {response_text}")
            return response_text
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return ""

    def _update_history(self, context: str, response: Dict[str, Any]) -> None:
        """Update conversation history."""
        try:
            # Add new interaction to history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'response': response
            })
            
            # Trim history if too long
            while len(str(self.conversation_history)) > self.context_window:
                self.conversation_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error updating history: {e}")
