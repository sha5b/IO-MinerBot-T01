"""
Autonomous Game AI System

A modular system for creating autonomous game-playing AI agents using computer vision,
machine learning, and large language models.

Core Components:
- Vision System: Screen capture and analysis
- Memory Manager: State and memory handling
- Decision Engine: AI decision making
- Action Controller: Game control
- Ollama Interface: LLM communication

The system uses a combination of computer vision (OpenCV, YOLOv8), machine learning (PyTorch),
and large language models (Ollama) to create an autonomous agent capable of playing games
through screen analysis and input simulation.
"""

from .core import (
    VisionSystem,
    MemoryManager,
    DecisionEngine,
    ActionController,
    OllamaInterface
)

__version__ = '0.1.0'
__author__ = 'AI Development Team'

__all__ = [
    'VisionSystem',
    'MemoryManager',
    'DecisionEngine',
    'ActionController',
    'OllamaInterface'
]
