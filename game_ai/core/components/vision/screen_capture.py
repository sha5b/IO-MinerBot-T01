"""Screen capture component for capturing game window or monitor."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import win32gui
import win32con
import win32ui
import numpy as np
import time
import cv2
from mss import mss
from pathlib import Path

class ScreenCapture:
    """Handles screen capture from game window or monitor."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize screen capture.
        
        Args:
            config (dict): Configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize screen capture
        self.sct = mss()
        self.active_window = None
        self.windows = {}
        self.refresh_window_list()
        
        # Capture settings
        self.target_fps = config.get('capture', {}).get('fps', 30)
        self.frame_time = 1.0 / self.target_fps
        self.last_capture = 0
        
    def refresh_window_list(self) -> Dict[str, int]:
        """
        Refresh the list of available windows.
        
        Returns:
            dict: Dictionary of window titles and their handles
        """
        def enum_window_callback(hwnd: int, windows: dict) -> bool:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and not title.startswith("_"):  # Filter out system windows
                    windows[title] = hwnd
            return True
        
        self.windows.clear()
        win32gui.EnumWindows(enum_window_callback, self.windows)
        return self.windows
    
    def get_available_windows(self) -> List[str]:
        """
        Get list of available window titles.
        
        Returns:
            list: List of window titles
        """
        return list(self.windows.keys())
    
    def set_active_window(self, title: str) -> bool:
        """
        Set the active window for capture.
        
        Args:
            title (str): Window title to capture
            
        Returns:
            bool: True if window was found and set
        """
        if title in self.windows:
            self.active_window = self.windows[title]
            return True
        return False
    
    def set_capture_window(self, title: str) -> bool:
        """
        Alias for set_active_window for backward compatibility.
        
        Args:
            title (str): Window title to capture
            
        Returns:
            bool: True if window was found and set
        """
        return self.set_active_window(title)
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture current frame based on settings.
        
        Returns:
            numpy.ndarray: Captured frame as BGR image or None if capture failed
        """
        try:
            # Respect frame rate limit
            current_time = time.time()
            if current_time - self.last_capture < self.frame_time:
                time.sleep(self.frame_time - (current_time - self.last_capture))
            
            # Capture frame
            if self.active_window:
                frame = self._capture_window()
            else:
                frame = self._capture_monitor()
            
            self.last_capture = time.time()
            return frame
            
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    def _capture_window(self) -> Optional[np.ndarray]:
        """Capture active window content."""
        try:
            # Get window coordinates
            left, top, right, bottom = win32gui.GetWindowRect(self.active_window)
            
            # Get client area offset
            client_left, client_top = win32gui.ClientToScreen(self.active_window, (0, 0))
            _, _, client_right, client_bottom = win32gui.GetClientRect(self.active_window)
            
            # Calculate actual client area in screen coordinates
            width = client_right
            height = client_bottom
            
            # Define capture region
            monitor = {
                "top": client_top,
                "left": client_left,
                "width": width,
                "height": height,
                "mon": 0  # Capture from all monitors
            }
            
            # Capture using mss
            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            # Convert from BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
            
        except Exception as e:
            self.logger.error(f"Window capture error: {e}")
            return None
    
    def _capture_monitor(self) -> Optional[np.ndarray]:
        """Capture monitor screen."""
        try:
            monitor_num = self.config.get('capture', {}).get('monitor', 1)
            if monitor_num >= len(self.sct.monitors):
                monitor_num = 1
                
            monitor = self.sct.monitors[monitor_num]
            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
        except Exception as e:
            self.logger.error(f"Monitor capture error: {e}")
            return None
    
    def get_window_title(self) -> Optional[str]:
        """
        Get active window title.
        
        Returns:
            str: Window title or None if no active window
        """
        if self.active_window:
            return win32gui.GetWindowText(self.active_window)
        return None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.sct.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
