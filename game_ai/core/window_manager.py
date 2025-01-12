"""Window management utilities for capturing specific application windows."""

import win32gui
import win32con
import win32ui
import numpy as np
import time
import cv2
from mss import mss
from typing import Dict, List, Tuple, Optional
import logging

class WindowManager:
    """Manages window enumeration and capture operations."""
    
    def __init__(self):
        """Initialize the window manager."""
        self.logger = logging.getLogger(__name__)
        self.windows = {}
        self.active_window = None
        self.refresh_window_list()

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

    def get_window_list(self) -> List[str]:
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

    def get_window_rect(self, hwnd: Optional[int] = None) -> Tuple[int, int, int, int]:
        """
        Get window rectangle coordinates.
        
        Args:
            hwnd (int, optional): Window handle, uses active window if None
            
        Returns:
            tuple: (left, top, right, bottom) coordinates
        """
        hwnd = hwnd or self.active_window
        if not hwnd:
            raise ValueError("No active window set")
        return win32gui.GetWindowRect(hwnd)

    def capture_window(self, hwnd: Optional[int] = None) -> np.ndarray:
        """
        Capture specific window contents using mss.
        
        Args:
            hwnd (int, optional): Window handle, uses active window if None
            
        Returns:
            numpy.ndarray: Captured window content as BGR image
        """
        hwnd = hwnd or self.active_window
        if not hwnd:
            raise ValueError("No active window set")

        try:
            # Get window coordinates
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            
            # Get client area offset
            client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))
            _, _, client_right, client_bottom = win32gui.GetClientRect(hwnd)
            
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
            with mss() as sct:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                # Convert from BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame

        except Exception as e:
            self.logger.error(f"Window capture error: {e}")
            raise

    def is_window_valid(self, hwnd: Optional[int] = None) -> bool:
        """
        Check if window is still valid.
        
        Args:
            hwnd (int, optional): Window handle to check, uses active window if None
            
        Returns:
            bool: True if window exists and is visible
        """
        hwnd = hwnd or self.active_window
        if not hwnd:
            return False
        try:
            return win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)
        except:
            return False

    def get_window_title(self, hwnd: Optional[int] = None) -> str:
        """
        Get window title.
        
        Args:
            hwnd (int, optional): Window handle, uses active window if None
            
        Returns:
            str: Window title
        """
        hwnd = hwnd or self.active_window
        if not hwnd:
            raise ValueError("No active window set")
        return win32gui.GetWindowText(hwnd)
