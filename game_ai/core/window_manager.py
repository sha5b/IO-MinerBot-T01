"""Window manager for controlling game window and mouse input."""

import win32gui
import win32con
import win32api
import win32ui
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

class WindowManager:
    """Manages game window focus and mouse input."""
    
    def __init__(self, window_title: str = "Minecraft"):
        """
        Initialize window manager.
        
        Args:
            window_title (str): Title of game window to control
        """
        self.logger = logging.getLogger(__name__)
        self.window_title = window_title
        self.game_hwnd = None
        self.window_rect = None
        self.center_pos = None
        self.is_focused = False
        self.active_window = False  # Track if we have an active window
        
    def find_game_window(self) -> bool:
        """Find game window by title."""
        try:
            self.game_hwnd = win32gui.FindWindow(None, self.window_title)
            if self.game_hwnd:
                self.window_rect = win32gui.GetWindowRect(self.game_hwnd)
                width = self.window_rect[2] - self.window_rect[0]
                height = self.window_rect[3] - self.window_rect[1]
                self.center_pos = (
                    self.window_rect[0] + width // 2,
                    self.window_rect[1] + height // 2
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error finding game window: {e}")
            return False
    
    def focus_game_window(self) -> bool:
        """Focus game window and lock mouse."""
        try:
            if not self.game_hwnd and not self.find_game_window():
                return False
                
            # Focus window
            if win32gui.GetForegroundWindow() != self.game_hwnd:
                win32gui.SetForegroundWindow(self.game_hwnd)
                
            # Center mouse in window
            if self.center_pos:
                win32api.SetCursorPos(self.center_pos)
                
            # Lock cursor to window
            win32gui.ClipCursor(self.window_rect)
            
            self.is_focused = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error focusing game window: {e}")
            return False
    
    def release_game_window(self) -> None:
        """Release mouse lock and window focus."""
        try:
            # Release cursor lock
            win32gui.ClipCursor(None)
            self.is_focused = False
        except Exception as e:
            self.logger.error(f"Error releasing game window: {e}")
            
    def get_window_list(self) -> List[str]:
        """Get list of all visible window titles."""
        windows = []
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and not title.isspace():  # Only add non-empty titles
                    windows.append(title)
                    
        try:
            win32gui.EnumWindows(enum_windows_callback, windows)
            return sorted(windows)  # Sort alphabetically for easier selection
        except Exception as e:
            self.logger.error(f"Error getting window list: {e}")
            return []
            
    def set_active_window(self, window_title: str) -> bool:
        """Set the active window for capture by title."""
        try:
            self.window_title = window_title
            if self.find_game_window():
                self.active_window = True
                return True
            self.active_window = False
            return False
        except Exception as e:
            self.logger.error(f"Error setting active window: {e}")
            self.active_window = False
            return False
            
    def get_window_title(self) -> str:
        """Get the current window title."""
        return self.window_title
        
    def capture_window(self) -> Optional[np.ndarray]:
        """Capture the current window content."""
        try:
            if not self.game_hwnd:
                return None
                
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(self.game_hwnd)
            width = right - left
            height = bottom - top
            
            # Get window DC
            hwnd_dc = win32gui.GetWindowDC(self.game_hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create bitmap object
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Copy window content
            result = win32gui.PrintWindow(self.game_hwnd, save_dc.GetSafeHdc(), 0)
            
            if result == 1:
                # Convert to numpy array
                bmpinfo = save_bitmap.GetInfo()
                bmpstr = save_bitmap.GetBitmapBits(True)
                img = np.frombuffer(bmpstr, dtype=np.uint8)
                img.shape = (height, width, 4)  # RGBA
                
                # Convert from RGBA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                # Cleanup
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(self.game_hwnd, hwnd_dc)
                
                return img
            else:
                self.logger.error("Failed to capture window content")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing window: {e}")
            return None
    
    def move_mouse_relative(self, dx: float, dy: float) -> None:
        """Move mouse relative to current position."""
        try:
            if not self.is_focused:
                return
                
            # Get current position
            x, y = win32api.GetCursorPos()
            
            # Apply movement
            new_x = int(x + dx)
            new_y = int(y + dy)
            
            # Keep within window bounds
            if self.window_rect:
                new_x = max(self.window_rect[0], min(new_x, self.window_rect[2]))
                new_y = max(self.window_rect[1], min(new_y, self.window_rect[3]))
            
            # Move cursor
            win32api.SetCursorPos((new_x, new_y))
            
        except Exception as e:
            self.logger.error(f"Error moving mouse: {e}")
    
    def set_mouse_position(self, x: float, y: float) -> None:
        """Set absolute mouse position within game window."""
        try:
            if not self.is_focused or not self.window_rect:
                return
                
            # Convert to window coordinates
            window_x = int(self.window_rect[0] + x)
            window_y = int(self.window_rect[1] + y)
            
            # Keep within window bounds
            window_x = max(self.window_rect[0], min(window_x, self.window_rect[2]))
            window_y = max(self.window_rect[1], min(window_y, self.window_rect[3]))
            
            # Set position
            win32api.SetCursorPos((window_x, window_y))
            
        except Exception as e:
            self.logger.error(f"Error setting mouse position: {e}")
