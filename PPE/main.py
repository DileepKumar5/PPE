import ctypes
ctypes.windll.user32.MessageBoxW(0, "This is a test notification", "Test Notification", 0x40 | 0x1)
