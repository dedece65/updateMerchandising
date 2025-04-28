#!/usr/bin/env python3
"""
Example script demonstrating proper usage of Tkinter's askopenfilename on macOS.
"""
import tkinter as tk
from tkinter import filedialog
import os

def select_file():
    try:
        # Create a root window but keep it hidden
        # This is CRUCIAL for macOS compatibility
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Make sure the dialog window comes to the front on macOS
        # This is important for macOS specifically
        root.attributes("-topmost", True)
        
        # Use askopenfilename to create a native file dialog
        # initialdir sets the starting directory (using user's home directory here)
        file_path = filedialog.askopenfilename(
            title="Select a file",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("All Files", "*.*"),
                ("Text Files", "*.txt"),
                ("Python Files", "*.py")
            ]
        )
        
        # Clean up the root window
        root.destroy()
        
        # Check if a file was selected (user didn't cancel)
        if file_path:
            print(f"Selected file: {file_path}")
            return file_path
        else:
            print("No file selected")
            return None
    
    except Exception as e:
        print(f"Error in file dialog: {str(e)}")
        return None

if __name__ == "__main__":
    print("Opening file dialog...")
    result = select_file()
    
    if result:
        # Do something with the selected file
        print(f"You can now process the file: {result}")
    else:
        print("File selection was canceled or encountered an error")