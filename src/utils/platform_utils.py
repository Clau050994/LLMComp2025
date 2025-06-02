"""
Platform-specific utilities to ensure cross-platform compatibility.
Provides functions for handling paths and commands across different operating systems.
"""

import os
import platform
import subprocess
import sys
from typing import List, Optional, Union

def is_windows() -> bool:
    """Check if the current platform is Windows."""
    return platform.system().lower() == "windows"

def is_macos() -> bool:
    """Check if the current platform is macOS."""
    return platform.system().lower() == "darwin"

def is_linux() -> bool:
    """Check if the current platform is Linux."""
    return platform.system().lower() == "linux"

def get_platform_name() -> str:
    """Get a string representation of the current platform."""
    if is_windows():
        return "windows"
    elif is_macos():
        return "macos"
    elif is_linux():
        return "linux"
    else:
        return "unknown"

def normalize_path(path: str) -> str:
    """
    Normalize a path for the current operating system.
    
    Args:
        path (str): The path to normalize
        
    Returns:
        str: Normalized path for the current OS
    """
    # Replace forward slashes with the OS-specific separator
    return os.path.normpath(path)

def run_platform_command(command: Union[str, List[str]], shell: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command with platform-specific adjustments.
    
    Args:
        command (str or List[str]): Command to run
        shell (bool): Whether to use shell execution
        
    Returns:
        subprocess.CompletedProcess: Result of the command
    """
    if isinstance(command, list):
        cmd_str = " ".join(command)
    else:
        cmd_str = command
    
    if is_windows():
        # Adjust commands for Windows if necessary
        # Example: Replace Unix commands with Windows equivalents
        cmd_str = cmd_str.replace("cp ", "copy ")
        cmd_str = cmd_str.replace("rm -rf ", "rmdir /s /q ")
        cmd_str = cmd_str.replace("rm ", "del ")
        cmd_str = cmd_str.replace("mkdir -p ", "mkdir ")
        
        # Use cmd.exe on Windows
        shell_cmd = cmd_str
        use_shell = True
    else:
        # Use the command as is on Unix-like systems
        shell_cmd = cmd_str
        use_shell = shell
    
    return subprocess.run(shell_cmd, shell=use_shell, check=True, text=True, capture_output=True)

def get_python_executable() -> str:
    """Get the path to the Python executable."""
    return sys.executable

def get_temp_dir() -> str:
    """Get the platform-appropriate temporary directory."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"))

def setup_platform_specific_configs():
    """
    Set up platform-specific configurations.
    Call this during initialization to adjust settings for the current platform.
    """
    if is_windows():
        # Windows-specific setup
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Check if running in PowerShell and adjust accordingly
        if os.environ.get('PSModulePath'):
            print("Detected PowerShell environment. Some commands may need adjustment.")
            
    # Create temp directory if it doesn't exist
    os.makedirs(get_temp_dir(), exist_ok=True)
