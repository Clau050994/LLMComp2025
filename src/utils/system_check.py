"""
System check utility to verify platform compatibility and dependencies.
This helps ensure the framework runs correctly across different operating systems.
"""

import importlib
import os
import platform
import shutil
import sys
from typing import Dict, List, Tuple

def check_system_compatibility() -> Tuple[bool, List[str]]:
    """
    Check if the current system is compatible with the framework.
    
    Returns:
        Tuple[bool, List[str]]: (is_compatible, list_of_issues)
    """
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        issues.append(f"Python 3.8+ required, but found {python_version.major}.{python_version.minor}")
    
    # Check operating system
    os_name = platform.system()
    if os_name not in ['Windows', 'Linux', 'Darwin']:
        issues.append(f"Unsupported operating system: {os_name}")
    
    # Windows-specific checks
    if os_name == 'Windows':
        # Check if Git Bash or WSL is available for better compatibility
        has_git_bash = shutil.which('bash.exe') is not None
        has_wsl = shutil.which('wsl.exe') is not None
        
        if not (has_git_bash or has_wsl):
            issues.append("Windows detected. For better compatibility, install Git Bash or WSL (Windows Subsystem for Linux)")
    
    return len(issues) == 0, issues

def check_required_packages() -> Tuple[bool, Dict[str, bool]]:
    """
    Check if required packages are installed.
    
    Returns:
        Tuple[bool, Dict[str, bool]]: (all_installed, {package: is_installed})
    """
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'boto3',
        'sagemaker',
        'numpy',
        'psutil'
    ]
    
    package_status = {}
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            package_status[package] = True
        except ImportError:
            package_status[package] = False
    
    all_installed = all(package_status.values())
    return all_installed, package_status

def check_optional_packages() -> Dict[str, bool]:
    """
    Check if optional packages are installed.
    
    Returns:
        Dict[str, bool]: {package: is_installed}
    """
    optional_packages = [
        'onnx',
        'onnxruntime',
        'bitsandbytes',
        'onnxsim',
        'tensorrt',
        'matplotlib',
        'seaborn'
    ]
    
    package_status = {}
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            package_status[package] = True
        except ImportError:
            package_status[package] = False
    
    return package_status

def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check if GPU is available.
    
    Returns:
        Tuple[bool, str]: (is_available, description)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda or "Unknown"
            
            return True, f"CUDA available: {device_count} GPU(s), {device_name}, CUDA {cuda_version}"
        else:
            return False, "CUDA not available, running in CPU mode"
    except:
        return False, "Error checking GPU - torch.cuda might not be properly installed"

def run_system_check() -> Dict:
    """
    Run a complete system check and return results.
    
    Returns:
        Dict: Check results
    """
    results = {
        "platform": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_path": sys.executable,
    }
    
    # System compatibility
    compatible, issues = check_system_compatibility()
    results["system_compatible"] = compatible
    results["compatibility_issues"] = issues
    
    # Required packages
    required_all_installed, required_status = check_required_packages()
    results["required_packages_installed"] = required_all_installed
    results["required_packages_status"] = required_status
    
    # Optional packages
    optional_status = check_optional_packages()
    results["optional_packages_status"] = optional_status
    
    # GPU availability
    gpu_available, gpu_info = check_gpu_availability()
    results["gpu_available"] = gpu_available
    results["gpu_info"] = gpu_info
    
    return results

def print_check_results(results: Dict):
    """
    Print the system check results in a readable format.
    
    Args:
        results: Results from run_system_check()
    """
    print("\n" + "="*60)
    print(" "*20 + "SYSTEM CHECK RESULTS" + " "*20)
    print("="*60)
    
    print(f"\nPlatform: {results['platform']}")
    print(f"Python: {results['python_version']} ({results['python_path']})")
    
    print("\nSystem Compatibility:")
    if results['system_compatible']:
        print("  ✓ System is compatible")
    else:
        print("  ✗ System has compatibility issues:")
        for issue in results['compatibility_issues']:
            print(f"    - {issue}")
    
    print("\nRequired Packages:")
    for package, installed in results['required_packages_status'].items():
        status = "✓" if installed else "✗"
        print(f"  {status} {package}")
    
    print("\nOptional Packages:")
    for package, installed in results['optional_packages_status'].items():
        status = "✓" if installed else "○"
        print(f"  {status} {package}")
    
    print("\nGPU Support:")
    print(f"  {'✓' if results['gpu_available'] else '○'} {results['gpu_info']}")
    
    print("\nRecommendations:")
    if not results['system_compatible']:
        print("  - Fix the compatibility issues listed above")
    
    missing_required = [pkg for pkg, installed in results['required_packages_status'].items() if not installed]
    if missing_required:
        packages_str = ", ".join(missing_required)
        print(f"  - Install missing required packages: {packages_str}")
    
    useful_missing = [
        pkg for pkg, installed in results['optional_packages_status'].items() 
        if not installed and pkg in ['onnx', 'onnxruntime', 'bitsandbytes']
    ]
    if useful_missing:
        packages_str = ", ".join(useful_missing)
        print(f"  - Consider installing useful optional packages: {packages_str}")
    
    if not results['gpu_available']:
        print("  - For better performance, use a system with GPU support")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    results = run_system_check()
    print_check_results(results)
