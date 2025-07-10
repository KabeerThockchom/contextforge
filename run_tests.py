#!/usr/bin/env python3
"""Test runner script for contextforge."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False


def main():
    """Main test runner function."""
    # Check if we're in the right directory
    if not Path("contextforge").exists():
        print("Error: Run this script from the project root directory")
        sys.exit(1)
    
    # Install test dependencies
    if not run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
    ], "Installing test dependencies"):
        print("Failed to install test dependencies")
        sys.exit(1)
    
    # Install the package in development mode
    if not run_command([
        sys.executable, "-m", "pip", "install", "-e", "."
    ], "Installing contextforge in development mode"):
        print("Failed to install contextforge")
        sys.exit(1)
    
    # Run the tests
    test_commands = [
        # Basic test run
        ([sys.executable, "-m", "pytest", "tests/", "-v"], "Running all tests"),
        
        # Test with coverage
        ([sys.executable, "-m", "pytest", "tests/", "--cov=contextforge", "--cov-report=html", "--cov-report=term"], "Running tests with coverage"),
        
        # Test specific modules
        ([sys.executable, "-m", "pytest", "tests/test_core.py", "-v"], "Testing core module"),
        ([sys.executable, "-m", "pytest", "tests/test_providers.py", "-v"], "Testing providers module"),
        ([sys.executable, "-m", "pytest", "tests/test_memory.py", "-v"], "Testing memory module"),
        ([sys.executable, "-m", "pytest", "tests/test_tools.py", "-v"], "Testing tools module"),
        ([sys.executable, "-m", "pytest", "tests/test_retrieval.py", "-v"], "Testing retrieval module"),
        ([sys.executable, "-m", "pytest", "tests/test_utils.py", "-v"], "Testing utils module"),
    ]
    
    success_count = 0
    total_count = len(test_commands)
    
    for cmd, description in test_commands:
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"❌ {description} failed")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Passed: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())