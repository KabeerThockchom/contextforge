#!/usr/bin/env python3
"""Script to help with publishing the package to PyPI."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result


def main():
    """Main publishing workflow."""
    # Ensure we're in the right directory
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    print("🚀 Starting publish process...")
    
    # Run tests
    print("\n📋 Running tests...")
    run_command("python -m pytest tests/ -v")
    
    # Run linting
    print("\n🔍 Running linting...")
    run_command("flake8 contextforge tests")
    
    # Run type checking
    print("\n🔧 Running type checking...")
    run_command("mypy contextforge")
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    run_command("rm -rf build/ dist/ *.egg-info/")
    
    # Build package
    print("\n📦 Building package...")
    run_command("python -m build")
    
    # Check package
    print("\n✅ Checking package...")
    run_command("twine check dist/*")
    
    # Ask for confirmation
    print("\n🤔 Ready to publish to PyPI!")
    print("Files to upload:")
    for file in Path("dist").glob("*"):
        print(f"  - {file}")
    
    response = input("\nDo you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Publish cancelled.")
        return
    
    # Upload to PyPI
    print("\n📤 Uploading to PyPI...")
    run_command("twine upload dist/*")
    
    print("\n🎉 Package published successfully!")
    print("You can now install it with: pip install contextforge")


if __name__ == "__main__":
    main() 