#!/usr/bin/env python3
"""
Setup script for DFT Agent
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is 3.12+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.12+")
        return False

def check_uv_installed():
    """Check if UV is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV package manager is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV package manager not found")
        return False

def install_uv():
    """Install UV package manager."""
    print("🔄 Installing UV package manager...")
    try:
        subprocess.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh"], 
                      shell=True, check=True)
        print("✅ UV installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install UV. Please install manually from https://docs.astral.sh/uv/")
        return False

def setup_environment():
    """Set up the environment file."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        print("🔄 Creating .env file from template...")
        try:
            with open(env_example, "r") as f:
                content = f.read()
            with open(env_file, "w") as f:
                f.write(content)
            print("✅ .env file created. Please edit it with your API keys.")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("❌ env.example file not found")
        return False

def install_dependencies():
    """Install project dependencies."""
    if check_uv_installed():
        return run_command("uv sync", "Installing dependencies with UV")
    else:
        print("🔄 UV not available, trying pip...")
        return run_command("pip install -e .", "Installing dependencies with pip")

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/inputs/pseudopotentials",
        "data/outputs/calculations",
        "data/outputs/structures",
        "logs",
        "WORKSPACE",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    return True

def run_tests():
    """Run basic tests."""
    return run_command("python test_basic_structure.py", "Running basic structure tests")

def main():
    """Main setup function."""
    print("🚀 DFT Agent Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing UV (if needed)", lambda: check_uv_installed() or install_uv()),
        ("Setting up environment", setup_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Running tests", run_tests),
    ]
    
    failed_steps = []
    
    for description, func in steps:
        print(f"\n📋 {description}")
        if not func():
            failed_steps.append(description)
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    
    if failed_steps:
        print(f"⚠️  Some steps failed: {', '.join(failed_steps)}")
        print("Please check the errors above and fix them manually.")
    else:
        print("✅ All setup steps completed successfully!")
    
    print("\n📖 Next Steps:")
    print("1. Edit .env file with your API keys")
    print("2. Start the backend: cd backend && python run_service.py")
    print("3. Start the frontend: cd frontend && streamlit run app.py")
    print("4. Open http://localhost:8501 in your browser")
    
    print("\n📚 Documentation:")
    print("- README.md: Complete usage guide")
    print("- env.example: Environment configuration template")
    print("- test_*.py: Test files for different components")

if __name__ == "__main__":
    main()
