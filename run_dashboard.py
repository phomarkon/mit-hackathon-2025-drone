#!/usr/bin/env python3
"""
Cross-platform script to run the Thermal Drone Anomaly Detector dashboard.
This will work on Windows, macOS, and Linux.
"""

import os
import sys
import subprocess
import platform

def run_streamlit():
    """Run the Streamlit dashboard application"""
    print("="*60)
    print("Thermal Drone Anomaly Detector Dashboard")
    print("="*60)
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("Note: You're not running in a virtual environment.")
        print("If you encounter issues, consider creating and activating a virtual environment.")
        print("-"*60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"Using Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit is not installed. Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Error installing dependencies. Please run manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    print("Starting the dashboard...")
    print("(Press Ctrl+C to quit)")
    print("-"*60)
    
    # Run the Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit() 