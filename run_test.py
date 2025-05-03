#!/usr/bin/env python3
"""
Test script for Thermal Drone Anomaly Detection

This script provides a simple interface to test the application
by either launching the Streamlit dashboard or processing a directory
of images with both ResNet18 and ResNet50 models.
"""

import os
import sys
import argparse
import subprocess

def validate_model_files():
    """Check if model files exist and report their status"""
    model_files = [
        "outputs/patchcore_resnet18/patchcore_resnet18.pth",
        "outputs/patchcore_resnet50/patchcore_resnet50.pth"
    ]
    
    all_found = True
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model: {model_file}")
        else:
            print(f"❌ Missing model: {model_file}")
            all_found = False
    
    return all_found

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "dashboard.py"])

def process_directory(input_dir, output_dir=None, save_images=False):
    """Process a directory with both models"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process with ResNet18
    print("\n===== Processing with ResNet18 =====")
    cmd = ["python", "process_directory.py", "--input", input_dir]
    if output_dir:
        cmd.extend(["--output", os.path.join(output_dir, "resnet18")])
    if save_images:
        cmd.append("--save")
    subprocess.run(cmd)
    
    # Process with ResNet50
    print("\n===== Processing with ResNet50 =====")
    cmd = ["python", "process_directory.py", "--input", input_dir, "--model", "resnet50"]
    if output_dir:
        cmd.extend(["--output", os.path.join(output_dir, "resnet50")])
    if save_images:
        cmd.append("--save")
    subprocess.run(cmd)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test Thermal Drone Anomaly Detection")
    parser.add_argument("--mode", choices=["dashboard", "process"], required=True,
                        help="Mode of operation: 'dashboard' to launch the Streamlit dashboard, 'process' to process a directory")
    parser.add_argument("--input", help="Input directory containing images (for 'process' mode)")
    parser.add_argument("--output", help="Output directory for results (for 'process' mode)")
    parser.add_argument("--save", action="store_true", help="Save processed images with annotations (for 'process' mode)")
    args = parser.parse_args()
    
    # Validate model files
    print("Checking for model files...")
    models_exist = validate_model_files()
    if not models_exist:
        print("\nWarning: Some model files are missing. The application may not work correctly.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run selected mode
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "process":
        if not args.input:
            print("Error: When using 'process' mode, you must specify an input directory with --input")
            return
        process_directory(args.input, args.output, args.save)

if __name__ == "__main__":
    main() 