#!/usr/bin/env python3
"""
Model Format Converter for PyTorch

This script converts PyTorch models between different formats to ensure
compatibility across PyTorch versions, especially for models that use numpy arrays
in their state dictionaries which can cause issues with PyTorch 2.6+'s security measures.

Usage:
  python convert_model.py --input <input_model_path> --output <output_model_path> [--mode <mode>]

Modes:
  - pickle: Convert to basic pickle format (least secure but most compatible)
  - safe: Convert to PyTorch's safe format with allowlisted classes (recommended)
  - torch16: Convert to PyTorch 1.6 format for maximum backward compatibility
"""

import os
import sys
import argparse
import torch
import pickle
import shutil

def convert_to_pickle_format(input_path, output_path):
    """Convert model to basic pickle format"""
    print(f"Loading model from {input_path}...")
    
    try:
        # Try to load with torch first (for backward compatibility)
        state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # If that fails, try with pickle directly
        print(f"Torch loading failed, trying with pickle: {e}")
        with open(input_path, 'rb') as f:
            state_dict = pickle.load(f)
    
    print(f"Saving model to {output_path} in pickle format...")
    with open(output_path, 'wb') as f:
        pickle.dump(state_dict, f)
    
    print("Conversion complete!")

def convert_to_safe_format(input_path, output_path):
    """Convert model to PyTorch's safe format with allowlisted classes"""
    print(f"Loading model from {input_path}...")
    
    try:
        # Try to load with torch first
        import numpy
        import torch.serialization
        torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
        state_dict = torch.load(input_path, map_location='cpu')
    except Exception as e:
        # If that fails, try with pickle directly
        print(f"Torch loading failed, trying with pickle: {e}")
        with open(input_path, 'rb') as f:
            state_dict = pickle.load(f)
    
    print(f"Saving model to {output_path} in safe format...")
    torch.save(state_dict, output_path)
    
    print("Conversion complete!")

def convert_to_torch16_format(input_path, output_path):
    """Convert model to PyTorch 1.6 format for maximum backwards compatibility"""
    print(f"Loading model from {input_path}...")
    
    try:
        # Try to load with torch first
        state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # If that fails, try with pickle directly
        print(f"Torch loading failed, trying with pickle: {e}")
        with open(input_path, 'rb') as f:
            state_dict = pickle.load(f)
    
    print(f"Saving model to {output_path} in PyTorch 1.6 format...")
    torch.save(state_dict, output_path, _use_new_zipfile_serialization=False)
    
    print("Conversion complete!")

def backup_model(model_path):
    """Create a backup of the original model file"""
    backup_path = f"{model_path}.backup"
    if not os.path.exists(backup_path):
        print(f"Creating backup of original model at {backup_path}")
        shutil.copy2(model_path, backup_path)
        return backup_path
    else:
        print(f"Backup already exists at {backup_path}")
        return None

def batch_convert_models():
    """Convert all models in the outputs directory"""
    model_paths = [
        "outputs/patchcore_resnet18/patchcore_resnet18.pth",
        "outputs/patchcore_resnet50/patchcore_resnet50.pth"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\nProcessing model: {model_path}")
            backup_path = backup_model(model_path)
            
            # Convert the model to the most compatible format
            convert_to_torch16_format(model_path, model_path)
            print(f"✅ Converted {model_path} to PyTorch 1.6 format")
        else:
            print(f"❌ Model not found: {model_path}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Model Format Converter")
    parser.add_argument("--input", help="Input model path")
    parser.add_argument("--output", help="Output model path")
    parser.add_argument("--mode", choices=["pickle", "safe", "torch16"], default="torch16",
                       help="Conversion mode (pickle, safe, or torch16)")
    parser.add_argument("--batch", action="store_true", 
                       help="Batch convert all models in the outputs directory")
    args = parser.parse_args()
    
    if args.batch:
        print("Running batch conversion of all models...")
        batch_convert_models()
        return
    
    if not args.input:
        parser.error("--input is required unless --batch is specified")
    
    if not args.output and not args.batch:
        # Use the same filename if output is not specified
        args.output = args.input
        backup_model(args.input)
    
    # Perform the appropriate conversion
    if args.mode == "pickle":
        convert_to_pickle_format(args.input, args.output)
    elif args.mode == "safe":
        convert_to_safe_format(args.input, args.output)
    elif args.mode == "torch16":
        convert_to_torch16_format(args.input, args.output)

if __name__ == "__main__":
    main() 