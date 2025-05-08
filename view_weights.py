import torch
import json
import argparse
import os
import numpy as np

def view_and_save_weights(weights_path, output_json=None):
    """
    Load weights from a .pth file, display info, and save them as JSON.
    
    Args:
        weights_path (str): Path to the .pth weights file
        output_json (str, optional): Path to save the JSON file
    """
    # Default JSON output path if none specified
    if output_json is None:
        output_json = os.path.splitext(weights_path)[0] + ".json"
    
    # Load the weights
    print(f"Loading weights from {weights_path}...")
    weights = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # Print basic information
    print(f"\n{'='*50}")
    print(f"WEIGHT FILE SUMMARY")
    print(f"{'='*50}")
    print(f"Number of layers: {len(weights)}")
    
    # Display layer information
    print("\nLAYER DETAILS:")
    print(f"{'-'*50}")
    print(f"{'Layer Name':<40} {'Shape':<20} {'Type'}")
    print(f"{'-'*50}")
    
    total_params = 0
    for key, value in weights.items():
        if isinstance(value, torch.Tensor):
            shape_str = str(list(value.shape))
            params = np.prod(value.shape)
            total_params += params
            print(f"{key:<40} {shape_str:<20} {value.dtype}")
        else:
            print(f"{key:<40} {'N/A':<20} {type(value).__name__}")
    
    print(f"{'-'*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"{'-'*50}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View PyTorch weights and save as JSON")
    parser.add_argument("weights_path", help="Path to the .pth weights file")
    parser.add_argument("--output", "-o", help="Path to save the JSON file (optional)")
    
    args = parser.parse_args()
    view_and_save_weights(args.weights_path, args.output)