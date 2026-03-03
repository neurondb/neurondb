#!/usr/bin/env python3
"""
Simple HuggingFace to ONNX exporter for NeuronDB
Uses PyTorch's built-in ONNX export functionality
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModel

def export_embedding_model(model_name, output_dir):
    """Export a sentence transformer/embedding model to ONNX"""
    
    print(f"\n{'='*80}")
    print(f"Exporting HuggingFace model to ONNX")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    print("Creating dummy input...")
    dummy_text = "This is a sample sentence."
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    
    # Get input names
    input_names = list(inputs.keys())
    print(f"Input names: {input_names}")
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"\nExporting to {onnx_path}...")
    
    try:
        # Use legacy export mode to avoid Python 3.14 compatibility issues
        with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
            torch.onnx.export(
                model,
                tuple(inputs.values()),
                onnx_path,
                export_params=True,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes={
                    **{name: {0: "batch", 1: "sequence"} for name in input_names},
                    "output": {0: "batch", 1: "sequence"}
                },
                opset_version=14,
                do_constant_folding=True,
                dynamo=False,  # Use legacy export
            )
        print("Export successful!")
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        print(f"\nSaving tokenizer to {tokenizer_path}...")
        tokenizer.save_pretrained(tokenizer_path)
        print("Tokenizer saved!")
        
        # Create model info file
        info_path = os.path.join(output_dir, "model_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Task: feature-extraction\n")
            f.write(f"Input names: {', '.join(input_names)}\n")
            f.write(f"Output names: output\n")
            f.write(f"Max length: 128\n")
            f.write(f"ONNX opset: 14\n")
        
        print(f"\n{'='*80}")
        print("Export complete!")
        print(f"{'='*80}")
        print(f"\nFiles created:")
        print(f"  - {onnx_path}")
        print(f"  - {tokenizer_path}/")
        print(f"  - {info_path}")
        print(f"\nTo use in PostgreSQL:")
        print(f"  SELECT neurondb_hf_embedding('{os.path.basename(output_dir)}', 'your text here');")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to ONNX format for NeuronDB"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for ONNX model"
    )
    
    args = parser.parse_args()
    
    success = export_embedding_model(args.model, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

