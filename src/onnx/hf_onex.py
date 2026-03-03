#!/usr/bin/env python3
"""
================================================================================
NeuronDB HuggingFace Model Exporter — Elegant, Modular, Thorough
================================================================================

This meticulously crafted script empowers you to export HuggingFace Transformer
models into ONNX format, ensuring seamless integration with NeuronDB’s ONNX
Runtime C API. It includes advanced verification, optional quantization, rich
usage examples, and modular extension points for future-proof workflows.

------------------------------------------------------------------------------
REQUIREMENTS:
    pip install transformers optimum[onnxruntime] onnx

EXAMPLES:
    python export_hf_to_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction --output ./models/all-MiniLM-L6-v2
    python export_hf_to_onnx.py --model distilbert-base-uncased-finetuned-sst-2-english --task text-classification --output ./models/sst2-sentiment --quantize

USAGE IN NEURONDB:
    1. Copy the exported directory to your ONNX model path.
    2. Set neurondb.onnx_model_path = '/path/to/models'
    3. Example SQL: SELECT neurondb_hf_embedding('all-MiniLM-L6-v2', 'some text');
------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Type, Dict, Any, List

# ==============================
# Dependency Checking and Import
# ==============================
def check_and_import_dependencies():
    """
    Ensures required packages are present, providing a beautiful error if not.
    """
    try:
        import onnx
        from transformers import AutoTokenizer
        from optimum.onnxruntime import (
            ORTModelForFeatureExtraction,
            ORTModelForSequenceClassification,
            ORTModelForTokenClassification,
            ORTModelForQuestionAnswering,
            ORTModelForSeq2SeqLM,
        )
        return {
            "onnx": onnx,
            "AutoTokenizer": AutoTokenizer,
            "ORTModelForFeatureExtraction": ORTModelForFeatureExtraction,
            "ORTModelForSequenceClassification": ORTModelForSequenceClassification,
            "ORTModelForTokenClassification": ORTModelForTokenClassification,
            "ORTModelForQuestionAnswering": ORTModelForQuestionAnswering,
            "ORTModelForSeq2SeqLM": ORTModelForSeq2SeqLM,
        }
    except ImportError as e:
        print("\n" + "="*80, file=sys.stderr)
        print("Error: One or more required packages are missing.", file=sys.stderr)
        print("- transformers", file=sys.stderr)
        print("- optimum[onnxruntime]", file=sys.stderr)
        print("- onnx", file=sys.stderr)
        print()
        print("Install with:", file=sys.stderr)
        print("    pip install transformers optimum[onnxruntime] onnx", file=sys.stderr)
        print(f"\nDetails: {e}", file=sys.stderr)
        print("="*80, file=sys.stderr)
        sys.exit(1)


# ===========================
# Supported Model Task Table
# ===========================
def get_supported_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Defines which tasks and their corresponding optimum ONNX classes and features.
    Keys are canonical tasks, extra entries are aliases.
    """
    return {
        'feature-extraction': {
            'cls': 'ORTModelForFeatureExtraction',
            'description': 'Sentence embeddings or vector extraction',
            'aliases': ['sentence-similarity'],
        },
        'text-classification': {
            'cls': 'ORTModelForSequenceClassification',
            'description': 'Sequence or single-label classification (e.g. sentiment)',
            'aliases': ['sentiment-analysis'],
        },
        'token-classification': {
            'cls': 'ORTModelForTokenClassification',
            'description': 'Tokenwise classification (e.g. NER)',
            'aliases': ['ner'],
        },
        'question-answering': {
            'cls': 'ORTModelForQuestionAnswering',
            'description': 'Extractive question answering',
            'aliases': [],
        },
        'text-generation': {
            'cls': 'ORTModelForSeq2SeqLM',
            'description': 'Sequence to sequence or generative models',
            'aliases': [],
        },
    }


def resolve_task(task: str, classes: Dict[str, Any]) -> Optional[str]:
    """
    Maps input task name (with support for aliases) to a canonical supported task.
    Returns the canonical key, or None if unknown.
    """
    tasks = get_supported_tasks()
    for key, meta in tasks.items():
        if task == key or task in meta['aliases']:
            return key
    return None


def get_model_class_obj(task_key: str, classes: Dict[str, Any]):
    """
    Returns the optimum class object for the given task according to dependencies dict.
    """
    model_info = get_supported_tasks().get(task_key)
    if not model_info:
        return None
    return classes[model_info['cls']]


# ===========================
# Modular Export Subroutines
# ===========================
def prepare_output_directory(output_dir: str, force: bool = False):
    """
    Creates the output directory in a robust and idempotent way.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        if force:
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            raise RuntimeError(f"{output_dir} exists and is not a directory.")
    else:
        output_path.mkdir(parents=True, exist_ok=True)


def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """
    Persists the ONNX model and its tokenizer to the output path.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def verify_onnx_model(onnx_path: Path, onnx_module):
    """
    Loads and checks the ONNX model for correctness. Prints input/output shapes.
    """
    print(f"Step 4: Verifying ONNX model at '{onnx_path}' ...")
    if not onnx_path.exists():
        print(f"WARNING: No ONNX file at {onnx_path}")
        return False

    onnx_model = onnx_module.load(str(onnx_path))
    onnx_module.checker.check_model(onnx_model)
    print(f"ONNX model verified successfully\n")
    print("Model Inputs and Outputs:")
    for inp in onnx_model.graph.input:
        shape = [dim.dim_value or '?' for dim in inp.type.tensor_type.shape.dim]
        print(f"   [Input]  {inp.name:20s} shape: {shape}")
    for out in onnx_model.graph.output:
        shape = [dim.dim_value or '?' for dim in out.type.tensor_type.shape.dim]
        print(f"   [Output] {out.name:20s} shape: {shape}")

    return True


def quantize_onnx_model(output_dir: str, file_name="model.onnx"):
    """
    Optionally quantizes the ONNX model for performance boost (e.g., avx512_vnni).
    """
    print("\nStep 5: Quantizing model for x86 AVX512-VNNI ...")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    quantizer = ORTQuantizer.from_pretrained(output_dir, file_name=file_name)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
    print("Model quantized successfully!")


def pretty_success_message(output_dir: str, onnx_path: str):
    "Displays instructions for NeuronDB usage."
    name_only = Path(output_dir).stem or Path(output_dir).name
    print("")
    print("=" * 80)
    print("Export Complete!\n")
    print(f"Model saved to: {output_dir}")
    print(f"ONNX file: {onnx_path}\n")
    print("To use with NeuronDB:\n"
          f"  1. Copy '{output_dir}' to your ONNX model directory"
          f"\n  2. Set neurondb.onnx_model_path = '/path/to/models'"
          f"\n  3. Run: SELECT neurondb_hf_embedding('{name_only}', 'your text');")
    print("=" * 80)
    print("")


def export_model(
    model_id: str,
    task: str,
    output_dir: str,
    quantize: bool = False,
    verbose: bool = True,
    force: bool = False,
) -> bool:
    """
    Comprehensive pipeline for exporting a HuggingFace model to ONNX, with extra beauty.
    """
    # Dependency resolution (import required classes, fail early if missing)
    classes = check_and_import_dependencies()

    print("\n" + "=" * 80)
    print("NeuronDB HuggingFace Exporter")
    print("=" * 80)
    print(f"Model      : {model_id}")
    print(f"Task       : {task}")
    print(f"Output Dir : {output_dir}")
    print(f"Quantize   : {quantize}")
    print("")

    # Resolve canonical task name and load its class
    task_key = resolve_task(task, classes)
    if not task_key:
        supported = ", ".join(sorted(get_supported_tasks().keys() | {a for v in get_supported_tasks().values() for a in v['aliases']}))
        print(f"ERROR: Unknown or unsupported task '{task}'\nSupported tasks: {supported}")
        return False

    model_cls = get_model_class_obj(task_key, classes)
    if not model_cls:
        print(f"ERROR: No export class for task {task_key}")
        return False

    try:
        # 1. Prepare output directory
        prepare_output_directory(output_dir)

        # 2. Load and export model to ONNX with optimum
        print("Step 1: Downloading model from HuggingFace Hub & exporting to ONNX ...")
        model = model_cls.from_pretrained(model_id, export=True)
        print("Model loaded & exported.")

        # 3. Load and save tokenizer
        print("Step 2: Loading and saving tokenizer ...")
        tokenizer = classes["AutoTokenizer"].from_pretrained(model_id)
        save_model_and_tokenizer(model, tokenizer, output_dir)
        print("Tokenizer saved.")

        # 4. Verify ONNX file and print model info
        onnx_path = Path(output_dir) / "model.onnx"
        verify_onnx_model(onnx_path, classes["onnx"])

        # 5. Quantize (optional)
        if quantize:
            quantize_onnx_model(output_dir)

        # 6. Pretty usage instructions
        pretty_success_message(output_dir, onnx_path)

        return True

    except Exception as e:
        print("="*80)
        print(f"ERROR during export: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return False

# ===============================
# CLI Modular Entry and Examples
# ===============================
def create_arg_parser():
    tasks_info = get_supported_tasks()
    task_choices = []
    for key, val in tasks_info.items():
        task_choices.append(key)
        task_choices.extend(val.get("aliases", []))
    task_choices = sorted(set(task_choices))

    usage_examples = """
Beautiful Examples:
  # 1. Export a sentence transformer for embeddings
  python export_hf_to_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction --output ./models/all-MiniLM-L6-v2

  # 2. Export a sentiment model with quantization
  python export_hf_to_onnx.py --model distilbert-base-uncased-finetuned-sst-2-english --task text-classification --output ./models/sst2-sentiment --quantize

  # 3. Export a NER model
  python export_hf_to_onnx.py --model dslim/bert-base-NER --task ner --output ./models/bert-ner

  # 4. Export a QA model
  python export_hf_to_onnx.py --model distilbert-base-cased-distilled-squad --task question-answering --output ./models/distilbert-qa

  # See all HuggingFace models at https://huggingface.co/models
    """
    return argparse.ArgumentParser(
        description='Beautifully export HuggingFace models to ONNX for NeuronDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=usage_examples,
    ), task_choices

def main():
    parser, task_choices = create_arg_parser()
    parser.add_argument('--model', required=True,
                        help='HuggingFace model ID (e.g., sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--task', required=True, choices=task_choices,
                        help=f"Model task type. Supported: {', '.join(task_choices)}")
    parser.add_argument('--output', required=True,
                        help='Output directory for the exported ONNX model')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize model for faster inference (x86 CPU, AVX512/VNNI)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite output directory if it exists')
    args = parser.parse_args()

    ok = export_model(
        model_id=args.model,
        task=args.task,
        output_dir=args.output,
        quantize=args.quantize,
        force=args.force,
    )
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
