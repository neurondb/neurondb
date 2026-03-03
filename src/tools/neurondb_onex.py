#!/usr/bin/env python3
"""
NeuronDB ONNX Exchange (ONEX)
Clean, modular tool for importing and exporting models between NeuronDB and ONNX formats.

Usage:
    Export: python neurondb_onex.py --export <model_data_file> <algorithm> <output_path>
    Import: python neurondb_onex.py --import <onnx_file> <algorithm> <output_path>
    
    algorithm: ridge, lasso, linear_regression, logistic_regression
"""

import struct
import sys
import os
import argparse
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("ERROR: onnx library not installed. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Binary Format Reader/Writer
# ============================================================================

class NeuronDBModelReader:
    """Reads NeuronDB custom binary model formats (PostgreSQL pq format, network byte order)"""
    
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
    
    def read_uint8(self) -> int:
        """Read 8-bit unsigned integer"""
        if self.pos + 1 > len(self.data):
            raise ValueError("Unexpected end of data")
        val = struct.unpack('>B', self.data[self.pos:self.pos+1])[0]
        self.pos += 1
        return val
    
    def read_int32(self) -> int:
        """Read 32-bit integer (network byte order = big-endian)"""
        if self.pos + 4 > len(self.data):
            raise ValueError("Unexpected end of data")
        val = struct.unpack('>i', self.data[self.pos:self.pos+4])[0]
        self.pos += 4
        return val
    
    def read_float64(self) -> float:
        """Read 64-bit float (network byte order = big-endian)"""
        if self.pos + 8 > len(self.data):
            raise ValueError("Unexpected end of data")
        val = struct.unpack('>d', self.data[self.pos:self.pos+8])[0]
        self.pos += 8
        return val


class NeuronDBModelWriter:
    """Writes NeuronDB custom binary model formats"""
    
    def __init__(self):
        self.data = bytearray()
    
    def write_uint8(self, val: int):
        """Write 8-bit unsigned integer"""
        self.data.extend(struct.pack('>B', val))
    
    def write_int32(self, val: int):
        """Write 32-bit integer (network byte order = big-endian)"""
        self.data.extend(struct.pack('>i', val))
    
    def write_float64(self, val: float):
        """Write 64-bit float (network byte order = big-endian)"""
        self.data.extend(struct.pack('>d', val))
    
    def get_bytes(self) -> bytes:
        """Get final bytes with varlena header"""
        # Add varlena header (4 bytes for length)
        length = len(self.data) + 4
        header = struct.pack('>I', length)
        return header + bytes(self.data)


# ============================================================================
# Deserializers: NeuronDB Binary -> Python Dict
# ============================================================================

def deserialize_ridge_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Ridge regression model from NeuronDB binary format"""
    reader = NeuronDBModelReader(data)
    
    # Skip varlena header (4 bytes for length)
    if len(data) < 4:
        raise ValueError("Invalid data: too short")
    reader.pos = 4  # Skip varlena header
    
    model = {
        'algorithm': 'ridge',
        'n_features': reader.read_int32(),
        'n_samples': reader.read_int32(),
        'intercept': reader.read_float64(),
        'lambda': reader.read_float64(),
        'r_squared': reader.read_float64(),
        'mse': reader.read_float64(),
        'mae': reader.read_float64(),
    }
    
    # Read coefficients
    n_features = model['n_features']
    if n_features > 0:
        coefficients = []
        for i in range(n_features):
            coefficients.append(reader.read_float64())
        model['coefficients'] = np.array(coefficients, dtype=np.float32)
    
    return model


def deserialize_lasso_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Lasso regression model from NeuronDB binary format"""
    model = deserialize_ridge_model(data)
    model['algorithm'] = 'lasso'
    return model


def deserialize_linear_regression_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Linear Regression model from NeuronDB binary format"""
    reader = NeuronDBModelReader(data)
    
    if len(data) < 4:
        raise ValueError("Invalid data: too short")
    reader.pos = 4  # Skip varlena header
    
    model = {
        'algorithm': 'linear_regression',
        'n_features': reader.read_int32(),
        'n_samples': reader.read_int32(),
        'intercept': reader.read_float64(),
        'r_squared': reader.read_float64(),
        'mse': reader.read_float64(),
        'mae': reader.read_float64(),
    }
    
    n_features = model['n_features']
    if n_features > 0:
        coefficients = []
        for i in range(n_features):
            coefficients.append(reader.read_float64())
        model['coefficients'] = np.array(coefficients, dtype=np.float32)
    
    return model


def deserialize_logistic_regression_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Logistic Regression model from NeuronDB binary format (unified ONEX format)"""
    reader = NeuronDBModelReader(data)
    
    if len(data) < 5:
        raise ValueError("Invalid data: too short")
    reader.pos = 4  # Skip varlena header
    
    # Read training_backend byte (0=CPU, 1=GPU)
    training_backend = reader.read_uint8()
    
    model = {
        'algorithm': 'logistic_regression',
        'training_backend': training_backend,  # 0=CPU, 1=GPU
        'n_features': reader.read_int32(),
        'n_samples': reader.read_int32(),
        'bias': reader.read_float64(),
        'learning_rate': reader.read_float64(),
        'lambda': reader.read_float64(),
        'max_iters': reader.read_int32(),
        'final_loss': reader.read_float64(),
        'accuracy': reader.read_float64(),
    }
    
    n_features = model['n_features']
    if n_features > 0:
        weights = []
        for i in range(n_features):
            weights.append(reader.read_float64())
        model['weights'] = np.array(weights, dtype=np.float32)
    
    return model


# ============================================================================
# Serializers: Python Dict -> NeuronDB Binary
# ============================================================================

def serialize_ridge_model(model: Dict[str, Any]) -> bytes:
    """Serialize Ridge regression model to NeuronDB binary format"""
    writer = NeuronDBModelWriter()
    
    writer.write_int32(model['n_features'])
    writer.write_int32(model['n_samples'])
    writer.write_float64(model['intercept'])
    writer.write_float64(model['lambda'])
    writer.write_float64(model['r_squared'])
    writer.write_float64(model['mse'])
    writer.write_float64(model['mae'])
    
    # Write coefficients
    if 'coefficients' in model and model['n_features'] > 0:
        coefficients = model['coefficients']
        for i in range(model['n_features']):
            writer.write_float64(float(coefficients[i]))
    
    return writer.get_bytes()


def serialize_linear_regression_model(model: Dict[str, Any]) -> bytes:
    """Serialize Linear Regression model to NeuronDB binary format"""
    writer = NeuronDBModelWriter()
    
    writer.write_int32(model['n_features'])
    writer.write_int32(model['n_samples'])
    writer.write_float64(model['intercept'])
    writer.write_float64(model['r_squared'])
    writer.write_float64(model['mse'])
    writer.write_float64(model['mae'])
    
    # Write coefficients
    if 'coefficients' in model and model['n_features'] > 0:
        coefficients = model['coefficients']
        for i in range(model['n_features']):
            writer.write_float64(float(coefficients[i]))
    
    return writer.get_bytes()


def serialize_logistic_regression_model(model: Dict[str, Any]) -> bytes:
    """Serialize Logistic Regression model to NeuronDB binary format (unified ONEX format)"""
    writer = NeuronDBModelWriter()
    
    # Write training_backend byte first (0=CPU, 1=GPU)
    training_backend = model.get('training_backend', 0)  # Default to CPU if not specified
    writer.write_uint8(training_backend)
    
    writer.write_int32(model['n_features'])
    writer.write_int32(model['n_samples'])
    writer.write_float64(model['bias'])
    writer.write_float64(model['learning_rate'])
    writer.write_float64(model['lambda'])
    writer.write_int32(model['max_iters'])
    writer.write_float64(model['final_loss'])
    writer.write_float64(model['accuracy'])
    
    # Write weights
    if 'weights' in model and model['n_features'] > 0:
        weights = model['weights']
        for i in range(model['n_features']):
            writer.write_float64(float(weights[i]))
    
    return writer.get_bytes()


# ============================================================================
# ONNX Export: NeuronDB Dict -> ONNX
# ============================================================================

def export_ridge_to_onnx(model: Dict[str, Any], output_path: str) -> str:
    """Convert Ridge model to ONNX format"""
    n_features = model['n_features']
    coefficients = model['coefficients']
    intercept = model['intercept']
    
    # Input tensor: [batch_size, n_features]
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, n_features]
    )
    
    # Coefficients as initializer: [n_features, 1]
    coeff_2d = coefficients.reshape(n_features, 1)
    coeff_tensor = helper.make_tensor(
        'coefficients',
        TensorProto.FLOAT,
        [n_features, 1],
        coeff_2d.flatten().tolist()
    )
    
    # Intercept as initializer: [1]
    intercept_tensor = helper.make_tensor(
        'intercept',
        TensorProto.FLOAT,
        [1],
        [float(intercept)]
    )
    
    # Output tensor: [batch_size, 1]
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, 1]
    )
    
    # Graph: MatMul(input, coefficients) + intercept
    matmul_node = helper.make_node(
        'MatMul',
        ['input', 'coefficients'],
        ['matmul_output'],
        name='MatMul'
    )
    
    add_node = helper.make_node(
        'Add',
        ['matmul_output', 'intercept'],
        ['output'],
        name='Add'
    )
    
    graph = helper.make_graph(
        [matmul_node, add_node],
        'ridge_regression',
        [input_tensor],
        [output_tensor],
        [coeff_tensor, intercept_tensor]
    )
    
    model_proto = helper.make_model(graph, producer_name='NeuronDB')
    model_proto.opset_import[0].version = 14
    
    onnx.save(model_proto, output_path)
    return output_path


def export_linear_regression_to_onnx(model: Dict[str, Any], output_path: str) -> str:
    """Convert Linear Regression model to ONNX format"""
    return export_ridge_to_onnx(model, output_path)


def export_logistic_regression_to_onnx(model: Dict[str, Any], output_path: str) -> str:
    """Convert Logistic Regression model to ONNX format"""
    n_features = model['n_features']
    weights = model['weights']
    bias = model['bias']
    
    # Input tensor: [batch_size, n_features]
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, n_features]
    )
    
    # Weights as initializer: [n_features, 1]
    weights_2d = weights.reshape(n_features, 1)
    weights_tensor = helper.make_tensor(
        'weights',
        TensorProto.FLOAT,
        [n_features, 1],
        weights_2d.flatten().tolist()
    )
    
    # Bias as initializer: [1]
    bias_tensor = helper.make_tensor(
        'bias',
        TensorProto.FLOAT,
        [1],
        [float(bias)]
    )
    
    # Output tensor: [batch_size, 1]
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, 1]
    )
    
    # Graph: MatMul -> Add -> Sigmoid
    matmul_node = helper.make_node(
        'MatMul',
        ['input', 'weights'],
        ['matmul_output'],
        name='MatMul'
    )
    
    add_node = helper.make_node(
        'Add',
        ['matmul_output', 'bias'],
        ['add_output'],
        name='Add'
    )
    
    sigmoid_node = helper.make_node(
        'Sigmoid',
        ['add_output'],
        ['output'],
        name='Sigmoid'
    )
    
    graph = helper.make_graph(
        [matmul_node, add_node, sigmoid_node],
        'logistic_regression',
        [input_tensor],
        [output_tensor],
        [weights_tensor, bias_tensor]
    )
    
    model_proto = helper.make_model(graph, producer_name='NeuronDB')
    model_proto.opset_import[0].version = 14
    
    onnx.save(model_proto, output_path)
    return output_path


# ============================================================================
# ONNX Import: ONNX -> NeuronDB Dict
# ============================================================================

def import_onnx_to_ridge(onnx_path: str) -> Dict[str, Any]:
    """Import ONNX model to Ridge format"""
    model_proto = onnx.load(onnx_path)
    
    # Extract coefficients and intercept from initializers
    coeff_tensor = None
    intercept_tensor = None
    
    for initializer in model_proto.graph.initializer:
        if initializer.name == 'coefficients':
            if initializer.raw_data:
                coeff_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            else:
                # Fallback to float_data if raw_data not available
                coeff_data = np.array(initializer.float_data, dtype=np.float32)
            coeff_tensor = coeff_data.reshape(initializer.dims)
        elif initializer.name == 'intercept':
            if initializer.raw_data:
                intercept_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            else:
                intercept_data = np.array(initializer.float_data, dtype=np.float32)
            intercept_tensor = intercept_data
    
    if coeff_tensor is None or intercept_tensor is None:
        raise ValueError("ONNX model missing required initializers (coefficients, intercept)")
    
    n_features = coeff_tensor.shape[0]
    coefficients = coeff_tensor.flatten()
    intercept = float(intercept_tensor[0])
    
    return {
        'algorithm': 'ridge',
        'n_features': n_features,
        'n_samples': 0,  # Not available from ONNX
        'intercept': intercept,
        'lambda': 0.0,  # Not available from ONNX
        'r_squared': 0.0,  # Not available from ONNX
        'mse': 0.0,  # Not available from ONNX
        'mae': 0.0,  # Not available from ONNX
        'coefficients': coefficients.astype(np.float32)
    }


def import_onnx_to_linear_regression(onnx_path: str) -> Dict[str, Any]:
    """Import ONNX model to Linear Regression format"""
    model = import_onnx_to_ridge(onnx_path)
    model['algorithm'] = 'linear_regression'
    return model


def import_onnx_to_logistic_regression(onnx_path: str) -> Dict[str, Any]:
    """Import ONNX model to Logistic Regression format"""
    model_proto = onnx.load(onnx_path)
    
    # Extract weights and bias from initializers
    weights_tensor = None
    bias_tensor = None
    
    for initializer in model_proto.graph.initializer:
        if initializer.name == 'weights':
            if initializer.raw_data:
                weights_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            else:
                weights_data = np.array(initializer.float_data, dtype=np.float32)
            weights_tensor = weights_data.reshape(initializer.dims)
        elif initializer.name == 'bias':
            if initializer.raw_data:
                bias_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            else:
                bias_data = np.array(initializer.float_data, dtype=np.float32)
            bias_tensor = bias_data
    
    if weights_tensor is None or bias_tensor is None:
        raise ValueError("ONNX model missing required initializers (weights, bias)")
    
    n_features = weights_tensor.shape[0]
    weights = weights_tensor.flatten()
    bias = float(bias_tensor[0])
    
    return {
        'algorithm': 'logistic_regression',
        'n_features': n_features,
        'n_samples': 0,  # Not available from ONNX
        'bias': bias,
        'learning_rate': 0.0,  # Not available from ONNX
        'lambda': 0.0,  # Not available from ONNX
        'max_iters': 0,  # Not available from ONNX
        'final_loss': 0.0,  # Not available from ONNX
        'accuracy': 0.0,  # Not available from ONNX
        'weights': weights.astype(np.float32)
    }


# ============================================================================
# Main Export/Import Functions
# ============================================================================

def export_model_to_onnx(model_data: bytes, algorithm: str, output_path: str) -> str:
    """Export NeuronDB model to ONNX format"""
    # Deserialize based on algorithm
    if algorithm == 'ridge':
        model = deserialize_ridge_model(model_data)
        return export_ridge_to_onnx(model, output_path)
    elif algorithm == 'lasso':
        model = deserialize_lasso_model(model_data)
        return export_ridge_to_onnx(model, output_path)  # Same structure
    elif algorithm == 'linear_regression':
        model = deserialize_linear_regression_model(model_data)
        return export_linear_regression_to_onnx(model, output_path)
    elif algorithm == 'logistic_regression':
        model = deserialize_logistic_regression_model(model_data)
        return export_logistic_regression_to_onnx(model, output_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: ridge, lasso, linear_regression, logistic_regression")


def import_onnx_to_model(onnx_path: str, algorithm: str, output_path: str) -> str:
    """Import ONNX model to NeuronDB format"""
    # Import based on algorithm
    if algorithm == 'ridge':
        model = import_onnx_to_ridge(onnx_path)
        data = serialize_ridge_model(model)
    elif algorithm == 'lasso':
        model = import_onnx_to_ridge(onnx_path)
        model['algorithm'] = 'lasso'
        data = serialize_ridge_model(model)
    elif algorithm == 'linear_regression':
        model = import_onnx_to_linear_regression(onnx_path)
        data = serialize_linear_regression_model(model)
    elif algorithm == 'logistic_regression':
        model = import_onnx_to_logistic_regression(onnx_path)
        data = serialize_logistic_regression_model(model)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: ridge, lasso, linear_regression, logistic_regression")
    
    # Write to file
    with open(output_path, 'wb') as f:
        f.write(data)
    
    return output_path


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='NeuronDB ONNX Exchange (ONEX) - Import/Export models between NeuronDB and ONNX formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export: python neurondb_onex.py --export model.bin ridge output.onnx
  Import: python neurondb_onex.py --import model.onnx ridge output.bin
        """
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export NeuronDB model to ONNX format'
    )
    parser.add_argument(
        '--import',
        dest='import_mode',
        action='store_true',
        help='Import ONNX model to NeuronDB format'
    )
    parser.add_argument(
        'input_file',
        help='Input file (model_data_file for export, onnx_file for import)'
    )
    parser.add_argument(
        'algorithm',
        choices=['ridge', 'lasso', 'linear_regression', 'logistic_regression'],
        help='Model algorithm type'
    )
    parser.add_argument(
        'output_path',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    # Validate that either --export or --import is specified
    if not args.export and not args.import_mode:
        parser.error("Must specify either --export or --import")
    if args.export and args.import_mode:
        parser.error("Cannot specify both --export and --import")
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.export:
            # Export: NeuronDB -> ONNX
            with open(args.input_file, 'rb') as f:
                model_data = f.read()
            
            result_path = export_model_to_onnx(model_data, args.algorithm, args.output_path)
            print(f"SUCCESS: Model exported to {result_path}")
            sys.exit(0)
        
        elif args.import_mode:
            # Import: ONNX -> NeuronDB
            result_path = import_onnx_to_model(args.input_file, args.algorithm, args.output_path)
            print(f"SUCCESS: Model imported to {result_path}")
            sys.exit(0)
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

