#!/usr/bin/env python3
"""Generate minimal ONNX models for robowbc-ort testing.

Usage:
    python gen_test_models.py

Creates test_identity.onnx in the same directory as this script.
"""

import os
import onnx
from onnx import TensorProto, helper


def gen_identity_model(path: str) -> None:
    """Create a minimal Identity model: input [1,4] -> output [1,4]."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "test_identity", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def gen_relu_model(path: str) -> None:
    """Create a Relu model: input [1,8] -> output [1,8]."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "test_relu", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def gen_dynamic_identity_model(path: str) -> None:
    """Create an Identity model with dynamic second dimension: input [1,N] -> output [1,N]."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, None])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, None])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "test_dynamic_identity", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_identity_model(os.path.join(script_dir, "test_identity.onnx"))
    gen_relu_model(os.path.join(script_dir, "test_relu.onnx"))
    gen_dynamic_identity_model(os.path.join(script_dir, "test_dynamic_identity.onnx"))
