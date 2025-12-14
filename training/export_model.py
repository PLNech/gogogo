#!/usr/bin/env python3
"""Export PyTorch Go model to ONNX and TensorFlow.js.

Pipeline: PyTorch → ONNX → TensorFlow.js

Usage:
    # Export to ONNX only
    poetry run python export_model.py --checkpoint checkpoints/go_9x9_final.pt --output models/go_9x9

    # Full pipeline including TFJS conversion
    poetry run python export_model.py --checkpoint checkpoints/go_9x9_final.pt --output models/go_9x9 --tfjs

After TFJS export, copy models/go_9x9_tfjs/ to public/models/ for browser use.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from model import GoNet
from config import Config


class GoNetExport(nn.Module):
    """Simplified GoNet wrapper for ONNX export.

    Only exports policy and value heads (no optional outputs).
    This is sufficient for playing - ownership is a training signal.
    """

    def __init__(self, model: GoNet):
        super().__init__()
        self.model = model
        self.board_size = model.board_size

    def forward(self, x):
        """Forward pass returning (policy_probs, value).

        Returns:
            policy: (batch, board_size^2 + 1) softmax probabilities
            value: (batch, 1) in [-1, 1]
        """
        # Run backbone
        h = torch.relu(self.model.bn_init(self.model.conv_init(x)))
        for block in self.model.res_blocks:
            h = block(h)

        # Policy head - return probabilities (not log_softmax)
        p = torch.relu(self.model.policy_bn(self.model.policy_conv(h)))
        p_flat = p.view(p.size(0), -1)
        policy = torch.softmax(self.model.policy_fc(p_flat), dim=1)

        # Value head
        v = torch.relu(self.model.value_bn(self.model.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.model.value_fc1(v))
        value = torch.tanh(self.model.value_fc2(v))

        return policy, value


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[GoNet, Config]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or create default
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"  Board size: {config.board_size}x{config.board_size}")
        print(f"  Input planes: {config.input_planes}")
        print(f"  Filters: {config.num_filters}, Blocks: {config.num_blocks}")
    else:
        # Infer from state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Try to infer board size from policy FC layer
        policy_fc_weight = state_dict.get('policy_fc.weight')
        if policy_fc_weight is not None:
            action_size = policy_fc_weight.shape[0]
            board_size = int((action_size - 1) ** 0.5)
            print(f"  Inferred board size: {board_size}x{board_size}")
            config = Config(board_size=board_size)
        else:
            print("  Using default config")
            config = Config()

    # Create and load model
    config.device = device
    model = GoNet(config)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    return model, config


def export_onnx(model: GoNet, config: Config, output_path: str) -> str:
    """Export model to ONNX format."""
    print(f"\nExporting to ONNX: {output_path}")

    # Wrap model for simplified export
    export_model = GoNetExport(model)
    export_model.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, config.input_planes, config.board_size, config.board_size)

    # Export to ONNX
    onnx_path = f"{output_path}.onnx"
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['board'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'board': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

    # Verify export
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Saved: {onnx_path} ({file_size:.2f} MB)")
    print(f"  Input shape: (batch, {config.input_planes}, {config.board_size}, {config.board_size})")
    print(f"  Output: policy (batch, {config.board_size ** 2 + 1}), value (batch, 1)")

    return onnx_path


def convert_to_tfjs(onnx_path: str, output_dir: str) -> str:
    """Convert ONNX model to TensorFlow.js format."""
    print(f"\nConverting to TensorFlow.js: {output_dir}")

    tfjs_dir = f"{output_dir}_tfjs"
    os.makedirs(tfjs_dir, exist_ok=True)

    # Use tensorflowjs_converter
    try:
        result = subprocess.run([
            sys.executable, "-m", "onnx_tf.frontend",
            "--onnx", onnx_path,
            "--output", f"{output_dir}_saved_model"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  onnx-tf conversion failed: {result.stderr}")
            # Try direct ONNX to TFJS via onnxruntime-web compatible format
            print("  Trying alternative: Keep ONNX for onnxruntime-web inference")
            # Copy ONNX to output for potential onnxruntime-web usage
            import shutil
            shutil.copy(onnx_path, f"{tfjs_dir}/model.onnx")

            # Also write a simple metadata file
            with open(f"{tfjs_dir}/metadata.json", "w") as f:
                import json
                json.dump({
                    "format": "onnx",
                    "backend": "onnxruntime-web",
                    "source": onnx_path
                }, f, indent=2)

            print(f"  Note: Use onnxruntime-web for browser inference")
            return tfjs_dir

        # Convert SavedModel to TFJS
        subprocess.run([
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            f"{output_dir}_saved_model",
            tfjs_dir
        ], check=True)

        print(f"  Saved: {tfjs_dir}/")
        return tfjs_dir

    except Exception as e:
        print(f"  TFJS conversion error: {e}")
        print("  You can manually convert using:")
        print(f"    pip install onnx-tf tensorflowjs")
        print(f"    python -m onnx_tf.tool.convert -t tf -i {onnx_path} -o {output_dir}_saved_model")
        print(f"    tensorflowjs_converter --input_format=tf_saved_model {output_dir}_saved_model {tfjs_dir}")
        return None


def test_onnx_inference(onnx_path: str, config: Config):
    """Test ONNX model inference."""
    print(f"\nTesting ONNX inference...")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)

        # Create test input (empty board with current player = black)
        import numpy as np
        test_input = np.zeros((1, config.input_planes, config.board_size, config.board_size), dtype=np.float32)
        # Set "current player to move" plane to 1 (typically last plane)
        test_input[0, -1, :, :] = 1.0

        # Run inference
        outputs = session.run(None, {'board': test_input})
        policy, value = outputs

        print(f"  Policy shape: {policy.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Value: {value[0, 0]:.4f}")
        print(f"  Top 5 policy moves:")
        top5 = np.argsort(policy[0])[-5:][::-1]
        for idx in top5:
            if idx == config.board_size ** 2:
                move = "PASS"
            else:
                row, col = idx // config.board_size, idx % config.board_size
                move = f"{chr(ord('A') + col)}{config.board_size - row}"
            print(f"    {move}: {policy[0, idx]:.4f}")

        print("  ONNX inference test passed!")
        return True

    except ImportError:
        print("  onnxruntime not installed, skipping test")
        print("  Install with: poetry add onnxruntime")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export Go model to ONNX/TFJS")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", required=True, help="Output path prefix (e.g., models/go_9x9)")
    parser.add_argument("--tfjs", action="store_true", help="Also convert to TensorFlow.js")
    parser.add_argument("--test", action="store_true", help="Test ONNX inference after export")
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint)

    # Export to ONNX
    onnx_path = export_onnx(model, config, args.output)

    # Test inference
    if args.test:
        test_onnx_inference(onnx_path, config)

    # Convert to TFJS
    if args.tfjs:
        tfjs_dir = convert_to_tfjs(onnx_path, args.output)
        if tfjs_dir:
            print(f"\nTo use in browser, copy {tfjs_dir}/ to public/models/")

    print("\nExport complete!")
    print(f"  ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
