"""
Export a trained linear head (weight matrix + bias) to the SMHD binary
format `src/layers/linear_head.rs` loads.

Deliberately numpy-only — however you trained the head (sklearn, PyTorch,
plain gradient descent), get it down to a plain (n_out, n_in) weight matrix
and (n_out,) bias vector and hand them to `export()`. No torch/sklearn
dependency here, to keep this script runnable without whatever framework you
used to train.

Format:
    [magic:   4 bytes  = b"SMHD"]
    [version: u32 LE   = 1]
    [n_in:    u32 LE]
    [n_out:   u32 LE]
    [weights: n_in * n_out f32 LE, row-major: n_out rows of n_in values each]
    [bias:    n_out f32 LE]

Usage:
    import numpy as np
    from export_linear_head import export
    export("critic_head.bin", weights, bias)

`weights` must be shape (n_out, n_in) and `bias` shape (n_out,) — e.g. for
the critic head n_out=3 (coherence, completion, safety); for the native
router n_out=n_experts.
"""

import struct

import numpy as np

MAGIC = b"SMHD"
VERSION = 1


def export(path: str, weights: np.ndarray, bias: np.ndarray) -> None:
    weights = np.asarray(weights, dtype="<f4")
    bias = np.asarray(bias, dtype="<f4")

    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D (n_out, n_in), got shape {weights.shape}")
    n_out, n_in = weights.shape

    if bias.shape != (n_out,):
        raise ValueError(f"bias must be shape ({n_out},), got {bias.shape}")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", n_in))
        f.write(struct.pack("<I", n_out))
        f.write(weights.tobytes(order="C"))  # row-major: matches Rust's row-major read
        f.write(bias.tobytes(order="C"))

    print(f"Wrote {path}: n_in={n_in} n_out={n_out} ({weights.nbytes + bias.nbytes} bytes of weights)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_npy", help="path to a .npy file, shape (n_out, n_in)")
    parser.add_argument("bias_npy", help="path to a .npy file, shape (n_out,)")
    parser.add_argument("out_path", help="output .bin path (e.g. critic_head.bin)")
    args = parser.parse_args()

    export(args.out_path, np.load(args.weights_npy), np.load(args.bias_npy))
