/*!
Trainable linear head — the shared "bring your own trained weights" hook
used by both the critic's reward head (`layers/critic.rs`) and the native
router (`brain/native_router.rs`).

Both consume llama.cpp's pooled sequence embeddings
(`LlamaContext::embeddings_seq_ith`) as input, so what's needed on top is
just `y = Wx + b`. Rather than pull in a tensor/ML framework (candle,
safetensors, ONNX) for that, this is a hand-rolled minimal binary format —
zero new dependencies, matching the reason candle got dropped in the first
place.

## File format

```
[magic:   4 bytes  = b"SMHD"]
[version: u32 LE   = 1]
[n_in:    u32 LE]              // must equal the embedding model's n_embd
[n_out:   u32 LE]              // e.g. 3 for the critic, n_experts for the router
[weights: n_in * n_out f32 LE] // row-major: n_out rows of n_in values each
[bias:    n_out f32 LE]
```

`scripts/export_linear_head.py` (numpy-only) writes this format from a
`(n_out, n_in)` weight matrix and `(n_out,)` bias vector — that's the
concrete target for however the head was actually trained (sklearn, PyTorch,
whatever produces a plain weight matrix at the end).
*/

use std::{fs, path::Path};

use anyhow::{bail, Context, Result};

const MAGIC: &[u8; 4] = b"SMHD";
const VERSION: u32 = 1;

pub struct LinearHead {
    n_in: usize,
    n_out: usize,
    /// row-major: weights[o * n_in + i]
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearHead {
    /// Load a trained head from disk. `expected_n_in` should be the
    /// embedding model's `n_embd` — mismatches are almost always a sign of
    /// pointing the config at the wrong head file, so this fails loudly
    /// rather than silently producing garbage.
    pub fn load(path: &Path, expected_n_in: usize) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read linear head at {}", path.display()))?;

        if bytes.len() < 16 {
            bail!("linear head file {} is too short to contain a header", path.display());
        }
        if &bytes[0..4] != MAGIC {
            bail!("linear head file {} has bad magic (not an SMHD file)", path.display());
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            bail!("linear head file {} has unsupported version {version}", path.display());
        }

        let n_in = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let n_out = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

        if n_in != expected_n_in {
            bail!(
                "linear head file {} was trained for n_in={n_in} but the model's \
                 embedding size is {expected_n_in} — wrong head for this model?",
                path.display()
            );
        }

        let weights_len = n_in * n_out;
        let expected_bytes = 16 + (weights_len + n_out) * 4;
        if bytes.len() != expected_bytes {
            bail!(
                "linear head file {} is {} bytes, expected {expected_bytes} for n_in={n_in} n_out={n_out}",
                path.display(),
                bytes.len()
            );
        }

        let mut offset = 16;
        let weights = read_f32_vec(&bytes, &mut offset, weights_len);
        let bias = read_f32_vec(&bytes, &mut offset, n_out);

        Ok(Self { n_in, n_out, weights, bias })
    }

    /// `y = Wx + b`. Panics if `x.len() != n_in` — a programming error at
    /// the call site (embedding size mismatches are caught at `load()`
    /// time), not a runtime condition worth a `Result` for.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n_in, "linear head input size mismatch");

        (0..self.n_out)
            .map(|o| {
                let row = &self.weights[o * self.n_in..(o + 1) * self.n_in];
                let dot: f32 = row.iter().zip(x).map(|(w, xi)| w * xi).sum();
                dot + self.bias[o]
            })
            .collect()
    }
}

fn read_f32_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = *offset + i * 4;
        out.push(f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()));
    }
    *offset += count * 4;
    out
}
