/*!
Persistent SSM context memory.

The Mamba hidden state is a fixed-size tensor that summarises all prior context.
We write it to disk after every turn and restore it at the start of the next —
giving the Brain implicit conversation memory with zero extra tokens.

Struct on disk:
  [4 bytes: n_layers][for each layer: [conv_state bytes][ssm_state bytes]]
*/

use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// One layer's SSM hidden state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerState {
    /// Conv1d state — shape: (batch, d_inner, d_conv - 1)
    pub conv_state: Vec<f32>,
    pub conv_shape: Vec<usize>,
    /// SSM state — shape: (batch, d_inner, d_state)
    pub ssm_state: Vec<f32>,
    pub ssm_shape: Vec<usize>,
}

/// Full model context: one LayerState per layer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelState {
    pub layers: Vec<LayerState>,
}

impl ModelState {
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

pub struct ContextMemory {
    session_id: String,
    storage_dir: PathBuf,
}

impl ContextMemory {
    pub fn new(session_id: impl Into<String>, storage_dir: impl Into<PathBuf>) -> Result<Self> {
        let storage_dir = storage_dir.into();
        fs::create_dir_all(&storage_dir)?;
        Ok(Self { session_id: session_id.into(), storage_dir })
    }

    fn path(&self, key: &str) -> PathBuf {
        self.storage_dir.join(format!("{}_{}.bin", self.session_id, key))
    }

    /// Serialize a model's hidden state to disk.
    pub fn save(&self, key: &str, state: &ModelState) -> Result<()> {
        let bytes = bincode_encode(state)?;
        fs::write(self.path(key), &bytes)?;
        tracing::debug!("Saved {} state ({} bytes)", key, bytes.len());
        Ok(())
    }

    /// Restore a model's hidden state from disk. Returns empty state if not found.
    pub fn load(&self, key: &str) -> Result<ModelState> {
        let p = self.path(key);
        if !p.exists() {
            return Ok(ModelState::default());
        }
        let bytes = fs::read(&p)?;
        let state = bincode_decode(&bytes)?;
        Ok(state)
    }

    pub fn clear(&self, key: &str) -> Result<()> {
        let p = self.path(key);
        if p.exists() {
            fs::remove_file(p)?;
        }
        Ok(())
    }

    pub fn clear_all(&self) -> Result<()> {
        for entry in fs::read_dir(&self.storage_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&self.session_id) {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
}

// Minimal bincode-compatible encode/decode using serde_json for now.
// Replace with actual bincode crate for smaller/faster serialisation.
fn bincode_encode(state: &ModelState) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(state)?)
}

fn bincode_decode(bytes: &[u8]) -> Result<ModelState> {
    Ok(serde_json::from_slice(bytes)?)
}
