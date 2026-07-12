/*!
Persistent SSM context memory — path resolver only.

llama.cpp's own `LlamaContext::state_seq_save_file` / `state_seq_load_file`
(see `llama-cpp-2`'s `context/session.rs`) already serialize a sequence's
recurrent/KV state to a single opaque file — `LlamaStateSeqFlags::PARTIAL_ONLY`
is even documented upstream as existing specifically for "SWA KV cache or
recurrent cache (e.g. Mamba)". That does everything the old hand-rolled
`ModelState`/`LayerState` byte layout was trying to reimplement (and never
actually populated — `ExpertModel::generate()` used to always return
`ModelState::default()`).

So this module no longer owns a state *format* — it just hands out a
deterministic file path per `(session_id, key)` for `experts/model.rs` and
`layers/critic.rs` to save/load llama.cpp session state directly against.
*/

use std::{fs, path::PathBuf};

use anyhow::Result;

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

    /// Deterministic path for a given state key (e.g. an expert or critic
    /// name) within this session. Callers save/load llama.cpp session state
    /// directly against this path — it may or may not exist yet.
    pub fn path(&self, key: &str) -> PathBuf {
        self.storage_dir.join(format!("{}_{}.llama-state", self.session_id, key))
    }

    /// Whether a saved state file already exists for this key.
    pub fn exists(&self, key: &str) -> bool {
        self.path(key).exists()
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
