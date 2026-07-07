/*!
Expert Model Registry — three-tier memory manager.

  🔥 Hot  — expert model constructed and resident (VRAM via candle device)
  🟡 Warm — evicted from hot but kept in the LRU cache (fast re-promotion,
            no re-download or re-mmap needed)
  💤 Cold — only the config is known; nothing constructed yet. Promotion
            downloads (or hits the HF Hub disk cache) and mmaps the checkpoint.

Since each expert is now a full standalone pretrained model rather than a
tiny LoRA delta, promotion/demotion moves much more data than the original
LoRA-adapter design — this is the accepted tradeoff for using off-the-shelf
checkpoints instead of training our own adapters (see config.rs comments).
*/

use std::{
    collections::HashMap,
    fs,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result};
use candle_core::Device;
use lru::LruCache;

use crate::config::ExpertConfig;
use super::model::ExpertModel;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelTier {
    Hot,
    Warm,
    Cold,
}

struct RegistryInner {
    hot: HashMap<String, Arc<ExpertModel>>,
    warm: LruCache<String, Arc<ExpertModel>>,
    /// Cold: config known, nothing constructed — promotion builds fresh
    cold_configs: HashMap<String, ExpertConfig>,
    device: Device,
}

pub struct ExpertRegistry {
    inner: Arc<RwLock<RegistryInner>>,
}

impl ExpertRegistry {
    pub fn new(_adapters_dir: PathBuf, warm_cache_size: usize, device: Device) -> Result<Self> {
        let warm = LruCache::new(NonZeroUsize::new(warm_cache_size).unwrap());
        let inner = RegistryInner {
            hot: HashMap::new(),
            warm,
            cold_configs: HashMap::new(),
            device,
        };

        Ok(Self { inner: Arc::new(RwLock::new(inner)) })
    }

    /// Register an expert's config. No loading happens until `activate()`.
    pub fn register_cold(&self, name: &str, cfg: ExpertConfig) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        inner.cold_configs.insert(name.to_string(), cfg);
        tracing::debug!("Registered expert: {name}");
        Ok(())
    }

    /// Promote an expert to Hot. Loads through warm tier if needed, or
    /// constructs fresh from its HF Hub checkpoint (cached on disk after the
    /// first pull).
    pub fn activate(&self, name: &str) -> Result<Arc<ExpertModel>> {
        let mut inner = self.inner.write().unwrap();

        if let Some(m) = inner.hot.get(name) {
            return Ok(Arc::clone(m));
        }

        if let Some(m) = inner.warm.pop(name) {
            tracing::debug!("Promoting {name}: warm → hot");
            inner.hot.insert(name.to_string(), Arc::clone(&m));
            return Ok(m);
        }

        tracing::info!("Loading {name}: cold → hot");
        let cfg = inner
            .cold_configs
            .get(name)
            .with_context(|| format!("No registered expert for '{name}'"))?
            .clone();

        let model = ExpertModel::load(&cfg, inner.device.clone())?;
        let arc = Arc::new(model);
        inner.hot.insert(name.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Demote an expert from Hot → Warm (frees VRAM, keeps the constructed
    /// model around in case it's needed again soon).
    pub fn hibernate(&self, name: &str) {
        let mut inner = self.inner.write().unwrap();
        if let Some(m) = inner.hot.remove(name) {
            tracing::debug!("Hibernating {name}: hot → warm");
            inner.warm.put(name.to_string(), m);
        }
    }

    /// Demote all hot experts except the given set.
    pub fn evict_except(&self, keep: &[&str]) {
        let mut inner = self.inner.write().unwrap();
        let to_evict: Vec<String> = inner
            .hot
            .keys()
            .filter(|k| !keep.contains(&k.as_str()))
            .cloned()
            .collect();

        for name in to_evict {
            if let Some(m) = inner.hot.remove(&name) {
                tracing::debug!("Evicting {name}: hot → warm");
                inner.warm.put(name.clone(), m);
            }
        }
    }

    pub fn tier(&self, name: &str) -> ModelTier {
        let inner = self.inner.read().unwrap();
        if inner.hot.contains_key(name)   { ModelTier::Hot  }
        else if inner.warm.contains(name)  { ModelTier::Warm }
        else                                { ModelTier::Cold }
    }
}
