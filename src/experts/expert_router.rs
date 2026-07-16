/*!
Expert Router — single persistent llama-server process serving all experts.

Replaces the earlier design of spawning one `llama-server` subprocess per
activated expert (see git history: `registry.rs` + `llama_server.rs`) with
llama.cpp's own built-in multi-model router server, confirmed present in our
b9910 build via `tools/server/server-models.cpp` and `tools/server/README.md`
(not assumed — checked the actual `--help` output and source before
committing to this design):

  --models-preset <ini>   named model presets (routing alias -> GGUF path
                           + per-model args like n-gpu-layers)
  --models-max <n>        cap on simultaneously loaded models, LRU-evicted
                           with a graceful stop-timeout (not a hard kill)
  --models-autoload       load a model on first request that names it
                           (default: enabled — this is what we rely on;
                           we never call an explicit "load" endpoint)

Requests route by the `"model"` field in the JSON body via `/v1/completions`
(the OpenAI-compatible endpoint — legacy `/completion` is explicitly
documented as not participating in router model-selection).

Crash isolation is preserved: `server-models.cpp` spawns each loaded model
as its own child subprocess (`subprocess_create_ex`), so one expert crashing
does not affect others or the router process itself. We get the same
isolation as the old per-expert-subprocess design, just orchestrated by
llama.cpp's own (more mature) supervisor instead of our hand-rolled one.
*/

use std::{
    fs,
    process::{Child, Command, Stdio},
    time::Duration,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::{ExpertConfig, LoadMode, MoEConfig};

pub struct ExpertRouter {
    child: Child,
    http: reqwest::blocking::Client,
    base_url: String,
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: usize,
    temperature: f64,
}

#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    text: String,
}

impl ExpertRouter {
    pub fn spawn(config: &MoEConfig) -> Result<Self> {
        fs::create_dir_all(&config.sessions_dir)?;
        let preset_path = config.sessions_dir.join("experts_preset.ini");
        let ini = build_preset_ini(&config.experts)?;
        fs::write(&preset_path, ini)
            .with_context(|| format!("failed to write preset INI to {}", preset_path.display()))?;

        tracing::info!(
            "Spawning expert router server on port {} (models-max={}, preset={})",
            config.llama_server_base_port,
            config.k_max.max(1),
            preset_path.display()
        );

        let child = Command::new(&config.llama_server_exe)
            .arg("--models-preset").arg(&preset_path)
            .arg("--models-max").arg(config.k_max.max(1).to_string())
            .arg("--port").arg(config.llama_server_base_port.to_string())
            // Release VRAM automatically after idle time — production
            // hardening, not just an inference detail: an idle expert
            // shouldn't hold GPU memory indefinitely between conversations.
            .arg("--sleep-idle-seconds").arg("300")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn expert router llama-server")?;

        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(180)) // first load of a cold expert can be slow
            .build()?;

        let base_url = format!("http://127.0.0.1:{}", config.llama_server_base_port);
        let router = Self { child, http, base_url };
        router.wait_until_ready(Duration::from_secs(60))?;
        Ok(router)
    }

    fn wait_until_ready(&self, timeout: Duration) -> Result<()> {
        let health_url = format!("{}/health", self.base_url);
        let start = std::time::Instant::now();

        loop {
            if let Ok(resp) = self.http.get(&health_url).send() {
                if resp.status().is_success() {
                    tracing::info!("Expert router server is ready");
                    return Ok(());
                }
            }

            if start.elapsed() > timeout {
                anyhow::bail!("Expert router server did not become ready within {timeout:?}");
            }

            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Generate a completion from the named expert. The router loads the
    /// model on demand and evicts others per `--models-max` automatically —
    /// no explicit activate/hibernate call needed on our side, unlike the
    /// design this replaced.
    pub fn generate(
        &self,
        expert_name: &str,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Result<String> {
        let url = format!("{}/v1/completions", self.base_url);
        let resp: CompletionResponse = self
            .http
            .post(&url)
            .json(&CompletionRequest {
                model: expert_name,
                prompt,
                max_tokens: max_new_tokens,
                temperature,
            })
            .send()
            .with_context(|| format!("completion request to expert '{expert_name}' failed"))?
            .json()
            .with_context(|| format!("malformed completion response for expert '{expert_name}'"))?;

        resp.choices
            .into_iter()
            .next()
            .map(|c| c.text)
            .ok_or_else(|| anyhow::anyhow!("empty choices in completion response for '{expert_name}'"))
    }
}

impl Drop for ExpertRouter {
    fn drop(&mut self) {
        tracing::debug!("Stopping expert router server");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Build a `--models-preset` INI file from our expert configs. Each expert
/// becomes a named preset section; the section name is the routing alias
/// used in the `"model"` field of every request. `LoadMode::Cpu` maps to
/// `n-gpu-layers = 0` (the "creative" expert today); `LoadMode::Gpu` offloads
/// everything — llama.cpp caps this automatically if the model has fewer
/// layers or VRAM runs out.
fn build_preset_ini(experts: &[ExpertConfig]) -> Result<String> {
    let mut ini = String::from("version = 1\n\n[*]\nc = 4096\n\n");

    for expert in experts {
        let gguf_path = fs::canonicalize(&expert.gguf_path).with_context(|| {
            format!(
                "expert '{}' GGUF not found at {} — run the model download/conversion step first",
                expert.name,
                expert.gguf_path.display()
            )
        })?;

        // canonicalize() on Windows prepends a \\?\ UNC prefix; llama.cpp's
        // path handling doesn't expect it, so strip it back off.
        let gguf_str = gguf_path.display().to_string();
        let gguf_str = gguf_str.strip_prefix(r"\\?\").unwrap_or(&gguf_str);

        ini.push_str(&format!(
            "[{}]\nmodel = {}\nn-gpu-layers = {}\n\n",
            expert.name, gguf_str, ngl_for_load_mode(expert.load_mode)
        ));
    }

    Ok(ini)
}

/// `LoadMode::Cpu` maps to `n-gpu-layers = 0`; `LoadMode::Gpu` offloads
/// everything (llama.cpp caps this automatically if the model has fewer
/// layers or VRAM runs out). Pulled out as a pure function — unlike
/// `build_preset_ini`, which needs real GGUF files on disk to canonicalize
/// paths, this is unit-testable with no filesystem or process dependency.
fn ngl_for_load_mode(mode: LoadMode) -> u32 {
    match mode {
        LoadMode::Gpu => 999,
        LoadMode::Cpu => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_load_mode_disables_gpu_offload() {
        assert_eq!(ngl_for_load_mode(LoadMode::Cpu), 0);
    }

    #[test]
    fn gpu_load_mode_offloads_everything() {
        assert_eq!(ngl_for_load_mode(LoadMode::Gpu), 999);
    }
}
