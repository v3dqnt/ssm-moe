# ssm-moe

Mixture-of-experts inference engine over Mamba (SSM) checkpoints, with an OpenAI-compatible HTTP endpoint for plugging into Vivianne (or any other harness that speaks OpenAI's chat-completions shape against a custom base URL).

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  ssm-moe.exe --serve                                                    │
│                                                                         │
│   POST /v1/chat/completions ──► MoEPipeline (one shared instance)      │
│                                    │                                    │
│                                    ▼                                    │
│   ┌──────────┐    ┌────────────┐   ┌──────────────┐   ┌────────────┐  │
│   │ Brain    │───▶│ Adaptive-K │──▶│ Expert       │──▶│ Critic     │  │
│   │ Router   │    │ Gate       │   │ Router       │   │ (verifier) │  │
│   │ (BART    │    │            │   │              │   │            │  │
│   │ sidecar) │    │            │   │              │   │            │  │
│   └──────────┘    └────────────┘   └──────┬───────┘   └────────────┘  │
│                                            │                            │
│                                            ▼                            │
│                              llama.cpp router server                    │
│                       (--models-preset, --models-max, LRU eviction)     │
│                            │        │        │        │         │       │
│                            ▼        ▼        ▼        ▼         ▼       │
│                         coding   math   reasoning  general  creative    │
│                         (GGUF)  (GGUF)   (GGUF)    (GGUF)    (GGUF)    │
└────────────────────────────────────────────────────────────────────────┘
```

- **Brain Router**: `facebook/bart-large-mnli` zero-shot classifier running as a small Python sidecar (`router_server.py`) — scores each prompt against the expert pool. Temporary: replace with a native SSM router once one is trained.
- **Adaptive-K Gate**: entropy-based selection of how many experts to activate (default `k_max=1` for the 8GB VRAM budget).
- **Expert Router**: llama.cpp's built-in `--models-preset` router server hosts all experts in one persistent process, loading/evicting per `--models-max` with a graceful stop-timeout. Crash isolation is preserved: llama.cpp spawns each loaded model as its own child subprocess internally.
- **Experts**: pretrained Mamba-1 / Mamba-2 checkpoints (Codestral Mamba, PromptCoT-Math, OpenHermes-tuned Mamba-2.8B, mamba-chat) served as GGUF Q4_K_M for real int4 quantization on an 8GB GPU.
- **Critic**: small 130M Mamba-1 model, always CPU, verifies output coherence + safety and can trigger a re-route. Score head is currently untrained (see roadmap).

## Prerequisites

1. **llama.cpp Windows CUDA release** at `../llama.cpp/llama-server.exe` (b9910 or later — needs the `--models-preset` / `--models-max` router flags).
2. **GGUF checkpoints** at `../models/`:
   - `codestral-mamba-q4_k_m.gguf`
   - `promptcot-math-q4_k_m.gguf`
   - `openhermes-2.8b-q4_k_m.gguf`
   - `mamba-chat-q4_k_m.gguf`
3. **Router sidecar Python env** with `fastapi`, `uvicorn`, `transformers`, `torch` (see `router_server.py`).

## Running

**Serve mode (Vivianne / any OpenAI-compatible harness):**

```
# Terminal 1 — Brain sidecar
python router_server.py

# Terminal 2 — main engine
./target/release/ssm-moe.exe --serve --port 8090
```

Point your harness at `http://127.0.0.1:8090/v1` as an OpenAI-compatible base URL. Model id `ssm-moe`. Include a stable `user` field per session for context-memory isolation (or send an `x-session-id` header).

### Wiring into Vivianne (this repo)

Vivianne's provider registry already includes `ssm-moe` as an OpenAI-compatible provider with no default base URL (see `packages/gui/src-tauri/src/ai/provider_registry.rs`). To use it:

1. Start the two processes above (`router_server.py` + `ssm-moe.exe --serve`).
2. In Vivianne, add a model with these fields (via `cmd_agent_set_model` or the equivalent UI):
   - `provider`: `ssm-moe`
   - `api`: `openai-chat` (or whatever the `ApiKind::OpenAiChat` label resolves to in your UI)
   - `base_url`: `http://127.0.0.1:8090/v1` (required — no default, since the port is user-chosen)
   - `id` / `name`: `ssm-moe` (any label works; the router ignores the request's `model` field)
3. Set any non-empty API key (`local`, `unused`, anything) — the ssm-moe server ignores the Authorization header but Vivianne's OpenAI client requires *some* value to be present.

**One-shot / REPL mode:**

```
./target/release/ssm-moe.exe --prompt "write a function to reverse a linked list"
./target/release/ssm-moe.exe --interactive
```

## Development

```
cargo test        # 4 tests: gate math, config invariants, LoadMode -> ngl mapping
cargo build --release
```

## Known limitations

- **Critic's `score_head` is zero-initialized / untrained** — always outputs ~0.5, so every request currently exhausts `max_retries` before returning "best effort" output. Not a bug, an intentional stub; needs a real pass/fail-labeled training pass.
- **`reasoning` and `general` share the same GGUF** (OpenHermes-2.8B) until a distinct fit-for-purpose SSM lands for each.
- **Brain router is a Python sidecar (BART)** — candle has no BART implementation, so this is scaffolding until an SSM router is trained; that will make the whole engine a single Rust binary again.
- **Requests serialize behind a mutex** — by design at `k_max=1` (one GPU, one expert hot); parallel handling would just contend for the same VRAM.
</content>
