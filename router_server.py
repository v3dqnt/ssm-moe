"""
Temporary Brain Router sidecar.

Wraps facebook/bart-large-mnli as a zero-shot classifier behind a tiny local
HTTP endpoint. The Rust pipeline calls this over localhost instead of running
BART natively, since candle has no BART support.

This is scaffolding: once a real SSM-based router is trained and loadable in
candle, this whole file goes away and router.rs talks to it directly.

Run:
    pip install fastapi uvicorn transformers torch
    python router_server.py
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

EXPERTS = ["coding", "math", "reasoning", "creative", "general"]
THRESHOLD = 0.65  # matches the threshold validated during dataset labeling

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    # CPU, not GPU: same reasoning as the Critic (layers/critic.rs) — every
    # byte of local 8GB VRAM should go to the active expert's GGUF, not
    # compete with a classifier that runs once per turn and isn't latency
    # critical.
    device=-1,
)


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    labels: list[str]
    scores: list[float]  # parallel to EXPERTS, in fixed order — not sorted


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    result = classifier(req.prompt, candidate_labels=EXPERTS, multi_label=True)

    # result["labels"]/["scores"] come back sorted by score descending;
    # re-order into EXPERTS' fixed order so the Rust side can zip against
    # its own expert index list without re-matching strings each call.
    score_by_label = dict(zip(result["labels"], result["scores"]))
    ordered_scores = [score_by_label[label] for label in EXPERTS]

    return RouteResponse(labels=EXPERTS, scores=ordered_scores)


@app.get("/health")
def health():
    return {"status": "ok", "experts": EXPERTS, "threshold": THRESHOLD}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)
