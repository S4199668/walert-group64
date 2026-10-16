# Reranker service  ->  port 8002  (zeroentropy/zerank-1-small, local GPU)
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

import config

model = None
lock = threading.Lock()   # one GPU job at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        config.RERANK_MODEL, device=device, trust_remote_code=True,
        model_kwargs={"dtype": torch.float16 if device == "cuda" else torch.float32},
    )
    print(f"Reranker loaded on {device}, dtype={next(model.model.parameters()).dtype}")
    yield


app = FastAPI(title="Reranker service", lifespan=lifespan)


class Req(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
def rerank(req: Req):
    pairs = [(req.query, d) for d in req.documents]
    out = []
    with lock, torch.inference_mode():
        for k in range(0, len(pairs), 4):          # small groups keep GPU memory low
            out.extend(np.asarray(model.predict(pairs[k:k + 4], batch_size=2)).reshape(-1).tolist())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"scores": out}


@app.get("/health")
def health():
    return {"ok": model is not None, "model": config.RERANK_MODEL}