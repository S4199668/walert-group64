# HTTP client for the reranker service (the only separate model service).
import httpx
import numpy as np

import config

_http = httpx.Client(timeout=180)


def rerank(query, documents):
    r = _http.post(f"{config.RERANK_URL}/rerank", json={"query": query, "documents": documents})
    r.raise_for_status()
    return np.asarray(r.json()["scores"], dtype=np.float32) 