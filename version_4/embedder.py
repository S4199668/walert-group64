from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain_openai import OpenAIEmbeddings

import config

_emb = OpenAIEmbeddings(model=config.EMBED_MODEL, dimensions=config.EMBED_DIM)


def _norm(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-12)


def embed_documents(texts, batch=64, workers=4):
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    batches = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(_emb.embed_documents, batches))
    return _norm([v for b in out for v in b])


def embed_query(text):
    return _norm(_emb.embed_query(text))