import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

import clients
import config
import db


def _tok(s):
    return re.findall(r"\w+", s.lower())


class Retriever:
    def __init__(self):
        self._lock = threading.RLock()
        self.sources, self.texts = [], []
        self.matrix = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        self.bm25 = None

    def reload(self):
        sources, texts, matrix = db.load_all()
        bm25 = BM25Okapi([_tok(t) for t in texts]) if texts else None
        with self._lock:
            self.sources, self.texts, self.matrix, self.bm25 = sources, texts, matrix, bm25
        print(f"Index loaded: {len(texts)} chunks")

    def search(self, query: str, qvec: np.ndarray):
        with self._lock:
            sources, texts, matrix, bm25 = self.sources, self.texts, self.matrix, self.bm25
        if not texts:
            return []

        # BM25 top-N and cosine top-N, picked independently
        bm = np.asarray(bm25.get_scores(_tok(query)), dtype=np.float32)
        bm_top = [int(i) for i in np.argsort(-bm)[: config.CANDIDATES_PER_METHOD] if bm[i] > 0]
        cos = matrix @ qvec            # vectors are normalised, so dot = cosine similarity
        cos_top = [int(i) for i in np.argsort(-cos)[: config.CANDIDATES_PER_METHOD]]

        # Union (duplicates removed) -> reranker
        pool = list(dict.fromkeys(bm_top + cos_top))
        print(f"Candidates: bm25={len(bm_top)} cosine={len(cos_top)} union={len(pool)}")

        # Rerank via the reranker service
        scores = clients.rerank(query, [texts[i] for i in pool])

        # Confidence in 0-1: use the score as-is if the reranker already returns 0-1,
        # otherwise squash raw logits with a sigmoid.
        conf = scores if (scores.min() >= 0 and scores.max() <= 1) else 1 / (1 + np.exp(-scores))
        print("Rerank confidences:", np.round(np.sort(conf)[::-1], 3).tolist())

        # Keep every chunk whose confidence >= threshold (could be 0, could be all)
        results = []
        for j in np.argsort(-conf):
            if conf[j] < config.RERANK_THRESHOLD:
                break
            i = int(pool[j])
            results.append({"source": sources[i], "text": texts[i], "score": float(conf[j])})
        return results


retriever = Retriever()