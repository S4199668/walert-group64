import pickle
import threading
import uuid
from collections import defaultdict

import config
import db

_locks = defaultdict(threading.Lock)


def new_id() -> str:
    return uuid.uuid4().hex


def lock(sid: str):
    return _locks[sid]


def _path(sid: str):
    safe = "".join(ch for ch in sid if ch.isalnum() or ch in "-_")
    return config.SESSION_DIR / f"{safe}.pkl"


def load(sid: str) -> list:
    p = _path(sid)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return []


def save(sid: str, history: list):
    p = _path(sid)
    with open(p, "wb") as f:
        pickle.dump(history, f)
    db.upsert_session(sid, p, len(history))
