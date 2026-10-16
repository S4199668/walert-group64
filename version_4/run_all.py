# Starts the reranker service (port 8002) and the main app (port 8000). Stop with Ctrl+C.
import os
import subprocess
import sys
import time

SERVICES = [("rerank_service:app", 8002), ("main:app", 8000)]
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
procs = []
try:
    for app, port in SERVICES:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port)], env=env))
        time.sleep(1)
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    pass
finally:
    for p in procs:
        p.terminate()