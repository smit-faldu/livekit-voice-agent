"""One command runs the whole app (besides livekit-server):

    uv run server.py      ->  http://localhost:3000

One Python process holds:
  - the React frontend (built into frontend/dist, served as static files)
  - the FastAPI token API (/api/token)
  - the LiveKit agent worker (VAD + Whisper + LangGraph/Gemini + Piper)
"""

import asyncio
import logging
import os
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
HOST, PORT = "127.0.0.1", 3000

# Privacy: all models are already in models/, so forbid Hugging Face network lookups.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
for noisy in ("httpx", "google_genai", "faster_whisper", "piper", "livekit", "livekit.agents.telemetry"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("livekit.agents").setLevel(logging.INFO)  # worker registered, job received, errors
log = logging.getLogger("server")


def build_frontend():
    """Build the React app if dist/ is missing or older than the sources."""
    dist_index = FRONTEND / "dist" / "index.html"
    sources = [p for p in (FRONTEND / "src").rglob("*") if p.is_file()] + [FRONTEND / "index.html"]
    if dist_index.exists() and dist_index.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return
    npm = shutil.which("npm")
    if not npm:
        raise SystemExit("npm not found: install Node.js to build the frontend")
    log.info("building frontend (first run or sources changed)...")
    if not (FRONTEND / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)


def ensure_port_free():
    # Fail fast with a clear message; otherwise uvicorn exits late while an old server keeps answering.
    with socket.socket() as s:
        if s.connect_ex((HOST, PORT)) == 0:
            raise SystemExit(f"port {PORT} is already in use: stop the other server (old uvicorn / server.py) first")


ensure_port_free()
build_frontend()  # before importing the API, which mounts frontend/dist

import uvicorn  # noqa: E402

from agent.main import server as agent_server  # noqa: E402
from backend.api import app  # noqa: E402


@asynccontextmanager
async def lifespan(_app):
    # The agent worker runs as a background task in the same event loop as FastAPI.
    # Jobs (conversations) run as threads in this process and share the loaded models.
    worker = asyncio.create_task(agent_server.run(devmode=False))
    log.info("ready: open http://localhost:%d", PORT)
    try:
        yield
    finally:
        await agent_server.aclose()
        worker.cancel()


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
