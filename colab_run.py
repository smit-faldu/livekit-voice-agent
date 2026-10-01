"""Run the whole voice assistant on Google Colab (Linux) with ONE command.

In a Colab cell:
    !git clone https://github.com/smit-faldu/livekit-voice-agent.git
    %cd livekit-voice-agent
    %run colab_run.py

(`%run` instead of `!python` so the script can read Colab secrets.)

Colab secrets (key icon in the left sidebar, enable "Notebook access"):
    GOOGLE_API_KEY        required  (Gemini)
    Relay for the audio (WebRTC media), ONE of:
      CF_TURN_KEY_ID + CF_TURN_API_TOKEN     Cloudflare Realtime TURN (free tier)
      TURN_HOST + TURN_USERNAME + TURN_CREDENTIAL   any TURN server (e.g. metered.ca free tier)
    APP_ACCESS_KEY        optional (random one is generated and printed)

Why a TURN relay: Cloudflare quick tunnels only carry HTTP/WebSocket, which covers the
web page, /api/token and LiveKit signaling, but not WebRTC audio (UDP). Colab accepts no
inbound connections, so browser and Colab both connect OUT to the relay, which forwards
the (still end-to-end encrypted) audio between them.

What runs:
    livekit-server :7880  <- tunnel 1 (wss://...trycloudflare.com)  <- browser signaling
    server.py      :3000  <- tunnel 2 (https://...trycloudflare.com) <- browser page + token
    browser audio  <-> TURN relay <-> livekit-server (Colab)
Stop with the Colab stop button (interrupt): everything is shut down.
"""

import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = Path("/usr/local/bin")
LOGS = ROOT / "debug" / "colab-logs"
MODELS = {
    # STT (streaming, default STT_MODEL=nemo80)
    "models/sherpa/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8":
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8.tar.bz2",
}


def sh(cmd: str, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT, **kw)


def secret(name: str, required: bool = False) -> str:
    """Colab secret -> environment variable -> empty."""
    value = ""
    try:
        from google.colab import userdata  # only inside a Colab kernel (%run)

        value = userdata.get(name) or ""
    except Exception:
        pass
    value = value or os.getenv(name, "")
    if required and not value:
        raise SystemExit(f"missing secret {name}: add it in Colab's Secrets panel (key icon) and enable notebook access")
    return value


# ---- 1. tools ------------------------------------------------------------------

def install_tools():
    if not shutil.which("uv"):
        sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
        os.environ["PATH"] = f"{Path.home() / '.local/bin'}:{os.environ['PATH']}"

    node = shutil.which("node")
    version = subprocess.run([node, "-v"], capture_output=True, text=True).stdout if node else ""
    if not re.match(r"v(2[2-9]|[3-9]\d)\.", version):  # Vite 8 needs Node >= 22.12
        sums = urllib.request.urlopen("https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt").read().decode()
        tarball = re.search(r"(node-v[\d.]+-linux-x64\.tar\.xz)", sums).group(1)
        sh(f"curl -fsSL https://nodejs.org/dist/latest-v22.x/{tarball} | tar -xJ -C /opt")
        os.environ["PATH"] = f"/opt/{tarball.removesuffix('.tar.xz')}/bin:{os.environ['PATH']}"

    if not shutil.which("livekit-server"):
        sh("curl -sSL https://get.livekit.io | bash")

    if not shutil.which("cloudflared"):
        sh(f"curl -fsSL -o {BIN}/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
           f" && chmod +x {BIN}/cloudflared")


# ---- 2. python deps + models -----------------------------------------------------

def install_app():
    sh("uv sync")
    for target, url in MODELS.items():
        if not (ROOT / target).exists():
            (ROOT / target).parent.mkdir(parents=True, exist_ok=True)
            sh(f"curl -fsSL {url} | tar -xj -C {(ROOT / target).parent}")
    if not (ROOT / "models/piper/en_US-lessac-low.onnx").exists():
        sh("uv run python -m piper.download_voices --download-dir models/piper en_US-lessac-low")


# ---- 3. config -------------------------------------------------------------------

def turn_servers() -> list[dict]:
    """TURN relay(s) handed to clients by LiveKit (rtc.turn_servers)."""
    key_id, token = secret("CF_TURN_KEY_ID"), secret("CF_TURN_API_TOKEN")
    if key_id and token:
        req = urllib.request.Request(
            f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
            data=json.dumps({"ttl": 86400}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        ice = json.load(urllib.request.urlopen(req))["iceServers"]
        ice = ice if isinstance(ice, list) else [ice]
        cred = next(s for s in ice if s.get("username"))
        user, password = cred["username"], cred["credential"]
        return [
            {"host": "turn.cloudflare.com", "port": 3478, "protocol": "udp", "username": user, "credential": password},
            {"host": "turn.cloudflare.com", "port": 443, "protocol": "tls", "username": user, "credential": password},
        ]
    host = secret("TURN_HOST")
    if host:
        user, password = secret("TURN_USERNAME", True), secret("TURN_CREDENTIAL", True)
        return [
            {"host": host, "port": 80, "protocol": "udp", "username": user, "credential": password},
            {"host": host, "port": 443, "protocol": "tls", "username": user, "credential": password},
        ]
    print("WARNING: no TURN relay configured: the page will load but audio will NOT connect "
          "(see the secrets list at the top of colab_run.py).", flush=True)
    return []


def check_outbound_udp():
    """The relay needs Colab -> internet UDP. Send one STUN binding request to Google."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(3)
            s.sendto(b"\x00\x01\x00\x00\x21\x12\xa4\x42" + secrets.token_bytes(12), ("stun.l.google.com", 19302))
            s.recvfrom(1024)
        print("outbound UDP: ok", flush=True)
    except OSError:
        print("WARNING: outbound UDP looks blocked; only the TLS relay (port 443) can work.", flush=True)


def write_config(api_key: str, api_secret: str, turns: list[dict]):
    lines = [
        "port: 7880",
        "rtc:",
        "  tcp_port: 7881",
        "  port_range_start: 50000",
        "  port_range_end: 60000",
        "  use_external_ip: false",  # Colab's public IP is not reachable anyway; the relay is
    ]
    if turns:
        lines.append("  turn_servers:")
        for t in turns:
            lines += [f"    - host: {t['host']}", f"      port: {t['port']}", f"      protocol: {t['protocol']}",
                      f"      username: {json.dumps(t['username'])}", f"      credential: {json.dumps(t['credential'])}"]
    lines += ["keys:", f"  {api_key}: {api_secret}"]
    (ROOT / "livekit.yaml").write_text("\n".join(lines) + "\n")


def write_env(api_key: str, api_secret: str, access_key: str, public_lk_url: str):
    (ROOT / ".env.local").write_text("\n".join([
        "LIVEKIT_URL=ws://127.0.0.1:7880",          # agent -> local server
        f"LIVEKIT_PUBLIC_URL={public_lk_url}",       # browser -> server through the tunnel
        f"LIVEKIT_API_KEY={api_key}",
        f"LIVEKIT_API_SECRET={api_secret}",
        f"GOOGLE_API_KEY={secret('GOOGLE_API_KEY', True)}",
        f"APP_ACCESS_KEY={access_key}",
    ]) + "\n")


# ---- 4. processes ----------------------------------------------------------------

procs: list[subprocess.Popen] = []


def start(name: str, cmd: list[str], env: dict | None = None) -> Path:
    log = LOGS / f"{name}.log"
    procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env))
    return log


def wait_for(log: Path, pattern: str, timeout: int = 180) -> re.Match:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if m := re.search(pattern, log.read_text(errors="ignore")):
            return m
        if any(p.poll() is not None for p in procs):
            break
        time.sleep(1)
    raise SystemExit(f"timed out waiting for {pattern!r}; last lines of {log}:\n" + "\n".join(log.read_text(errors="ignore").splitlines()[-25:]))


def tunnel(name: str, port: int) -> str:
    log = start(f"tunnel-{name}", ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"])
    return wait_for(log, r"https://[a-z0-9-]+\.trycloudflare\.com").group(0)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    install_tools()
    install_app()
    check_outbound_udp()

    api_key, api_secret = "API" + secrets.token_hex(6), secrets.token_urlsafe(32)  # fresh each run
    access_key = secret("APP_ACCESS_KEY") or secrets.token_urlsafe(16)
    write_config(api_key, api_secret, turn_servers())

    lk_log = start("livekit", ["livekit-server", "--config", "livekit.yaml", "--bind", "0.0.0.0"])
    wait_for(lk_log, r"starting LiveKit server", timeout=60)
    lk_public = tunnel("livekit", 7880).replace("https://", "wss://")
    write_env(api_key, api_secret, access_key, lk_public)

    app_log = start("app", ["uv", "run", "server.py"], env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    app_public = tunnel("app", 3000)
    wait_for(app_log, r"registered worker", timeout=600)  # first run builds the frontend

    print("\n" + "=" * 70)
    print(f"  OPEN:        {app_public}")
    print(f"  ACCESS KEY:  {access_key}")
    print(f"  (LiveKit signaling via {lk_public})")
    print("  Logs: debug/colab-logs/*.log  |  stop: interrupt this cell")
    print("=" * 70 + "\n", flush=True)

    # Stream the app log (turn latency lines etc.) until a process dies or the cell is interrupted.
    with app_log.open(errors="ignore") as f:
        f.seek(0, 2)
        while all(p.poll() is None for p in procs):
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.5)
    print("a process exited; see debug/colab-logs/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
