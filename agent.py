import asyncio
import base64
import codecs
import hashlib
import hmac
import json
import os
import random
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import httpx
import orjson
import inspect
import websockets

# If you pass HF_TOKEN, mirror it to HUGGING_FACE_HUB_TOKEN so huggingface_hub/vLLM can authenticate.
if os.getenv("HF_TOKEN") and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

def ws_connect(ws_url: str, headers: dict[str, str]):
    """Create a websockets.connect context manager with the correct header kw for the installed websockets version."""
    try:
        params = inspect.signature(websockets.connect).parameters
        if "extra_headers" in params:
            return websockets.connect(ws_url, extra_headers=headers)
        if "additional_headers" in params:  # forward-compat
            return websockets.connect(ws_url, additional_headers=headers)
    except Exception:
        pass
    # Fallback for older versions
    return websockets.connect(ws_url, extra_headers=headers)

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# -------------------------
# Required env vars
# -------------------------
WORKER_BASE_URL = os.environ["WORKER_BASE_URL"].rstrip("/")
INTERNAL_ADMIN_TOKEN = os.environ["INTERNAL_ADMIN_TOKEN"]
NODE_SHARED_SECRET = os.environ["NODE_SHARED_SECRET"]
MODEL_ID = os.environ["MODEL_ID"]

NODE_ID = os.environ.get("NODE_ID", str(uuid.uuid4()))
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "256"))

# vLLM runtime config (local inside container)
VLLM_MODEL = os.environ["VLLM_MODEL"]
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_ARGS = os.environ.get("VLLM_ARGS", "--dtype auto")
VLLM_INTERNAL_API_KEY = os.environ.get("VLLM_INTERNAL_API_KEY", "")

VLLM_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}"

# Public WS endpoint exposed by Worker which forwards to DO "/connect"
# If not set, we'll derive it from WORKER_BASE_URL:
#   https://<worker>/  -> wss://<worker>/do/nodebroker/connect
NODE_BROKER_WS_URL = os.environ.get("NODE_BROKER_WS_URL", "").strip()

# -------------------------
# Globals
# -------------------------
_vllm_proc: Optional[subprocess.Popen] = None
_vllm_log_task: Optional[asyncio.Task] = None

# in-flight control
_inflight_lock = asyncio.Lock()
_inflight = 0

# track DO invocations (rid -> task)
inflight_tasks: Dict[str, asyncio.Task] = {}


# -------------------------
# Utilities
# -------------------------
def now_sec() -> int:
    return int(time.time())


def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hmac_b64(secret: str, msg: str) -> str:
    sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    return base64.b64encode(sig).decode("ascii")


def hmac_b64url(secret: str, msg: str) -> str:
    sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")


def normalize_ws_base(url: str) -> str:
    """
    websockets client expects ws:// or wss://.
    If user provides http(s):// we convert.
    """
    u = url.strip()
    if u.startswith("https://"):
        return "wss://" + u[len("https://") :]
    if u.startswith("http://"):
        return "ws://" + u[len("http://") :]
    return u


def derive_nodebroker_ws_from_worker() -> str:
    """
    Derive from WORKER_BASE_URL:
      https://host -> wss://host/do/nodebroker/connect
    """
    base = WORKER_BASE_URL
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://") :]
    else:
        # if someone passed a bare host, assume wss
        ws_base = "wss://" + base
    return ws_base.rstrip("/") + "/do/nodebroker/connect"


def build_ws_url() -> str:
    """
    Construct the NodeBroker websocket URL with handshake params required by node_broker.ts:
      msg = `${ts}.${nonce}.CONNECT.${nodeId}`
      sig = hmac_b64url(shared_secret, msg)
    """
    base = NODE_BROKER_WS_URL or derive_nodebroker_ws_from_worker()
    base = normalize_ws_base(base).rstrip("/")

    ts = now_sec()
    nonce = str(uuid.uuid4())
    msg = f"{ts}.{nonce}.CONNECT.{NODE_ID}"
    sig = hmac_b64url(NODE_SHARED_SECRET, msg)

    return f"{base}?ts={ts}&nonce={nonce}&sig={sig}"


def vllm_headers() -> Dict[str, str]:
    if VLLM_INTERNAL_API_KEY:
        return {"authorization": f"Bearer {VLLM_INTERNAL_API_KEY}"}
    return {}


async def post_worker(path: str, payload: dict, *, sign_body: bool) -> None:
    """
    Posts to Worker internal endpoint with Bearer INTERNAL_ADMIN_TOKEN.
    Optionally signs the body with NODE_SHARED_SECRET to support replay protection.
    """
    body = orjson.dumps(payload)
    headers = {
        "authorization": f"Bearer {INTERNAL_ADMIN_TOKEN}",
        "content-type": "application/json",
    }

    if sign_body:
        ts = now_sec()
        nonce = str(uuid.uuid4())
        msg = f"{ts}.{nonce}.POST.{path}.{sha256_hex_bytes(body)}"
        headers["x-edge-timestamp"] = str(ts)
        headers["x-edge-nonce"] = nonce
        headers["x-edge-signature"] = hmac_b64(NODE_SHARED_SECRET, msg)

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{WORKER_BASE_URL}{path}", content=body, headers=headers)
        r.raise_for_status()


# -------------------------
# vLLM process management
# -------------------------
def start_vllm() -> None:
    global _vllm_proc
    if _vllm_proc and _vllm_proc.poll() is None:
        return

    api_key_flag = f" --api-key {VLLM_INTERNAL_API_KEY}" if VLLM_INTERNAL_API_KEY else ""
    full = f"vllm serve '{VLLM_MODEL}' --host {VLLM_HOST} --port {VLLM_PORT}{api_key_flag} {VLLM_ARGS}"

    # Using bash -lc matches your current Docker base image behavior
    _vllm_proc = subprocess.Popen(
        ["bash", "-lc", full],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


async def _tail_vllm_logs() -> None:
    """Optional: print vLLM logs to container stdout for debugging."""
    global _vllm_proc
    if not _vllm_proc or not _vllm_proc.stdout:
        return
    try:
        while True:
            line = await asyncio.to_thread(_vllm_proc.stdout.readline)
            if not line:
                await asyncio.sleep(0.2)
                continue
            print(f"[vllm] {line.rstrip()}")
    except asyncio.CancelledError:
        return
    except Exception:
        return


async def wait_vllm_ready(timeout_sec: int = 600) -> None:
    deadline = time.time() + timeout_sec
    headers = vllm_headers()
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{VLLM_BASE}/v1/models", headers=headers)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError("vLLM did not become ready in time")


def stop_vllm() -> None:
    global _vllm_proc
    if not _vllm_proc:
        return
    try:
        _vllm_proc.terminate()
    except Exception:
        pass
    _vllm_proc = None


# -------------------------
# Usage reporting
# -------------------------
async def report_usage(
    job_id: str,
    user_id: str,
    api_key_id: str,
    model_id: str,
    status: str,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: int,
    tokens_per_sec: float,
    error: Optional[str],
) -> None:
    await post_worker(
        "/internal/usage-report",
        {
            "job_id": job_id,
            "user_id": user_id,
            "api_key_id": api_key_id,
            "model_id": model_id,
            "status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tokens_per_sec,
            "error": error,
        },
        sign_body=True,
    )


# -------------------------
# Stream helpers
# -------------------------
def force_include_usage(req_json: Dict[str, Any]) -> None:
    so = req_json.get("stream_options") or {}
    so.setdefault("include_usage", True)
    req_json["stream_options"] = so


def extract_usage_from_sse_lines(lines: list[str]) -> Optional[dict]:
    """
    Attempts to find a JSON object with `usage` in data lines.
    """
    for line in lines:
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = orjson.loads(data)
            if isinstance(obj, dict) and obj.get("usage"):
                return obj["usage"]
        except Exception:
            pass
    return None


# -------------------------
# DO WebSocket handling
# -------------------------
async def try_begin_inflight() -> bool:
    global _inflight
    async with _inflight_lock:
        if _inflight >= MAX_INFLIGHT:
            return False
        _inflight += 1
        return True


async def end_inflight() -> None:
    global _inflight
    async with _inflight_lock:
        _inflight = max(0, _inflight - 1)


async def ws_ping_loop(ws) -> None:
    while True:
        await asyncio.sleep(20)
        async with _inflight_lock:
            inflight = _inflight
        payload = {"type": "ping", "t": now_sec(), "inflight": inflight, "max_inflight": MAX_INFLIGHT}
        try:
            await ws.send(orjson.dumps(payload).decode("utf-8"))
        except Exception:
            return


async def handle_do_invoke(ws, rid: str, path: str, body: Dict[str, Any], headers: Dict[str, str]) -> None:
    """
    Handle an invocation sent by NodeBroker:
      - Non-stream: send {type:"result"}
      - Stream: forward SSE chunks via {type:"sse"} and finish with {type:"done"}
    """
    started = time.time()
    first_token_at: Optional[float] = None

    if not await try_begin_inflight():
        try:
            await ws.send(
                orjson.dumps(
                    {
                        "type": "error",
                        "rid": rid,
                        "status": 429,
                        "error": {"code": "overloaded", "message": "node at capacity"},
                    }
                ).decode("utf-8")
            )
        except Exception:
            pass
        return

    job_id = headers.get("x-radiance-job-id", rid)
    user_id = headers.get("x-radiance-user-id", "")
    api_key_id = headers.get("x-radiance-api-key-id", "")
    model_id = body.get("model", MODEL_ID)
    is_stream = bool(body.get("stream"))

    upstream_headers = {"content-type": "application/json", **vllm_headers()}

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if not is_stream:
                r = await client.post(f"{VLLM_BASE}{path}", json=body, headers=upstream_headers)
                data = r.json()
                status_code = r.status_code

                try:
                    await ws.send(
                        orjson.dumps({"type": "result", "rid": rid, "status": status_code, "json": data}).decode("utf-8")
                    )
                except Exception:
                    pass

                usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))

                status_str = "succeeded" if status_code < 400 else "failed"
                error_msg = None if status_code < 400 else (data.get("error") if isinstance(data, dict) else str(data))

                await report_usage(
                    job_id, user_id, api_key_id, model_id,
                    status_str, prompt_tokens, completion_tokens,
                    0, 0.0, None if status_code < 400 else str(error_msg)
                )
                return

            # Streaming path
            force_include_usage(body)

            # Incremental UTF-8 decode avoids crashes if chunks split multi-byte sequences
            decoder = codecs.getincrementaldecoder("utf-8")()
            sse_line_buf = ""  # line buffer across chunks
            last_usage: Optional[dict] = None

            async with client.stream("POST", f"{VLLM_BASE}{path}", json=body, headers=upstream_headers) as upstream:
                if upstream.status_code != 200:
                    err_text = await upstream.aread()
                    msg = err_text.decode("utf-8", errors="replace")
                    try:
                        await ws.send(
                            orjson.dumps(
                                {
                                    "type": "error",
                                    "rid": rid,
                                    "status": upstream.status_code,
                                    "error": {"code": "upstream_error", "message": msg[:2000]},
                                }
                            ).decode("utf-8")
                        )
                    except Exception:
                        pass
                    await report_usage(job_id, user_id, api_key_id, model_id, "failed", 0, 0, 0, 0.0, msg[:2000])
                    return

                async for raw_chunk in upstream.aiter_bytes():
                    if not raw_chunk:
                        continue

                    if first_token_at is None:
                        first_token_at = time.time()

                    text = decoder.decode(raw_chunk)
                    if text:
                        # forward to DO as-is
                        try:
                            await ws.send(orjson.dumps({"type": "sse", "rid": rid, "chunk": text}).decode("utf-8"))
                        except Exception:
                            break

                        # parse SSE lines robustly across chunk boundaries
                        sse_line_buf += text
                        lines = []
                        while "\n" in sse_line_buf:
                            line, sse_line_buf = sse_line_buf.split("\n", 1)
                            lines.append(line)

                        u = extract_usage_from_sse_lines(lines)
                        if u:
                            last_usage = u

                # flush any remaining decoder state (rare)
                tail = decoder.decode(b"", final=True)
                if tail:
                    try:
                        await ws.send(orjson.dumps({"type": "sse", "rid": rid, "chunk": tail}).decode("utf-8"))
                    except Exception:
                        pass

            # tell DO we're done (DO will close SSE)
            try:
                await ws.send(orjson.dumps({"type": "done", "rid": rid}).decode("utf-8"))
            except Exception:
                pass

            ttft_ms = int((first_token_at - started) * 1000) if first_token_at else 0
            prompt_tokens = int((last_usage or {}).get("prompt_tokens", 0))
            completion_tokens = int((last_usage or {}).get("completion_tokens", 0))
            tps = 0.0
            if first_token_at and completion_tokens:
                dur = max(0.001, time.time() - first_token_at)
                tps = completion_tokens / dur

            await report_usage(job_id, user_id, api_key_id, model_id, "succeeded", prompt_tokens, completion_tokens, ttft_ms, tps, None)

    except asyncio.CancelledError:
        # DO cancel -> cancel task -> report cancelled
        try:
            await ws.send(
                orjson.dumps(
                    {"type": "error", "rid": rid, "status": 499, "error": {"code": "cancelled", "message": "cancelled"}}
                ).decode("utf-8")
            )
        except Exception:
            pass
        await report_usage(job_id, user_id, api_key_id, model_id, "cancelled", 0, 0, 0, 0.0, "cancelled")
    except Exception as e:
        try:
            await ws.send(
                orjson.dumps(
                    {"type": "error", "rid": rid, "status": 500, "error": {"code": "error", "message": str(e)}}
                ).decode("utf-8")
            )
        except Exception:
            pass
        await report_usage(job_id, user_id, api_key_id, model_id, "failed", 0, 0, 0, 0.0, str(e))
    finally:
        inflight_tasks.pop(rid, None)
        await end_inflight()


async def ws_listener(ws) -> None:
    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            text = message.decode("utf-8", errors="replace")
        else:
            text = message

        try:
            msg = json.loads(text)
        except Exception:
            continue

        mtype = msg.get("type")
        if mtype == "invoke":
            rid = msg["rid"]
            path = msg.get("path", "/v1/chat/completions")
            body = msg.get("body") or {}
            headers = msg.get("headers") or {}
            task = asyncio.create_task(handle_do_invoke(ws, rid, path, body, headers))
            inflight_tasks[rid] = task

        elif mtype == "cancel":
            rid = msg.get("rid")
            task = inflight_tasks.pop(rid, None)
            if task:
                task.cancel()


async def do_ws_loop() -> None:
    """
    Connect to NodeBroker and stay connected with reconnect/backoff.
    """
    backoff = 1.0
    while True:
        ws_url = build_ws_url()
        headers = {"x-radiance-node-id": NODE_ID}

        try:
            # websockets API changed: some versions use additional_headers, others use extra_headers.
            # We'll try the new name first and fall back.

            cm = ws_connect(ws_url, headers)

            async with cm as ws:
                # reset backoff on successful connect
                backoff = 1.0

                # hello handshake required by NodeBroker
                hello = {
                    "type": "hello",
                    "v": 1,
                    "node_id": NODE_ID,
                    "max_inflight": MAX_INFLIGHT,
                    "models": [{"model_id": MODEL_ID, "max_inflight": MAX_INFLIGHT}],
                }
                await ws.send(orjson.dumps(hello).decode("utf-8"))

                ping_task = asyncio.create_task(ws_ping_loop(ws))
                listener_task = asyncio.create_task(ws_listener(ws))

                done, pending = await asyncio.wait(
                    {ping_task, listener_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()

        except Exception as e:
            # reconnect with jittered exponential backoff
            sleep_for = min(20.0, backoff) * (0.8 + random.random() * 0.4)
            print(f"[node] ws disconnected ({type(e).__name__}: {e}); reconnecting in {sleep_for:.1f}s")
            await asyncio.sleep(sleep_for)
            backoff = min(20.0, backoff * 1.6)


# -------------------------
# FastAPI app (health/ready)
# -------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global _vllm_log_task

    # Start vLLM and wait until ready
    start_vllm()
    _vllm_log_task = asyncio.create_task(_tail_vllm_logs())
    await wait_vllm_ready()

    # Start NodeBroker WS loop
    ws_task = asyncio.create_task(do_ws_loop())

    try:
        yield
    finally:
        ws_task.cancel()
        if _vllm_log_task:
            _vllm_log_task.cancel()
        stop_vllm()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "node_id": NODE_ID, "model_id": MODEL_ID}


@app.get("/ready")
async def ready():
    try:
        await wait_vllm_ready(timeout_sec=2)
        return {"ok": True}
    except Exception:
        return JSONResponse({"ok": False}, status_code=503)