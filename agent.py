import asyncio, base64, hashlib, hmac, os, time, uuid
from typing import Optional, Dict, Any

import httpx, orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import subprocess

# ---- required env ----
WORKER_BASE_URL = os.environ["WORKER_BASE_URL"].rstrip("/")
INTERNAL_ADMIN_TOKEN = os.environ["INTERNAL_ADMIN_TOKEN"]
NODE_SHARED_SECRET = os.environ["NODE_SHARED_SECRET"]
NODE_ID = os.environ.get("NODE_ID", str(uuid.uuid4()))
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")  # reachable from Workers
MODEL_ID = os.environ["MODEL_ID"]                             # public model id (Radiance registry)
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "256"))

# vLLM runtime config
VLLM_MODEL = os.environ["VLLM_MODEL"]                         # hf repo id or local path
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_ARGS = os.environ.get("VLLM_ARGS", "--dtype auto")
VLLM_INTERNAL_API_KEY = os.environ.get("VLLM_INTERNAL_API_KEY", "")  # optional defense-in-depth

VLLM_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}"

# replay protection (in-memory; fine per-node)
NONCE_TTL_SEC = 300
_seen: Dict[str, float] = {}

# in-flight control (fast 429 at the edge node)
_sem = asyncio.Semaphore(MAX_INFLIGHT)

# vLLM process handle
_vllm_proc: Optional[subprocess.Popen] = None

app = FastAPI()

def now() -> int:
    return int(time.time())

def _clean_seen():
    t = time.time()
    dead = [k for k, exp in _seen.items() if exp <= t]
    for k in dead:
        _seen.pop(k, None)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hmac_b64(secret: str, msg: str) -> str:
    sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    return base64.b64encode(sig).decode()

def verify_worker_sig(req: Request, body: bytes) -> bool:
    ts_s = req.headers.get("x-edge-timestamp", "")
    nonce = req.headers.get("x-edge-nonce", "")
    sig = req.headers.get("x-edge-signature", "")
    if not ts_s or not nonce or not sig:
        return False
    try:
        ts = int(ts_s)
    except:
        return False
    if abs(now() - ts) > 60:
        return False

    _clean_seen()
    if nonce in _seen:
        return False
    _seen[nonce] = time.time() + NONCE_TTL_SEC

    path = req.url.path
    msg = f"{ts}.{nonce}.{req.method.upper()}.{path}.{sha256_hex(body)}"
    expected = hmac_b64(NODE_SHARED_SECRET, msg)
    return hmac.compare_digest(expected, sig)

async def post_worker(path: str, payload: dict, sign_body: bool) -> None:
    body = orjson.dumps(payload)
    headers = {
        "authorization": f"Bearer {INTERNAL_ADMIN_TOKEN}",
        "content-type": "application/json",
    }
    if sign_body:
        ts = now()
        nonce = str(uuid.uuid4())
        msg = f"{ts}.{nonce}.POST.{path}.{sha256_hex(body)}"
        headers["x-edge-timestamp"] = str(ts)
        headers["x-edge-nonce"] = nonce
        headers["x-edge-signature"] = hmac_b64(NODE_SHARED_SECRET, msg)

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{WORKER_BASE_URL}{path}", content=body, headers=headers)
        r.raise_for_status()

def start_vllm():
    global _vllm_proc
    if _vllm_proc and _vllm_proc.poll() is None:
        return

    cmd = ["bash", "-lc"]
    # vLLM OpenAI-compatible server via `vllm serve …` :contentReference[oaicite:3]{index=3}
    # bind to localhost; agent proxies
    api_key_flag = f" --api-key {VLLM_INTERNAL_API_KEY}" if VLLM_INTERNAL_API_KEY else ""
    full = f"vllm serve '{VLLM_MODEL}' --host {VLLM_HOST} --port {VLLM_PORT}{api_key_flag} {VLLM_ARGS}"
    cmd.append(full)

    _vllm_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

async def wait_vllm_ready(timeout_sec: int = 180):
    deadline = time.time() + timeout_sec
    headers = {}
    if VLLM_INTERNAL_API_KEY:
        headers["authorization"] = f"Bearer {VLLM_INTERNAL_API_KEY}"

    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{VLLM_BASE}/v1/models", headers=headers)
                if r.status_code == 200:
                    return
            except:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError("vLLM did not become ready in time")

@app.on_event("startup")
async def on_startup():
    # start vLLM inside the container
    start_vllm()
    await wait_vllm_ready()

    # Register node & advertise model capacity to the Worker
    await post_worker("/internal/nodes/register", {"id": NODE_ID, "base_url": PUBLIC_BASE_URL}, sign_body=False)
    await post_worker("/internal/nodes/models/set", {
        "node_id": NODE_ID,
        "models": [{"model_id": MODEL_ID, "max_concurrency": MAX_INFLIGHT}]
    }, sign_body=False)

    # heartbeat loop (keeps node "last_seen" fresh)
    async def heartbeat():
        while True:
            try:
                await post_worker("/internal/nodes/heartbeat", {"id": NODE_ID, "status": "healthy"}, sign_body=False)
            except:
                pass
            await asyncio.sleep(20)

    asyncio.create_task(heartbeat())

@app.get("/health")
async def health():
    return {"ok": True, "node_id": NODE_ID}

@app.get("/ready")
async def ready():
    try:
        await wait_vllm_ready(timeout_sec=2)
        return {"ok": True}
    except:
        return JSONResponse({"ok": False}, status_code=503)

def with_vllm_headers() -> dict:
    if VLLM_INTERNAL_API_KEY:
        return {"authorization": f"Bearer {VLLM_INTERNAL_API_KEY}"}
    return {}

async def report_usage(job_id: str, user_id: str, api_key_id: str, model_id: str,
                       status: str, prompt_tokens: int, completion_tokens: int,
                       ttft_ms: int, tokens_per_sec: float, error: Optional[str]):
    await post_worker("/internal/usage-report", {
        "job_id": job_id,
        "user_id": user_id,
        "api_key_id": api_key_id,
        "model_id": model_id,
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec,
        "error": error
    }, sign_body=True)

def force_include_usage(req_json: Dict[str, Any]) -> None:
    # include_usage handling is implemented in vLLM serving code, and behavior can differ by version :contentReference[oaicite:4]{index=4}
    so = req_json.get("stream_options") or {}
    so.setdefault("include_usage", True)
    req_json["stream_options"] = so

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.body()
    if not verify_worker_sig(req, body):
        return JSONResponse({"error": {"code": "unauthorized", "message": "bad signature"}}, status_code=401)

    # fast overload: don't queue locally (OpenRouter-style preference)
    if _sem.locked() and _sem._value <= 0:  # type: ignore
        return JSONResponse({"error": {"code": "overloaded", "message": "node at capacity"}}, status_code=429)

    async with _sem:
        job_id = req.headers.get("x-radiance-job-id", str(uuid.uuid4()))
        user_id = req.headers.get("x-radiance-user-id", "")
        api_key_id = req.headers.get("x-radiance-api-key-id", "")
        req_json = orjson.loads(body)
        model_id = req_json.get("model", MODEL_ID)

        is_stream = bool(req_json.get("stream"))
        headers = {"content-type": "application/json", **with_vllm_headers()}

        if is_stream:
            force_include_usage(req_json)
            started = time.time()
            first_byte_at: Optional[float] = None
            last_usage: Optional[dict] = None
            err: Optional[str] = None
            status = "succeeded"

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{VLLM_BASE}/v1/chat/completions", json=req_json, headers=headers) as upstream:
                    async def gen():
                        nonlocal first_byte_at, last_usage, err, status
                        keepalive_sec = 10
                        it = upstream.aiter_bytes()

                        while True:
                            try:
                                chunk = await asyncio.wait_for(it.__anext__(), timeout=keepalive_sec)
                            except asyncio.TimeoutError:
                                # SSE keep-alive comment
                                yield b":\n\n"
                                continue
                            except StopAsyncIteration:
                                break

                            if first_byte_at is None and chunk:
                                first_byte_at = time.time()

                            # capture usage if present in any SSE data lines
                            for line in chunk.split(b"\n"):
                                if line.startswith(b"data: "):
                                    data = line[6:].strip()
                                    if data and data != b"[DONE]":
                                        try:
                                            obj = orjson.loads(data)
                                            if isinstance(obj, dict) and obj.get("usage"):
                                                last_usage = obj["usage"]
                                        except:
                                            pass

                            yield chunk

                    resp = StreamingResponse(gen(), status_code=upstream.status_code,
                                            media_type=upstream.headers.get("content-type", "text/event-stream"))

            ttft_ms = int((first_byte_at - started) * 1000) if first_byte_at else 0
            prompt_tokens = int((last_usage or {}).get("prompt_tokens", 0))
            completion_tokens = int((last_usage or {}).get("completion_tokens", 0))
            tps = 0.0
            if first_byte_at and completion_tokens:
                dur = max(0.001, time.time() - first_byte_at)
                tps = completion_tokens / dur

            # report usage back to Worker (billing)
            await report_usage(job_id, user_id, api_key_id, model_id, status, prompt_tokens, completion_tokens, ttft_ms, tps, err)
            return resp

        # non-streaming
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{VLLM_BASE}/v1/chat/completions", json=req_json, headers=headers)
            data = r.json()

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        status = "succeeded" if r.status_code < 400 else "failed"
        error = None if r.status_code < 400 else (data.get("error") or str(data))

        await report_usage(job_id, user_id, api_key_id, model_id, status, prompt_tokens, completion_tokens, 0, 0.0, error)
        return JSONResponse(data, status_code=r.status_code)