# Official vLLM deployment image (OpenAI server capable) :contentReference[oaicite:5]{index=5}
FROM vllm/vllm-openai:latest

# Install agent deps
RUN pip install --no-cache-dir -r /dev/stdin <<'REQ'
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.27.2
orjson==3.10.12
REQ

WORKDIR /app
COPY agent.py /app/agent.py

EXPOSE 8080
CMD ["bash","-lc","uvicorn agent:app --host 0.0.0.0 --port 8080"]
