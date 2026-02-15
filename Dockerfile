FROM vllm/vllm-openai AS base

WORKDIR /workspace

# Install python dependencies required by the agent. Websockets is
# included to support the NodeBroker Durable Object integration.
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx orjson websockets

COPY agent.py ./agent.py

# Expose the port the FastAPI server listens on
EXPOSE 3000

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "3000"]