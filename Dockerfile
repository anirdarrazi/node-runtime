FROM vllm/vllm-openai AS base

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py ./agent.py

EXPOSE 3000

ENTRYPOINT ["uvicorn"]
CMD ["agent:app", "--host", "0.0.0.0", "--port", "3000"]
