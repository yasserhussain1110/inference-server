# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import hashlib
import aioredis
import openai

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI()

class InferenceRequest(BaseModel):
    text: str
    model_flag: str  # "openai" or "local"

# ------------------------
# Redis cache setup
# ------------------------
# redis = aioredis.from_url("redis://localhost", decode_responses=True)
redis = []

def cache_key(model: str, text: str) -> str:
    h = hashlib.sha256(text.encode()).hexdigest()
    return f"cache:{model}:{h}"

async def get_cached(model: str, text: str):
    return await redis.get(cache_key(model, text))

async def set_cache(model: str, text: str, value: str):
    await redis.set(cache_key(model, text), value, ex=3600)  # 1 hour TTL

# ------------------------
# Batching queue
# ------------------------
batch_queue = asyncio.Queue()
BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05  # 50 ms

async def batch_worker():
    while True:
        batch = []
        # Wait for at least one request
        item = await batch_queue.get()
        batch.append(item)

        # Collect more requests until BATCH_SIZE or timeout
        try:
            while len(batch) < BATCH_SIZE:
                item = await asyncio.wait_for(batch_queue.get(), timeout=BATCH_TIMEOUT)
                batch.append(item)
        except asyncio.TimeoutError:
            pass  # Timeout reached, process the batch

        # Extract texts and model flags
        texts = [req['text'] for req in batch]
        model_flag = batch[0]['model_flag']  # assume same model per batch for simplicity

        # Call model
        results = await route_to_model(model_flag, texts)

        # Set results for each request and cache
        for req, res in zip(batch, results):
            await set_cache(req['model_flag'], req['text'], res)
            req['future'].set_result(res)

# ------------------------
# Model routing
# ------------------------
async def route_to_model(model_flag: str, texts: list[str]) -> list[str]:
    if model_flag == "openai":
        return await call_openai(texts)
    else:
        return await call_local_llm(texts)

# ------------------------
# OpenAI call
# ------------------------
async def call_openai(texts: list[str]) -> list[str]:
    results = []
    for t in texts:
        resp = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": t}]
        )
        results.append(resp.choices[0].message.content)
    return results

# ------------------------
# Local LLM call placeholder
# ------------------------
async def call_local_llm(texts: list[str]) -> list[str]:
    # Replace this with your local LLM inference logic
    return [f"local model response for: {t}" for t in texts]

# ------------------------
# API endpoint
# ------------------------
@app.post("/infer")
async def infer(req: InferenceRequest):
    # Check cache first
    cached = await get_cached(req.model_flag, req.text)
    if cached:
        return {"result": cached}

    # Create a future for the batch worker to set result
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    # Push request to batch queue
    await batch_queue.put({
        "text": req.text,
        "model_flag": req.model_flag,
        "future": fut
    })

    # Wait for batch worker to process
    result = await fut
    return {"result": result}

# ------------------------
# Startup event: start batch worker
# ------------------------
@app.on_event("startup")
async def startup_event():
    print("Starting batch worker...")
    asyncio.create_task(batch_worker())
