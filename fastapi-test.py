from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI(
    title="Ask Me Anything API",
    description="""
    E2E llamaindex streaming RAG lightweight models and tradeoff accuracy vs latency:
    1. EMBEDDINGS MODEL: INT4 Quantized BAAI/bge-base-en-v1.5
    2. LLM MODEL: phi3-mini-128k-instruct
    3. RERANKING MODEL: BAAI/bge-reranker-base
    Note: Swagger UI waits to complete response, use example {CURL} command in description of API for non-buffered live streaming result.
    """,
)

class QueryRequest(BaseModel):
    question: str

async def some_async_processing_function(question: str):
    # Simulate an async task with parts of a response being generated over time
    for i in range(30):
        await asyncio.sleep(1)  # Simulate a delay (e.g., I/O-bound operation)
        yield f"Part {i+1}: Processed question: {question}\n"

@app.post("/query")
async def query_llamaindex(request: QueryRequest):
    question = request.question

    async def response_stream():
        async for part in some_async_processing_function(question):
            yield part

    return StreamingResponse(response_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
