from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from app.llm_client import OllamaClient
from pydantic import BaseModel
from typing import List

class GenRequest(BaseModel):
    prompt: str
    model: str = "llama3.2:3b"
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama3.2:3b"
    temperature: float = 0.7

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama = OllamaClient()
    yield
    await app.state.ollama.close()

app = FastAPI(lifespan=lifespan)

@app.get("/models")
async def get_models():
    return {"models": await app.state.ollama.list_models()}

@app.post("/generate")
async def generate(req: GenRequest):
    if req.stream:
        return StreamingResponse(app.state.ollama.generate(req.prompt, req.model, req.temperature, req.max_tokens, stream=True), media_type="text/plain")
    resp = await app.state.ollama.generate(req.prompt, req.model, req.temperature, req.max_tokens, stream=False)
    return {"response": resp}

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [m.dict() for m in req.messages]
    resp = await app.state.ollama.chat(messages, req.model, req.temperature)
    return {"response": resp}