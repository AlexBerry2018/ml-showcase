import httpx
from typing import AsyncGenerator

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120)

    async def generate(self, prompt, model="llama3.2:3b", temperature=0.7, max_tokens=512, stream=False):
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, "stream": stream}
        if stream:
            return self._stream(url, payload)
        resp = await self.client.post(url, json=payload)
        return resp.json()["response"]

    async def _stream(self, url, payload):
        async with self.client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break

    async def chat(self, messages, model="llama3.2:3b", temperature=0.7):
        resp = await self.client.post(f"{self.base_url}/api/chat", json={
            "model": model, "messages": messages, "temperature": temperature, "stream": False
        })
        return resp.json()["message"]["content"]

    async def list_models(self):
        resp = await self.client.get(f"{self.base_url}/api/tags")
        return [m["name"] for m in resp.json().get("models", [])]

    async def close(self):
        await self.client.aclose()