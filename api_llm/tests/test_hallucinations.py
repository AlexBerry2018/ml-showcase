import pytest
import httpx

BASE = "http://localhost:8000"

@pytest.mark.asyncio
async def test_factual():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/generate", json={"prompt": "Capital of France?", "temperature": 0.0})
        assert "paris" in r.json()["response"].lower()

@pytest.mark.asyncio
async def test_logical_trap():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/generate", json={"prompt": "How many months have 28 days?"})
        answer = r.json()["response"].lower()
        assert "february" not in answer or "12" in answer