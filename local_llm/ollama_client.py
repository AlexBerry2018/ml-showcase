import requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:16767"):
        self.base_url = base_url

    def generate(self, prompt, model="mistral", temperature=0.7, max_tokens=512):
        resp = requests.post(f"{self.base_url}/api/generate", json={
            "model": model, "prompt": prompt, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False
        })
        return resp.json()["response"]

    def chat(self, messages, model="mistral", temperature=0.7):
        resp = requests.post(f"{self.base_url}/api/chat", json={
            "model": model, "messages": messages, "temperature": temperature, "stream": False
        })
        return resp.json()["message"]["content"]

if __name__ == "__main__":
    client = OllamaClient()
    print(client.generate("What is the capital of Russia?"))