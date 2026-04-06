from transformers import pipeline
import torch
import time
from typing import Dict, List, Union

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("sentiment-analysis", model=model_name, device=device)
        print(f"Model loaded on {'GPU' if device==0 else 'CPU'}")

    def analyze_single(self, text: str) -> Dict[str, Union[str, float]]:
        start = time.time()
        result = self.classifier(text)[0]
        elapsed_ms = (time.time() - start) * 1000
        return {"label": result['label'], "confidence": result['score'], "inference_time_ms": round(elapsed_ms, 2)}

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        start = time.time()
        results = self.classifier(texts)
        elapsed_ms = (time.time() - start) * 1000
        out = []
        for i, r in enumerate(results):
            out.append({"text": texts[i][:50], "label": r['label'], "confidence": r['score']})
        print(f"Batch {len(texts)} texts: {elapsed_ms:.0f} ms total")
        return out

if __name__ == "__main__":
    sa = SentimentAnalyzer()
    print(sa.analyze_single("I love this movie!"))