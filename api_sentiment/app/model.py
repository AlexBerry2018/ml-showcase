from transformers import pipeline
import torch

class SentimentModel:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            device = 0 if torch.cuda.is_available() else -1
            cls._instance.pipeline = pipeline("sentiment-analysis", device=device)
        return cls._instance

model = SentimentModel()