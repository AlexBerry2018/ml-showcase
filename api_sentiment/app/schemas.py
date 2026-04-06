from pydantic import BaseModel, Field
from typing import Optional, List

class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=5000)
    texts: Optional[List[str]] = None

class PredictResponse(BaseModel):
    label: str
    confidence: float
    inference_time_ms: float

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total_time_ms: float