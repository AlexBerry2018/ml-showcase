import time
from fastapi import FastAPI, HTTPException
from app.model import model
from app.schemas import PredictRequest, PredictResponse, BatchResponse

app = FastAPI(title="Sentiment API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest):
    if req.texts:
        raise HTTPException(400, "Use /predict_batch for multiple texts")
    if not req.text:
        raise HTTPException(400, "Provide 'text' field")
    start = time.time()
    result = model.pipeline(req.text)[0]
    elapsed_ms = (time.time() - start) * 1000
    return PredictResponse(label=result['label'], confidence=result['score'], inference_time_ms=round(elapsed_ms,2))

@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(req: PredictRequest):
    if not req.texts:
        raise HTTPException(400, "Provide 'texts' list")
    start = time.time()
    results = model.pipeline(req.texts)
    total_ms = (time.time() - start) * 1000
    items = [PredictResponse(label=r['label'], confidence=r['score'], inference_time_ms=round(total_ms/len(req.texts),2)) for r in results]
    return BatchResponse(results=items, total_time_ms=round(total_ms,2))