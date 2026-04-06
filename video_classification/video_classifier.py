import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from decord import VideoReader, cpu
import numpy as np

class VideoClassifier:
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics"):
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label

    def classify(self, video_path: str, num_frames: int = 16, top_k: int = 5):
        vr = VideoReader(video_path, ctx=cpu(0))
        indices = np.linspace(0, len(vr)-1, num_frames).astype(np.int64)
        frames = vr.get_batch(indices).asnumpy()
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values).logits
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, top_k)
        return [{"label": self.labels[idx.item()], "confidence": top_probs[i].item()} for i, idx in enumerate(top_idxs)]

if __name__ == "__main__":
    vc = VideoClassifier()
    path = input("Enter video file path: ").strip()
    if path:
        print(vc.classify(path))