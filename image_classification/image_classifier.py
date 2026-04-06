import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from typing import Union, List, Dict, Optional

def load_image_from_source(source: Union[str, bytes]) -> Optional[Image.Image]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        if isinstance(source, str) and source.startswith(('http://', 'https://')):
            with requests.get(source, headers=headers, timeout=10) as resp:
                resp.raise_for_status()
                if not resp.headers.get('Content-Type', '').startswith('image/'):
                    raise ValueError("Not an image")
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                return img
        elif isinstance(source, str):
            img = Image.open(source).convert('RGB')
            return img
        else:
            img = Image.open(BytesIO(source)).convert('RGB')
            return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels = self._load_labels()
        print(f"ResNet-50 on {self.device}, {len(self.labels)} classes")

    def _load_labels(self):
        try:
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            resp = requests.get(url, timeout=5)
            return [line.strip() for line in resp.text.splitlines()]
        except:
            return [f"Class_{i}" for i in range(1000)]

    def predict(self, image_source: Union[str, bytes], top_k: int = 5) -> Optional[List[Dict]]:
        img = load_image_from_source(image_source)
        if img is None:
            return None
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, top_k)
        return [{"label": self.labels[idx], "confidence": round(top_probs[i].item(), 4)} for i, idx in enumerate(top_idxs)]

if __name__ == "__main__":
    clf = ImageClassifier()
    src = input("Enter image URL or path: ").strip()
    if src:
        preds = clf.predict(src)
        if preds:
            for i, p in enumerate(preds, 1):
                print(f"{i}. {p['label']} — {p['confidence']*100:.2f}%")