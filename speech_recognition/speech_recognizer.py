import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np

class SpeechRecognizer:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: str, target_sr: int = 16000) -> str:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        audio = audio / np.max(np.abs(audio))
        inputs = self.processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        return transcription.lower()

if __name__ == "__main__":
    sr = SpeechRecognizer()
    # sr.transcribe("sample.wav")
    print("Ready. Provide a .wav file.")