# ml-showcase

## Структура
- `text_sentiment/` – анализ тональности (Hugging Face)
- `speech_recognition/` – распознавание речи (Wav2Vec2)
- `image_classification/` – классификация изображений (ResNet-50)
- `video_classification/` – классификация видео (VideoMAE)
- `local_llm/` – локальный запуск Mistral-7B через Ollama
- `api_sentiment/` – FastAPI для модели тональности + тесты + CI
- `api_llm/` – FastAPI для LLM + тестирование галлюцинаций
- `recsys/` – рекомендательные системы (классика + two-tower)

## Установка общих зависимостей
```bash
pip install -r requirements.txt
```

## Запуск API тональности
```bash
cd api_sentiment
uvicorn app.main:app --reload
```