curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama run mistral "Explain AI in one sentence"
ollama serve