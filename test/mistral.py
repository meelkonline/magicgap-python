import requests
from transformers import pipeline
from huggingface_hub import login

login("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")

messages = [
    {"role": "user", "content": "Tu es un noble Français, tu parles de manière très châtiée. Qui es-tu ?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=4096)
output = chatbot(messages)
print(output)
