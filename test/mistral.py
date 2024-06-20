from transformers import pipeline
from huggingface_hub import login

# Authenticate with your Hugging Face API token
login("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")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
chatbot(messages)