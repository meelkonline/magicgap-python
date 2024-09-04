from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

from api_requests import ChatRequest

login("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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


def llama_ask(request: ChatRequest):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    response = pipe(request.messages, max_new_tokens=150)
    return response
