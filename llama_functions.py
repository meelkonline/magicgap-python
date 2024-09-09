import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from api_requests import ChatRequest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login("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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.float16)


def llama_ask(request: ChatRequest):
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct").to(device)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    with torch.no_grad():
        response = pipe(request.messages, max_new_tokens=150)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response
