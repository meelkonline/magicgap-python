import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
from huggingface_hub import login
from api_requests import ChatRequest
from nlp_functions import limit_to_sentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Delayed Login (not done on startup)
def lazy_login():
    login("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")


# Lazy-loaded model and tokenizer
llama_tokenizer = None
llama_model = None


def get_llama_model():
    global llama_tokenizer, llama_model
    if llama_model is None or llama_tokenizer is None:
        print("🔹 Loading Llama-3.2-3B-Instruct into memory...")
        lazy_login()  # Ensure Hugging Face authentication

        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=quant_config
        )

        llama_model.eval()  # Ensure model is in evaluation mode
        print("✅ Model Loaded Successfully!")

    return llama_tokenizer, llama_model


class StopAtFirstValidJSON(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        self.buffer += self.tokenizer.decode(input_ids[0][-1:], skip_special_tokens=True)
        if self.buffer.strip().endswith("}"):
            try:
                json.loads(self.buffer.strip())
                return True
            except json.JSONDecodeError:
                pass
        return False


class MaxWordStoppingCriteria(StoppingCriteria):
    def __init__(self, max_words):
        self.max_words = max_words
        self.current_words = 0

    def __call__(self, input_ids, scores, **kwargs):
        tokenizer, _ = get_llama_model()  # Ensure tokenizer is loaded
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.current_words = len(decoded_text.split())
        if self.current_words >= self.max_words:
            if decoded_text.strip()[-1] in {'.', '!', '?'} and not decoded_text.strip().endswith("..."):
                return True
        return False


def format_response(generated_text: str, max_sentences=3, max_words=None) -> str:
    """
    Format the response by:
    - Stopping at <|assistant|> tag
    - Limiting to a specific number of sentences
    - Merging sentences not exceeding max_words
    """
    if "<|assistant|>" in generated_text:
        generated_text = generated_text.split("<|assistant|>")[0]

    if "<|user|>" in generated_text:
        generated_text = generated_text.split("<|user|>")[0]

    formatted_text = limit_to_sentences(generated_text.strip(), max_sentences)

    if max_words is not None:
        words = formatted_text.split()
        if len(words) > max_words:
            formatted_text = " ".join(words[:max_words])

    return formatted_text


def llama_generate_response(request: ChatRequest, max_new_tokens=100, num_beams=1, temperature=0.1) -> str:
    """
    Generate a response from the model with customizable parameters.
    """
    tokenizer, model = get_llama_model()  # Ensure the model is loaded

    tokenizer_input = tokenizer(request.messages, return_tensors="pt")
    input_ids = tokenizer_input.input_ids.cuda()
    attention_mask = tokenizer_input.attention_mask.cuda()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            stopping_criteria=StoppingCriteriaList([MaxWordStoppingCriteria(max_words=50)]),
            top_k=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = input_ids.size(1)
    generated_tokens = outputs[:, prompt_length:]

    new_text = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True).strip()
    print(f"Generated raw response: {new_text}")

    return format_response(new_text)


def llama32_3b_ask(request: ChatRequest) -> str:
    """Function specific to the 'ask' scenario with predefined parameters."""
    return llama_generate_response(request, max_new_tokens=80, num_beams=1, temperature=0.1)


def llama32_3b_quiz(request: ChatRequest) -> str:
    """Function specific to the 'quiz' scenario with predefined parameters."""
    return llama_generate_response(request, max_new_tokens=100, num_beams=3, temperature=0.1)
