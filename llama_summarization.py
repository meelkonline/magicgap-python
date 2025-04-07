import json
from typing import List

import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
from huggingface_hub import login
from api_requests import ChatRequest, SummarizeRequest
from llama_functions import format_response
from nlp_functions import limit_to_sentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Delayed Login (not done on startup)
def lazy_login():
    login("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")


# Lazy-loaded model and tokenizer
llama_tokenizer = None
llama_model = None

model_name = "meta-llama/Llama-3.2-3B-Instruct"
nlp = spacy.load("en_core_web_sm")


def clean_first_sentence(sentences):
    """Removes any leading explanation before the first meaningful sentence."""
    if sentences and "\n\n" in sentences[0]:
        sentences[0] = sentences[0].split("\n\n", 1)[1]  # Keep only text after first \n\n
    return sentences


def get_llama_model():
    global llama_tokenizer, llama_model, model_name
    if llama_model is None or llama_tokenizer is None:
        print(f"🔹 Loading {model_name} into memory...")
        lazy_login()  # Ensure Hugging Face authentication

        llama_tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

        llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config
        )

        llama_model.eval()  # Ensure model is in evaluation mode
        print("✅ Model Loaded Successfully!")

    return llama_tokenizer, llama_model


def format_conversation(messages: List[str]) -> str:
    """Formats the conversation into a prompt string with preserved line breaks."""
    num_sentences = max(1, len(messages) // 2)  # Ensure at least 1 summary sentence
    conversation_text = "\n".join(messages)  # Keep line breaks between messages
    return f"<|user|>\nRésume cette conversation en {num_sentences} phrases :\n{conversation_text}\n<|assistant|>"


def llama_generate_summarize(request: SummarizeRequest, max_new_tokens=100, num_beams=1, temperature=0.1) -> list[str]:
    """
    Generate a response from the model with customizable parameters.
    """
    tokenizer, model = get_llama_model()  # Ensure the model is loaded

    conversation = format_conversation(request.messages)

    tokenizer_input = tokenizer(conversation, return_tensors="pt")
    input_ids = tokenizer_input.input_ids.cuda()
    attention_mask = tokenizer_input.attention_mask.cuda()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = input_ids.size(1)
    generated_tokens = outputs[:, prompt_length:]

    new_text = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True).strip()
    response = format_response(new_text)

    doc = nlp(response)
    summary_sentences = [sent.text.strip() for sent in doc.sents]
    return clean_first_sentence(summary_sentences)


def llama32_summarize(request: SummarizeRequest) -> list[str]:
    """Function specific to the 'ask' scenario with predefined parameters."""
    return llama_generate_summarize(request, max_new_tokens=request.max_length, num_beams=3, temperature=0.1)

# conversation = """<|user|>Résume cette conversation en 4 phrases :
# - Qui est Bruce?
# - Bruce est un homme brisé, rongé par ses vices.
# - Était-il dans l'armée italienne?
# - Oui, il y a quelques années.
# - Que fait-il maintenant?
# - Il est devenu pêcheur et navigue dans l'océan Indien.
# - Comment s'appelle son bateau?
# - Son bateau s'appelle le Bamboo."""
#
#
#
# r = ChatRequest(messages=conversation)
# s = llama32_summarize(r)
# print(s)
