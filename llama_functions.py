import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, StoppingCriteriaList, \
    StoppingCriteria
from huggingface_hub import login
from api_requests import ChatRequest
from nlp_functions import limit_to_sentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login("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")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=quant_config
)

# Ensure the model is in evaluation mode
model.eval()


class StopAtFirstValidJSON(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the latest tokens into text
        self.buffer += self.tokenizer.decode(input_ids[0][-1:], skip_special_tokens=True)

        # Attempt to parse JSON from the buffer
        if self.buffer.strip().endswith("}"):
            try:
                json.loads(self.buffer.strip())  # Check if it's a valid JSON
                return True  # Stop if JSON is valid and closed
            except json.JSONDecodeError:
                pass

        return False  # Continue generating otherwise


class MaxWordStoppingCriteria(StoppingCriteria):
    def __init__(self, max_words):
        self.max_words = max_words
        self.current_words = 0

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.current_words = len(decoded_text.split())
        if self.current_words >= self.max_words:
            # Check if the last token is a punctuation mark indicating end of sentence
            if decoded_text.strip()[-1] in {'.', '!', '?'}:
                return True
        return False


def llama32_3b_ask(request: ChatRequest):
    # Tokenize the prompt
    tokenizer_input = tokenizer(request.messages, return_tensors="pt")
    input_ids = tokenizer_input.input_ids.cuda()
    attention_mask = tokenizer_input.attention_mask.cuda()

    # tokenizer.add_special_tokens({'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>']})
    # stopping_criteria = StoppingCriteriaList([MaxWordStoppingCriteria(max_words=100)])

    # Generate the assistant's response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=80,  # Approx. 50 words
            num_beams=3,
            temperature=0.1,
            top_k=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id  # Prevents padding from being added
        )

        # Extract only the generated part (new tokens)
    prompt_length = input_ids.size(1)  # Handle batch size
    generated_tokens = outputs[:, prompt_length:]  # Slice for all batches

    # Decode only the new part
    new_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    new_text = [text.strip('"') for text in new_text]  # Clean up quotes

    # Limit to 2 sentences
    return limit_to_sentences(new_text[0], 2)


def llama32_3b_quiz(request: ChatRequest):
    # Tokenize the prompt
    tokenizer_input = tokenizer(request.messages, return_tensors="pt")
    input_ids = tokenizer_input.input_ids.cuda()
    attention_mask = tokenizer_input.attention_mask.cuda()

    stopping_criteria = StopAtFirstValidJSON(tokenizer)

    # Generate the assistant's response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,  # Limit to shorter responses
            num_beams=3,
            temperature=0.6,
            top_k=5,
            stopping_criteria=StoppingCriteriaList([stopping_criteria]),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id  # Prevents padding from being added
        )

    # Extract only the generated part (new tokens)
    prompt_length = input_ids.shape[1]  # Length of the original prompt
    generated_tokens = outputs[0][prompt_length:]  # Only the new tokens generated

    # Decode only the new part
    new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    new_text = new_text.strip('"')

    # print(f"Generated response: {new_text}")
    return new_text

