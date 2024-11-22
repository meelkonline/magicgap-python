import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, StoppingCriteriaList, \
    StoppingCriteria
from huggingface_hub import login
from api_requests import ChatRequest

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
    input_ids = tokenizer(request.messages, return_tensors="pt").input_ids.cuda()

    # tokenizer.add_special_tokens({'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>']})
    stopping_criteria = StoppingCriteriaList([MaxWordStoppingCriteria(max_words=35)])

    # Generate the assistant's response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,  # Limit to shorter responses
            num_beams=1,
            temperature=0.3,
            top_k=1,
            stopping_criteria=stopping_criteria,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id  # Prevents padding from being added
        )

    # Extract only the generated part (new tokens)
    prompt_length = input_ids.shape[1]  # Length of the original prompt
    generated_tokens = outputs[0][prompt_length:]  # Only the new tokens generated

    # Decode only the new part
    new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # print(f"Generated response: {new_text}")
    return new_text
