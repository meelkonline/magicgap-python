import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
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
    quantization_config=quant_config,
    device_map="auto"
)

# Ensure the model is in evaluation mode
model.eval()


def llama32_3b_ask(request: ChatRequest):
    # Tokenize the prompt
    input_ids = tokenizer(request.messages, return_tensors="pt").input_ids.cuda()

    # Generate the assistant's response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            num_beams=1,
            temperature=0.7,
            top_k=1
        )

    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    split_response = output_text.split('***')
    # Retrieve the part after the '***'
    if len(split_response) > 1:
        desired_output = split_response[1].strip()
    else:
        desired_output = "No valid answer found after '***'"

    return desired_output
