from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # Use "cpu" if you don't have a CUDA-compatible GPU
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Step 3: Running the Model
prompt = "My favourite condiment is"  # Example prompt
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
result = tokenizer.batch_decode(generated_ids)[0]

print(result)
