import requests

# Define the API endpoint
endpoint = "https://api-inference.huggingface.co/models/huggyllama/llama-30b"

# Define your headers - Replace 'YOUR_API_TOKEN' with your Hugging Face API token
headers = {
    "Authorization": "Bearer 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"
}

# Define the data you want to send
data = {
    "inputs": "Your input text here."
}

# Make the API call
response = requests.post(endpoint, headers=headers, json=data)

# Extract the results from the API response
result = response.json()

print(result)