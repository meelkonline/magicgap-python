import requests
from phonemizer import phonemize
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-lv-60-espeak-cv-ft"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("AC103.wav")
print(output)
