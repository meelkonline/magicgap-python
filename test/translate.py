
from transformers import pipeline
sentences = ["How are you?", "The house is wonderful.", "I like to work in NYC."]
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
output = pipe(sentences)
print(output)