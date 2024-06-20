from transformers import pipeline

text = "Mais parfois, mon humeur change et je me sens très triste."
pipe = pipeline("text-classification", model="ac0hik/emotions_detection_french")
r = pipe(text)
print(r)
