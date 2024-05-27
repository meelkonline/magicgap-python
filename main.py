from speech_functions import phonemize_audio
string = "Hello, I'm Sue... once a lady in Paris, married to a RICH MAN. The war made me flee to this island... Now, I'm a prostitute... these sailors, they re so RUDE! I MISS the love... and the elegance of Paris... every single day."

segments = phonemize_audio('ac75-en.wav', string)

print(segments)
