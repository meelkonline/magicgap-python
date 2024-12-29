import base64
import json
import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import re
import soundfile as sf

from sentiment_analysis import classify_emotion

# ---------------------------
# Load TTS components once
# ---------------------------
audio_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
audio_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def split_into_sentences(text: str):
    """
    Naive approach: split by punctuation + re-attach it.
    This can be replaced with your own chunking logic.
    """
    # e.g. "Hello. My dog is cute. Lorem ipsum?"
    # -> ["Hello.", "My dog is cute.", "Lorem ipsum?"]
    # This is just an example; adapt as needed.
    # We'll keep .?! as delimiters.
    parts = re.split(r"([.?!])", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i + 1]).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def synthesize_sentence(sentence: str):
    """
    Run TTS on a single chunk/sentence and return raw float32 array.
    """
    with torch.no_grad():
        inputs = audio_processor(text=sentence, return_tensors="pt")
        audio_tensor = audio_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    return audio_tensor.numpy()  # float32 NumPy array


def text_to_audio_stream(text: str, lang: str = "en"):
    """
    Generator yielding line-delimited JSON, each line containing:
      - text (the chunk)
      - emotions (label + score)
      - audio_base64
    """
    # 1) Split text into sentences
    sentences = split_into_sentences(text)

    # 2) For each chunk: classify, synthesize, and yield
    for sentence in sentences:
        # (a) Emotion
        emotion_dict = classify_emotion(sentence, lang)

        # (b) TTS
        audio_array = synthesize_sentence(sentence)
        audio_bytes = audio_array.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # (c) Build JSON line
        data_line = {
            "text": sentence,
            "emotions": emotion_dict,
            "audio_base64": audio_base64,
        }
        yield json.dumps(data_line) + "\n"


def text_to_audio_file(text: str, lang: str = "en"):
    inputs = audio_processor(text=text, return_tensors="pt")
    speech = audio_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    filename = "speech8.wav"
    sf.write(filename, speech.numpy(), samplerate=16000)
    audio_bytes = speech.numpy().tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_base64}
