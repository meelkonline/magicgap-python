import base64
import json

import torch
from datasets import load_dataset
import re
#from kokoro import KPipeline
import numpy as np
from api_requests import SentimentRequest
from sentiment_analysis import classify_emotion, evaluate_sentiment

# ---------------------------
# Load TTS components once
# ---------------------------
# audio_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# audio_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


# ---------------------------
# Lazy Load Pipeline
# ---------------------------
_kokoro_instance = None


def _get_kokoro_pipeline():
    """
    Lazy-loads the Kokoro pipeline on demand.
    Ensures the model is only loaded when needed.
    """
    global _kokoro_instance
    if _kokoro_instance is None:
        _kokoro_instance = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return _kokoro_instance


# Load speaker embeddings once
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def split_into_sentences(text: str):
    """
    Naive approach: split by punctuation + re-attach it.
    """
    parts = re.split(r"([.?!…])", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i + 1]).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


# OLD: SpeechT5 Inference
# def synthesize_sentence(sentence: str):
#     with torch.no_grad():
#         inputs = audio_processor(text=sentence, return_tensors="pt")
#         audio_tensor = audio_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
#     return audio_tensor.numpy()  # float32 NumPy array


def stream_output(data):
    """Helper function to handle streaming JSON output."""
    yield json.dumps(data) + "\n"


def kokoro_save_audio(text: str, lang: str = "en", voice_id: str = ""):
    TARGET_SAMPLE_RATE = 22050  # Ensure time calculations are accurate
    split_pattern = r"(?<=[.!?…])\s+"

    # 🔹 Load Kokoro Model on Demand
    kokoro_pipeline = _get_kokoro_pipeline()

    generator = kokoro_pipeline(text, voice=voice_id, speed=1.0, split_pattern=split_pattern)
    audio_chunks = []
    response_list = []  # List to store JSON responses (for non-stream mode)
    elapsed_time = 0.0  # Track cumulative time of sentences

    for i, (gs, ps, audio) in enumerate(generator):

        # 🔹 Emotion Detection
        emotion_dict = evaluate_sentiment(SentimentRequest(lang=lang, strings=[gs]))

        # 🔹 Ensure audio is a NumPy array
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        # 🔹 Convert float32 → int16 PCM (16-bit for compatibility)
        audio = (audio * 32767).astype(np.int16)
        audio_chunks.append(audio)

        chunk_duration = len(audio) / TARGET_SAMPLE_RATE  # Time in seconds

        # 🔹 Convert to Base64
        audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")

        # 🔹 Build JSON response
        data_line = {
            "text": gs,  # ✅ Sentence text
            "emotions": emotion_dict,  # ✅ Emotion classification
            "audio_base64": audio_base64,  # ✅ Base64 encoded audio
            "time": round(elapsed_time, 3)  # ✅ Start time relative to full audio
        }

        elapsed_time += chunk_duration
        response_list.append(data_line)  # Store for full response

    full_audio = np.concatenate(audio_chunks, axis=0)

    # 🔹 Convert to Base64
    full_audio_base64 = base64.b64encode(full_audio.tobytes()).decode("utf-8")
    # 🔹 Convert list to dictionary before adding `audio_base64`
    response_data = {"audio_base64": full_audio_base64, "sentences": response_list}

    return response_data


def kokoro_stream_audio(text: str, lang: str = "en", voice_id: str = ""):
    TARGET_SAMPLE_RATE = 22050  # Ensure time calculations are accurate
    split_pattern = r"(?<=[.!?…])\s+"
    # 🔹 Load Kokoro Model on Demand
    kokoro_pipeline = _get_kokoro_pipeline()

    generator = kokoro_pipeline(text, voice=voice_id, speed=1.0, split_pattern=split_pattern)
    audio_chunks = []
    response_list = []  # List to store JSON responses (for non-stream mode)
    elapsed_time = 0.0  # Track cumulative time of sentences

    for i, (gs, ps, audio) in enumerate(generator):

        # 🔹 Emotion Detection
        emotion_dict = evaluate_sentiment(SentimentRequest(lang=lang, strings=[gs]))

        # 🔹 Ensure audio is a NumPy array
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        # 🔹 Convert float32 → int16 PCM (16-bit for compatibility)
        audio = (audio * 32767).astype(np.int16)
        audio_chunks.append(audio)

        chunk_duration = len(audio) / TARGET_SAMPLE_RATE  # Time in seconds

        # 🔹 Convert to Base64
        audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")

        # 🔹 Build JSON response
        data_line = {
            "text": gs,  # ✅ Sentence text
            "emotions": emotion_dict,  # ✅ Emotion classification
            "audio_base64": audio_base64,  # ✅ Base64 encoded audio
            "time": round(elapsed_time, 3)  # ✅ Start time relative to full audio
        }

        print(f"data_line {i}: {data_line}")  # ✅ Get exact text Kokoro generated

        elapsed_time += chunk_duration

        yield json.dumps(data_line) + "\n"
