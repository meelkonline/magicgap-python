import base64
import json

import librosa
import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import re
import soundfile as sf
from kokoro import KPipeline
import numpy as np
from api_requests import SentimentRequest
from sentiment_analysis import classify_emotion, evaluate_sentiment

# ---------------------------
# Load TTS components once
# ---------------------------
# audio_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# audio_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

kokoro_pipeline = KPipeline(lang_code='a')
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


# OLD: SpeechT5 Inference
# def synthesize_sentence(sentence: str):
#     with torch.no_grad():
#         inputs = audio_processor(text=sentence, return_tensors="pt")
#         audio_tensor = audio_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
#     return audio_tensor.numpy()  # float32 NumPy array


def stream_output(data):
    """Helper function to handle streaming JSON output."""
    yield json.dumps(data) + "\n"


def text_to_audio(text: str, lang: str = "en", voice_id: str = "", stream: bool = True):
    """
    Generate speech using Kokoro, either as a streaming response or as a full JSON array.

    Parameters:
    - `text` (str): The input text to synthesize.
    - `lang` (str): The language for sentiment analysis.
    - `stream` (bool): If True, yields a streaming JSON response; otherwise, returns a full JSON list.
    - `split_pattern` (str): Defines how Kokoro splits text (default: `\n+`).

    Returns:
    - If `stream=True`: Generator yielding JSON per sentence (text, emotions, audio).
    - If `stream=False`: A JSON list of sentences with text, emotions, and Base64 audio.
    """
    TARGET_SAMPLE_RATE = 22050  # Ensure time calculations are accurate
    split_pattern = r"(?<=[.!?…])\s+"
    generator = kokoro_pipeline(text, voice=voice_id, speed=1.0, split_pattern=split_pattern)
    audio_chunks = []
    response_list = []  # List to store JSON responses (for non-stream mode)
    elapsed_time = 0.0  # Track cumulative time of sentences

    for i, (gs, ps, audio) in enumerate(generator):
        print(f"Chunk {i}: {gs}")  # ✅ Get exact text Kokoro generated
        print(f"Phonemes: {ps}")

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

        if stream:
            stream_output(data_line)
        else:
            response_list.append(data_line)  # Store for full response

    # If `stream=False`, return the full JSON array
    if not stream:
        full_audio = np.concatenate(audio_chunks, axis=0)

        # 🔹 Convert to Base64
        full_audio_base64 = base64.b64encode(full_audio.tobytes()).decode("utf-8")
        # 🔹 Convert list to dictionary before adding `audio_base64`
        response_data = {"audio_base64": full_audio_base64, "sentences": response_list}

        return response_data

