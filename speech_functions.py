# Load model directly
import subprocess
import re
import librosa
import torch
from transformers import AutoProcessor, AutoModelForCTC
from AudioSegmentsToVisemes import AudioSegmentsToVisemes


ffmpeg_path = 'ffmpeg'

processor = AutoProcessor.from_pretrained("Bluecast/wav2vec2-Phoneme")
model = AutoModelForCTC.from_pretrained("Bluecast/wav2vec2-Phoneme")


# Step 1: Detect silences using ffmpeg
def detect_silences(audio_path, noise_level="-30dB", duration=0.2):

    command = f"{ffmpeg_path} -i {audio_path} -af silencedetect=noise={noise_level}:d={duration} -f null - 2>&1"
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode()

    starts = [float(match) for match in re.findall(r'silence_start:\s+([0-9.]+)', output)]
    ends = [float(match) for match in re.findall(r'silence_end:\s+([0-9.]+)', output)]
    durations = [float(match) for match in re.findall(r'silence_duration:\s+([0-9.]+)', output)]

    silences = [{'start': start, 'end': end, 'duration': duration} for start, end, duration in
                zip(starts, ends, durations)]
    return silences


# Step 2: Create audio segments based on silences
def create_audio_segments(audio_path, silences):
    segments = []
    last_end = 0.0

    audio, sample_rate = librosa.load(audio_path, sr=16000)
    for silence in silences:
        if last_end < silence['start']:
            segment = {
                'start': last_end,
                'end': silence['start'],
                'type': 'speech',
                'data': audio[int(last_end * sample_rate):int(silence['start'] * sample_rate)]
            }
            segments.append(segment)

        segments.append({
            'start': silence['start'],
            'end': silence['end'],
            'type': 'silence'
        })

        last_end = silence['end']

    duration = librosa.get_duration(y=audio, sr=sample_rate)
    if last_end < duration:
        segment = {
            'start': last_end,
            'end': duration,
            'type': 'speech',
            'data': audio[int(last_end * sample_rate):int(duration * sample_rate)]
        }
        segments.append(segment)

    return segments


# Step 3: Process each speech segment with the model
def process_segments(segments):
    results = []
    min_length = 160  # Minimum length to ensure the segment is long enough (0.01 seconds at 16000 Hz)
    char_index = 0
    for segment in segments:
        if segment['type'] == 'speech' and len(segment['data']) >= min_length:
            inputs = processor(segment['data'], sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)

            results.append({
                'type': 'speech',
                'start': segment['start'],
                'end': segment['end'],
                'data': transcription[0]
            })
        else:
            # Adding silence segments to results
            results.append({
                'start': segment['start'],
                'end': segment['end'],
                'type': 'silence'
            })

    return results


def phonemize_audio(lang, audio_path):

    silences = detect_silences(audio_path)
    audio_segments = create_audio_segments(audio_path, silences)
    phonemes = process_segments(audio_segments)
    visemes_processor = AudioSegmentsToVisemes(lang)
    visemes = visemes_processor.process_visemes(phonemes)

    return [phonemes, visemes]

