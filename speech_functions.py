# Load model directly
import os
import subprocess
import re
import librosa
import torch
from dotenv import load_dotenv
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import AutoProcessor, AutoModelForCTC, AutoModelForAudioClassification

from AudioSegmentsToVisemes import AudioSegmentsToVisemes
from IPAtoARPAbetConverter import IPAtoARPAbetConverter
from api_requests import MultipleStringRequest
from nlp_functions import evaluate_sentiment

# Path to ffmpeg executable
ffmpeg_path = 'C:\\ffmpeg\\bin\\ffmpeg.exe'  # Adjust if needed

# processor = AutoProcessor.from_pretrained("Bluecast/wav2vec2-Phoneme")
# model = AutoModelForCTC.from_pretrained("Bluecast/wav2vec2-Phoneme")

processor = AutoProcessor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
model = AutoModelForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")


# emotionModel = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes",
#                                                                trust_remote_code=True)
# mean = emotionModel.config.mean
# std = emotionModel.config.std


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

    return [segments, duration]


# Step 3: Process each speech segment with the model
def process_segments(segments, total_chars):
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
            # print(transcription[0])
            start_percentage = char_index / total_chars
            char_index = char_index + count_characters(transcription[0])
            results.append({
                'type': 'speech',
                'start': segment['start'],
                'end': segment['end'],
                'emotions': [],
                'start_percentage': start_percentage,
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


def count_characters(text):
    character_count = len(text.replace(" ", ""))
    return character_count


def split_sentence(sentence):
    words = sentence.split()
    first_part = []
    total_length = len(sentence.replace(' ', ''))  # Total length without spaces
    current_length = 0

    for word in words:
        if current_length + len(word) <= total_length / 2:
            first_part.append(word)
            current_length += len(word)
        else:
            break

    # Join words to form the first part and the remaining words as the second part
    first_part_text = ' '.join(first_part)
    second_part_text = ' '.join(words[len(first_part):])

    return first_part_text, second_part_text


def get_next_speech_segment(segments, index):
    segment_index = -1
    for segment in segments:
        segment_index = segment_index + 1
        if segment_index <= index:
            continue
        if segment['type'] == 'speech':
            return segment

    return []


def apply_emotions(segments, raw_emotions):
    # Update emotion_positions with already calculated start percentages in emotions
    results = []
    used_emotions = []
    segment_max_length = 20
    char_index = 0
    segment_index = -1
    for segment in segments:
        segment_index = segment_index + 1
        if segment['type'] == 'speech':
            segment_duration = segment['end'] - segment['start']
            segment_start_time = segment['start']
            segment_characters = count_characters(segment['data'])

            split_time_duration = segment_duration

            # Split segments with more than twelve characters
            if segment_characters > segment_max_length:
                next_speech_segment = get_next_speech_segment(segments, segment_index)
                if len(next_speech_segment) > 0:
                    next_percentage = next_speech_segment['start_percentage']
                else:
                    next_percentage = 1

                split_point = segment_characters // 2
                # Calculate the time duration of the first split segment
                split_time_duration = (split_point / segment_characters) * segment_duration

            # Define sub-segments for emotion matching if split
            sub_segments = [
                {'start_time': segment_start_time,
                 'start_percentage': segment['start_percentage']},
                {'start_time': segment_start_time + split_time_duration,
                 'start_percentage': segment['start_percentage'] + (
                         (next_percentage - segment['start_percentage']) / 2)}
            ] if segment_characters > segment_max_length else [
                {'start_time': segment_start_time, 'start_percentage': segment['start_percentage']}]

            char_index = char_index + segment_characters

            for sub_segment in sub_segments:

                # Find the closest emotion based on character distribution and start percentages
                closest_emotion = min(raw_emotions,
                                      key=lambda e: abs(e['start_percentage'] / 100 - sub_segment['start_percentage']))

                text = closest_emotion['text']
                result_index = len(results)
                results.append({
                    'index': result_index,
                    'emotion': closest_emotion['sentiment'],
                    'text': text,
                    'time': sub_segment['start_time'],  # Use the calculated start time of the segment or sub-segment
                    'start_percentage': sub_segment['start_percentage']
                })

                if all(emotion['id'] != closest_emotion['id'] for emotion in used_emotions):
                    used_emotions.append({
                        'id': closest_emotion['id'],
                        'text': closest_emotion['text'],
                        'result_index': result_index
                    })

    # Dictionary to hold updates before applying
    updates = {emotion['id']: emotion['text'] for emotion in used_emotions}

    used_ids = set(emotion['id'] for emotion in used_emotions)

    for raw_emotion in raw_emotions:
        if raw_emotion['id'] not in used_ids:
            for i in range(len(used_emotions) - 1, -1, -1):
                if used_emotions[i]['id'] < raw_emotion['id']:
                    # Concatenate raw_emotion text to the appropriate used_emotion
                    updates[used_emotions[i]['id']] = used_emotions[i]['text'] + " " + raw_emotion['text']
                    used_ids.add(raw_emotion['id'])  # Mark this raw_emotion as used
                    break

        # Apply updates from the updates dictionary
    for used_emotion in used_emotions:
        text = updates[used_emotion['id']]
        for result in results:
            if result['index'] == used_emotion['result_index']:
                result['text'] = text
                break

    return results


def phonemize_audio(audio_path, text):
    load_dotenv()  # Load environment variables from a .env file
    app_env = os.getenv('APP_ENV')
    if app_env == "local":
        EspeakWrapper.set_library('C:\\Program Files\\eSpeak NG\\libespeak-ng.dll')

    pattern = r'(\.|\.\.\.|!|\?|\n)'

    # Get Filtered Sentences
    total_chars = 0
    cumulative_chars = 0  # To store cumulative character counts
    sentences = []
    arpabets = []
    positions = []  # Store starting positions for each sentence
    sentences_with_points = re.split(pattern, text)
    for sentence_or_point in sentences_with_points:
        if len(sentence_or_point.strip()) > 1:
            sentence = sentence_or_point.strip()
            ipa_string = phonemize(sentence, preserve_punctuation=False)
            sentences.append(sentence)
            num_char = count_characters(ipa_string)
            positions.append(cumulative_chars)  # Save the start position of the current sentence
            cumulative_chars += num_char  # Update cumulative characters
            total_chars += num_char

    # Convert start positions to percentages
    position_percentages = [pos / total_chars * 100 for pos in positions]

    # Get  Sentences Sentiment (except neutral)
    sentiments = []
    multi_sentiments = evaluate_sentiment(MultipleStringRequest(strings=sentences))
    for multi_sentiment in multi_sentiments:
        if multi_sentiment[0]['label'] == "neutral":
            sentiments.append(multi_sentiment[1])
        else:
            sentiments.append(multi_sentiment[0])

    # Combine sentences with their positions and sentiments
    raw_emotions = []
    texts = []
    for i, sentence in enumerate(sentences):
        raw_emotions.append({
            'id': i,
            'text': sentence,
            # 'arpabet': arpabets[i],
            'start_percentage': position_percentages[i],
            'sentiment': sentiments[i]
        })

    silences = detect_silences(audio_path)
    audio_segments = create_audio_segments(audio_path, silences)
    total_audio_duration = audio_segments[1]
    phonemes = process_segments(audio_segments[0], total_chars)
    emotions = apply_emotions(phonemes, raw_emotions)

    # Assuming 'segments' is a list of dicts with keys 'type', 'start', 'end', 'data'
    visemes_processor = AudioSegmentsToVisemes()
    visemes = visemes_processor.process_visemes(phonemes)
    # print(visemes)
    return [phonemes, emotions, visemes, texts]
