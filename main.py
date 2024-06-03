import librosa
import torch

from speech_functions import phonemize_audio

audio_path = 'ac75-en.wav'
string = "Hello, I'm Sue... once a lady in Paris, married to a RICH MAN. The war made me flee to this island... Now, I'm a prostitute... these sailors, they re so RUDE! I MISS the love... and the elegance of Paris... every single day."
# string = "Les voyelles en Français sont A...E...I....O...U"
# audio_path = 'ac63-fr.wav'

segments = phonemize_audio(audio_path, string)
print(segments[2])
#
# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC
#
# audio, sample_rate = librosa.load(audio_path, sr=16000)
# processor = AutoProcessor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
# model = AutoModelForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
#
# # processor = AutoProcessor.from_pretrained("Cnam-LMSSC/wav2vec2-french-phonemizer")
# # model = AutoModelForCTC.from_pretrained("Cnam-LMSSC/wav2vec2-french-phonemizer")
#
# inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
# with torch.no_grad():
#     logits = model(inputs.input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
# print(transcription)
#
#
# processor = AutoProcessor.from_pretrained("dg96/whisper-finetuning-phoneme-transcription-g2p-large-dataset-space-seperated-phonemes")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("dg96/whisper-finetuning-phoneme-transcription-g2p-large-dataset-space-seperated-phonemes")
# inputs = processor(audio, return_tensors="pt")
# input_features = inputs.input_features
# generated_ids = model.generate(inputs=input_features)
# transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(transcription)