import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Choose the desired Whisper ASR model for your target languages
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Example input audio in the form of a file path
audio_file_path = "sample.wav"

# Load and process the audio file
input_audio, _ = torchaudio.load(audio_file_path)

# Tokenize the audio waveform
inputs = processor(input_audio.squeeze().numpy(), return_tensors="pt", padding="longest")

# Perform speech recognition
with torch.no_grad():
    logits = model(input_values=inputs.input_values).logits

# Convert the model output to text using the processor
transcription = processor.batch_decode(logits.argmax(dim=-1))

print("Transcription:", transcription)
