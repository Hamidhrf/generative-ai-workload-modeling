#!/usr/bin/env python3
import whisper
import torch
import numpy as np
import os
from pydub import AudioSegment
import tempfile
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("small", device=device)

print(f"Using device: {device}")

# Configuration
duration_s = 5  # seconds for dummy audio
sample_rate = 16000
num_samples = duration_s * sample_rate

# Generate dummy audio (float32 normalized between -1.0 and +1.0)
audio = np.random.uniform(low=-1.0, high=1.0, size=(num_samples,)).astype(np.float32)

# Convert to 16-bit PCM and prepare raw data for pydub
pcm_int16 = (audio * 32767).astype(np.int16)

# Save as WAV temporarily using numpy + scipy (or directly construct bytes)
# Here we'll go via WAV to then convert to MP3 — could be optimized
from scipy.io.wavfile import write as wav_write
tmp_wav = "/tmp/input.wav"
wav_write(tmp_wav, sample_rate, pcm_int16)

# Use pydub + ffmpeg to convert to MP3
audio_seg = AudioSegment.from_file(tmp_wav, format="wav")
tmp_mp3 = "/tmp/input.mp3"
audio_seg.export(tmp_mp3, format="mp3", bitrate="64k")

print(f"Dummy MP3 saved: {tmp_mp3}")

# Now run inference using the MP3 file (Whisper will decode/handle it internally)
while True:
    result = model.transcribe(tmp_mp3, language="en", fp16=torch.cuda.is_available())
    text = result.get("text", "").strip()
    print(f"Transcription: {text}")
    time.sleep(1)
