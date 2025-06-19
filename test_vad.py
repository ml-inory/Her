from vad.vad_factory import VADFactory
import yaml
import librosa
import os
import soundfile as sf
import numpy as np
from loguru import logger
import shutil


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

vad_engine = VADFactory.create(config)

audio, sr = librosa.load("test_data/en.wav", sr=vad_engine.samplerate(), mono=True)
index = 0
window_size = 512
result_path = "vad_test_result"
if os.path.exists(result_path):
    shutil.rmtree(result_path)
os.makedirs(result_path, exist_ok=True)

pad_blank_length = int(config["vad"]["silence_ms"] * 2 / 1000 * vad_engine.samplerate())
audio = np.concatenate([audio, np.zeros((pad_blank_length,), dtype=np.float32)], axis=-1)

while True:
    audio_chunk = audio[index * window_size : (index + 1) * window_size]
    if len(audio_chunk) == 0:
        break

    result = vad_engine.run({
        "samplerate": sr,
        "data": [audio_chunk]
    })
    if result is not None:
        start_ts = result["start_timestamp"]
        end_ts   = result["end_timestamp"]
        out_wav_path = f"{result_path}/{start_ts}-{end_ts}.wav"
        logger.info(f"Save audio with speech to {out_wav_path}")
        sf.write(f"{out_wav_path}", result["data"], result["samplerate"])

    index += 1
