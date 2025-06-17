import yaml
import librosa
import os
import soundfile as sf
import numpy as np
from loguru import logger
import shutil

from vad.vad_factory import VADFactory
from asr.asr_factory import ASRFactory


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

vad_engine = VADFactory.create(config)
asr_engine = ASRFactory.create(config)

audio, _ = librosa.load("test_data/en.wav", sr=vad_engine.samplerate(), mono=True)
index = 0
window_size = 512

pad_blank_length = int(config["vad"]["silence_ms"] * 2 / 1000 * vad_engine.samplerate())
audio = np.concatenate([audio, np.zeros((pad_blank_length,), dtype=np.float32)], axis=-1)

while True:
    audio_chunk = audio[index * window_size : (index + 1) * window_size]
    if len(audio_chunk) == 0:
        break

    vad_result = vad_engine.run([audio_chunk])
    if vad_result is not None:
        start_ts = vad_result["start_timestamp"]
        end_ts   = vad_result["end_timestamp"]
        data     = vad_result["data"]

        if vad_result["samplerate"] != asr_engine.samplerate():
            data = librosa.resample(data, orig_sr=vad_result["samplerate"], target_sr=asr_engine.samplerate())

        asr_result = asr_engine.run({
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "data": data
        })
        text = asr_result["text"]
        
        logger.info(f"{start_ts} - {end_ts}: {text}")

    index += 1
