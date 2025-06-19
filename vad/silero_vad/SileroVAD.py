from vad.vad_interface import VADInterface
from vad.silero_vad.SileroOrt import SileroOrt
from typing import List
import numpy as np
from datetime import datetime
from loguru import logger
import librosa


class SileroVAD(VADInterface):
    def __init__(self, config):
        super(SileroVAD, self).__init__()
        self.initialize(config)

    
    def initialize(self, config):
        self.sensitivity = config['sensitivity']
        self.silence_ms  = config['silence_ms']

        import os
        self.model = SileroOrt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx"))
        self.window_size_samples = 512
        self._samplerate = self.model.sr
        self.audio_with_speech = []
        self.cur_silence_ms = 0

    
    def set_config(self, config):
        self.sensitivity = config['sensitivity']
        self.silence_ms  = config['silence_ms']


    def samplerate(self):
        return self._samplerate
    

    def run(self, audio_data: dict):
        sr = audio_data["samplerate"]
        audio_chunks = audio_data["data"]

        audio = np.concatenate(audio_chunks, axis=-1) if len(audio_chunks) > 1 else audio_chunks[0]
        if sr != self._samplerate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._samplerate)

        if len(self.audio_with_speech) == 0:
            self.start_timestamp = datetime.now().strftime("%Y%m%d %H%M%S.%f")
        
        speech_probs = []
        real_width = len(audio)
        padded_width = int(np.ceil(len(audio) / self.window_size_samples) * self.window_size_samples)
        audio = np.pad(audio, (0, padded_width - len(audio)))
        for i in range(0, len(audio), self.window_size_samples):
            chunk = audio[i: i+self.window_size_samples]
            speech_prob = self.model(chunk).item()
            speech_probs.append(speech_prob)
        self.model.reset_states() # reset model states after each audio

        for i, prob in enumerate(speech_probs):
            chunk = audio[i * self.window_size_samples : (i + 1) * self.window_size_samples]
            if i == len(speech_probs) - 1:
                chunk = chunk[:self.window_size_samples - (padded_width - real_width)]

            if prob > self.sensitivity:
                # logger.debug(f"append")
                self.audio_with_speech.append(chunk)
                self.cur_silence_ms = 0
            else:
                if len(self.audio_with_speech) > 0:
                    # silence
                    self.cur_silence_ms += len(chunk) / self._samplerate * 1000
                    # logger.debug(f"silence: {self.cur_silence_ms} chunk_size: {len(chunk)}")

                    if self.cur_silence_ms > self.silence_ms:
                        self.end_timestamp = datetime.now().strftime("%Y%m%d %H%M%S.%f")
                        # sf.write(f"mic_silero_test/{start_time}-{end_time}.wav", np.concatenate(audio_with_speech, axis=-1), mic.sample_rate)

                        data = np.concatenate(self.audio_with_speech, axis=-1)
                        self.audio_with_speech = []
                        self.cur_silence_ms = 0
                        return {
                            "samplerate": self._samplerate,
                            "data": data,
                            "start_timestamp": self.start_timestamp,
                            "end_timestamp": self.end_timestamp
                        }
                    else:
                        self.audio_with_speech.append(chunk)

        return None                        