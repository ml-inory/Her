from tts.tts_interface import TTSInterface
import os
import re
import numpy as np
import librosa


def remove_brackets_and_content(text):
    return re.sub(r'\([^)]*\)', '', text)


class EdgeTTS(TTSInterface):
    def __init__(self, config):
        self.initialize(config)


    def initialize(self, config: dict):
        """
        Initialize TTS engine.Do things like setting vad config, loading models, etc.

        config: yaml config 
        """
        self.voice = config["voice"]


    def set_config(self, config: dict):
        """
        Set config of TTS

        config: yaml config
        """
        self.voice = config["voice"]


    def run(self, text: str) -> np.ndarray:
        """
        Run TTS

        return: {
            audio[np.ndarray]
        }
        """
        text = remove_brackets_and_content(text)
        os.system(f'edge-tts --text "{text}" -v {self.voice} --write-media hello.mp3 --rate=-20%')
        audio, _ = librosa.load("hello.mp3", sr=16000)
        return {
            "audio": audio,
            "samplerate": 16000
        }