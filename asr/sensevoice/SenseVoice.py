from asr.asr_interface import ASRInterface
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
from datetime import datetime


class SenseVoice(ASRInterface):
    def __init__(self, config: dict):
        super(SenseVoice, self).__init__()

        self.initialize(config)

    
    def initialize(self, config: dict):
        self.language = config.get("language", "auto")
        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_dir = "iic/SenseVoiceSmall"
        self.model = AutoModel(model=model_dir, trust_remote_code=True, remote_code="./model.py", disable_update=True, device=self.device)


    def samplerate(self):
        return 16000
    

    def set_config(self, config: dict):
        self.language = config["language"]
    

    def run(self, audio_data):
        res = self.model.generate(
            input=audio_data["data"],
            cache={},
            language=self.language, # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size=1, 
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return {
            "start_timestamp": audio_data.get("start_timestamp", ""),
            "end_timestamp": audio_data.get("end_timestamp", ""),
            "text": text
        }
