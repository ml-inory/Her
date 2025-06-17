from typing import Type
from tts.tts_interface import TTSInterface
from loguru import logger


class TTSFactory:
    @staticmethod
    def create(config: dict) -> Type[TTSInterface]:
        config = config["tts"]
        tts_type = config["type"]

        if tts_type == "edge-tts":
            logger.info("Creating edge-tts...")
            from tts.edge_tts.EdgeTTS import EdgeTTS
            return EdgeTTS(config)
        else:
            raise ValueError(f"Unknown tts type: {tts_type}")
