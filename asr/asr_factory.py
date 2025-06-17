from typing import Type
from asr.asr_interface import ASRInterface
from loguru import logger


class ASRFactory:
    @staticmethod
    def create(config: dict) -> Type[ASRInterface]:
        config = config["asr"]
        asr_type = config["type"]

        if asr_type == "sensevoice":
            logger.info("Creating SenseVoice...")
            from asr.sensevoice.SenseVoice import SenseVoice
            return SenseVoice(config)
        else:
            raise ValueError(f"Unknown asr type: {asr_type}")
