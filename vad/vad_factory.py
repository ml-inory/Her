from typing import Type
from vad.vad_interface import VADInterface
from loguru import logger


class VADFactory:
    @staticmethod
    def create(config: dict) -> Type[VADInterface]:
        config = config["vad"]
        vad_type = config["type"]

        if vad_type == "silero-vad":
            logger.info("Creating silero-vad...")
            from vad.silero_vad.SileroVAD import SileroVAD
            return SileroVAD(config)
        else:
            raise ValueError(f"Unknown vad type: {vad_type}")
