from typing import Type
from llm.llm_interface import LLMInterface
from loguru import logger


class LLMFactory:
    @staticmethod
    def create(config: dict) -> Type[LLMInterface]:
        config = config["llm"]
        logger.info("Create LLM...")

        return LLMInterface(config)
