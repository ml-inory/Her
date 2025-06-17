import yaml
from loguru import logger

from llm.llm_factory import LLMFactory
from tts.tts_factory import TTSFactory


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

llm = LLMFactory.create(config)
tts = TTSFactory.create(config)

while True:
    message = input("用户: ")
    reply = llm.run(message)
    tts.run(reply)
    print("女友: ", reply)