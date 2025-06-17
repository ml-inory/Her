import yaml
from loguru import logger

from llm.llm_factory import LLMFactory


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

llm = LLMFactory.create(config)
while True:
    message = input("用户: ")
    print("女友: ", llm.run(message))