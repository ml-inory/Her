from openai import OpenAI
from loguru import logger
import os


class LLMInterface(object):
    def __init__(self, config):
        self.initialize(config)


    def initialize(self, config: dict):
        """
        Initialize LLM engine.Do things like setting vad config, loading models, etc.

        config: yaml config 
        """
        llm_type = config["type"]
        if llm_type == "deepseek":
            base_url = "https://api.deepseek.com"
            model = "deepseek-reasoner"
        else:
            raise ValueError(f"Unknown llm type {llm_type}")
        
        self.base_url = base_url
        self.model = model

        api_key = config.get("api_key")
        if api_key is None:
            api_key = os.environ.get("API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = config["system_prompt"]
        self.chat_history = [
            {'role': 'assistant', 'content': self.system_prompt}
        ]

        logger.debug(self.system_prompt)


    def run(self, message: str) -> str:
        """
        Run LLM

        """
        self.chat_history.append(
            {'role': 'user', 'content': message}
        )

        # 发送请求给 OpenAI GPT
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            max_tokens=1000 # 控制生成回复的最大长度
        )

        # 提取回复
        content = response.choices[0].message.content

        self.chat_history.append(
            {'role': 'assistant', 'content': content}
        )
        return content