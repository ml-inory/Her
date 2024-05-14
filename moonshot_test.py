from langchain_community.llms.moonshot import Moonshot

import os

# Generate your api key from: https://platform.moonshot.cn/console/api-keys
os.environ["MOONSHOT_API_KEY"] = "sk-gU4bx4vxYWpyZssilmsbUopjSv22rvFwgGfEYtNVaviMWga3"

llm = Moonshot(model="moonshot-v1-8k")
# or use a specific model
# Available models: https://platform.moonshot.cn/docs
# llm = Moonshot(model="moonshot-v1-128k")

# Prompt the model
print(llm.invoke("What is the difference between panda and bear?"))