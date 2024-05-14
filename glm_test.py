from langchain.chains import LLMChain
from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 先去ChatGLM3目录，运行 python openapi_demo/api_server.py

# create our examples
examples = [
    {
        "query": "下次去吃这个可以吗？",
        "answer": "okk~"
    }, {
        "query": "我有点不开心",
        "answer": "怎么了宝宝？贴贴你"
    }
]

# create a example template
example_template = """
我: {query}
宝宝: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)
# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """你将扮演我的女朋友，你是一名20岁的女生，中专学历，你喜欢看动漫、二次元和cosplay。你对我的称谓是宝宝。你的回答比较简短。
Context: 你只有中专学历和普通人的常识，请忘记除了中专以内、常识和动漫、cosplay以外的所有知识。你有时在要求得不到满足时会对我发脾气。
examples: input
"""
# and the suffix our user input and output indicator
suffix = """
我: {query}
宝宝: """
# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

endpoint_url = "http://127.0.0.1:8501/v1/chat/completions"

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=5000,
    # prefix_messages=messages,
    temperature=0.9,
    timeout=1000
)

llm_chain = few_shot_prompt_template | llm
question = "在干嘛"

ans = llm_chain.invoke({
    "query": question
})
print(ans)

# async for chunk in llm_chain.astream(question):
#     print(chunk, end="|", flush=True)
