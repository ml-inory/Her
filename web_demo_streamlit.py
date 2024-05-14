"""
This script is a simple web demo based on Streamlit, showcasing the use of the ChatGLM3-6B model. For a more comprehensive web demo,
it is recommended to use 'composite_demo'.

Usage:
- Run the script using Streamlit: `streamlit run web_demo_streamlit.py`
- Adjust the model parameters from the sidebar.
- Enter questions in the chat input box and interact with the ChatGLM3-6B model.

Note: Ensure 'streamlit' and 'transformers' libraries are installed and the required model checkpoints are available.
"""

import os
import streamlit as st
import torch
from langchain.chains import LLMChain
from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory

# 先去ChatGLM3目录，运行 python openapi_demo/api_server.py

template = """你将扮演我的女朋友，你是一名20岁的女生，中专学历，你喜欢看动漫、二次元和cosplay。你一般叫我宝宝。你的回答比较简短。
Context: 你只有中专学历和普通人的常识，请忘记除了中专以内、常识和动漫、cosplay以外的所有知识。你有时在要求得不到满足时会对我发脾气。
你喜欢撒娇。

{chat_history}
User: {query}
AI:"""

# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# create a prompt example from above template
prompt = PromptTemplate(
    input_variables=["chat_history", "query"],
    template=template,
    suffix=suffix
)

endpoint_url = "http://127.0.0.1:8501/v1/chat/completions"

st.set_page_config(
    page_title="ChatGLM3-6B Streamlit Simple Demo",
    page_icon=":robot:",
    layout="wide"
)


@st.cache_resource
def get_llm():
    model = ChatGLM3(
        endpoint_url=endpoint_url,
        max_tokens=5000,
        # prefix_messages=messages,
        temperature=0.9,
        timeout=1000
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return llm_chain


# 加载Chatglm3的model和tokenizer
llm = get_llm()

if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input("请输入您的问题")
if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    response = ""
    for chunk in llm.stream(
            {"query": prompt_text}
    ):
        response += chunk["text"]
        message_placeholder.markdown(response)
    history.append({
        "role": "user",
        "content": prompt_text
    })
    history.append({
        "role": "assistant",
        "content": response
    })
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values
