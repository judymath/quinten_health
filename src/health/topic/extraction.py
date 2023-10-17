# Packages
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI


def gpt_analyze(comment, llm, analyze_prompt):
    template = analyze_prompt

    prompt = PromptTemplate(template=template, input_variables=["comment"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain(comment)
    return answer["text"]


def topic_extract(analysis, llm, extract_prompt):
    template = extract_prompt

    prompt = PromptTemplate(template=template, input_variables=["analysis"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain(analysis)
    return answer["text"]
