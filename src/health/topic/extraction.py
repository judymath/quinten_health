# Packages
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI API
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = os.getenv("openai_base")
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key")


def gpt_analyze(comment):
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", temperature=0.1
    )

    template = """
    You will be given a patient's comment about their drug treatment.
    In the comment, please identify the main categories of side effects listed by the patient, and return them as a Python list (ex: ['chest pain', 'rash']).
    Act step by step. Remember to return a python list.
    If there are no side effects listed in the comment, simply return an empty Python list ([]).
    Act step by step and return a Python list, not a text list.

    COMMENT:

    {comment}

    MAIN SIDE EFFECTS:
    """

    prompt = PromptTemplate(template=template, input_variables=["comment"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain(comment)
    return answer["text"]


def topic_extract(analysis):
    llm = AzureOpenAI(
        deployment_name="text-davinci-003",
        model_name="text-davinci-003",
        temperature=0.1,
    )

    template = """
    You are an assistant helping to classify topics within the following text: {analysis}
    Your objective is to find the kinds of topics mentionned in the text.
    
    The different kinds of topics are:
    - No side effect: return 0
    - Fatigue (or sleepiness, tiredness, no energy): return 1
    - Diarrhea: return 2
    - Arthralgia (and anything related to joint pain): return 3
    - Headaches: return 4
    - Nausea: return 5
    - Rash (and anything related to skin problems): return 6
    - Hair loss: return 7
    - Constipation: return 8
    - Mental health issues (depression, ...): return 9
    - Leg cramps (and anything related to muscle pain): return 10
    - Heart or blood pressure issues : return 11
    - Liver or kidney pain: return 12
    - Weight loss: return 13
    - Weight gain: return 14

    Act step by step.

    Return only the topics numbers.
    """

    prompt = PromptTemplate(template=template, input_variables=["analysis"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain(analysis)
    return answer["text"]
