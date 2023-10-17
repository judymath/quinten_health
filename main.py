"""
Create the processed file on call
"""

import logging
import os

import pandas as pd
import yaml
from langchain.llms import HuggingFaceHub
from tqdm import tqdm
from dotenv import load_dotenv
from health.data import postprocess_topic, preprocess_topic
from health.topic import gpt_analyze, topic_extract


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

with open("config/config.yaml", "r", encoding="utf-8") as config_file:
    configuration = yaml.safe_load(config_file)

load_dotenv()

# HuggingFace API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = configuration["model"]["key"]

llm = HuggingFaceHub(
    repo_id="timdettmers/guanaco-33b-merged",
    model_kwargs={"temperature": 0.1, "max_new_tokens": 100},
)


def main(config):
    """
    Function to proceed topic extraction and create an output csv

    Parameters:
    config (dict): The configuration dict to load
    """
    tqdm.pandas()
    data_path = config["data"]["raw_data"]

    data = pd.read_csv(data_path)
    data_preprocessed = preprocess_topic(data)

    analyze_prompt_path = config["model"]["analyze_prompt"]
    extract_prompt_path = config["model"]["extract_prompt"]

    with open(analyze_prompt_path, "r", encoding="utf-8") as prompt_file:
        analyze_prompt = prompt_file.read()

    with open(extract_prompt_path, "r", encoding="utf-8") as prompt_file:
        extract_prompt = prompt_file.read()

    # Apply gpt analysis function to each comment
    logging.info("Start gpt analyis")
    data_preprocessed["gpt_analysis"] = data_preprocessed["comment"].progress_apply(
        gpt_analyze, args=[llm, analyze_prompt]
    )

    # Apply topic extraction function to each gpt_analysis
    logging.info("Start topic extraction")
    data_preprocessed["topics"] = data_preprocessed["gpt_analysis"].progress_apply(
        topic_extract, args=[llm, extract_prompt]
    )

    data_postprocessed = postprocess_topic(data_preprocessed)
    logging.info("Create output csv")
    data_postprocessed.to_csv(config["data"]["output_data"], index=False)


if __name__ == "__main__":
    main(configuration)
    logging.info("End of treatments")
