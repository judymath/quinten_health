import health
import yaml
from dotenv import load_dotenv
import os

import pandas as pd
from health.data import preprocess_topic, postprocess_topic
from health.topic import gpt_analyze, topic_extract
from langchain import HuggingFaceHub

load_dotenv()

# HuggingFace API
os.environ[
    "HUGGINGFACEHUB_API_TOKEN"
] = "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"  # (à garder secret si possible)

llm = HuggingFaceHub(
    repo_id="timdettmers/guanaco-33b-merged",
    model_kwargs={"temperature": 0.1, "max_new_tokens": 100},
)

with open("config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


def main(config):
    data_path = config["data"]["raw_data"]

    data = pd.read_csv(data_path)
    data_preprocessed = preprocess_topic(data)

    analyze_prompt_path = config["model"]["analyze_prompt"]
    extract_prompt_path = config["model"]["extract_prompt"]

    with open(analyze_prompt_path, "r") as prompt_file:
        analyze_prompt = prompt_file.read()

    with open(extract_prompt_path, "r") as prompt_file:
        extract_prompt = prompt_file.read()

    # Apply gpt analysis function to each comment
    data_preprocessed["gpt_analysis"] = data_preprocessed["comment"].apply(
        gpt_analyze, args=[llm, analyze_prompt]
    )

    # Apply topic extraction function to each gpt_analysis
    data_preprocessed["topics"] = data_preprocessed["gpt_analysis"].apply(
        topic_extract, args=[llm, extract_prompt]
    )

    data_postprocessed = postprocess_topic(data_preprocessed)
    print(data_postprocessed.head())


if __name__ == "__main__":
    main(config)
