"""
Preprocess the csv file with comments
"""

import re
import string
import warnings

import numpy as np
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob

warnings.simplefilter("ignore")


def preprocess_review(review):
    """
    Preprocess a text review to prepare it for analysis.

    This function takes a text review and performs a series of
    preprocessing steps, including converting the text to lowercase,
    removing HTML entities, URLs, special characters, non-ASCII
    characters, and extra spaces, and then lemmatizing the text using spaCy.

    Parameters:
    review (str): The input text review to be preprocessed.

    Returns:
    str: The preprocessed and lemmatized text review.
    """
    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    # Changing to lowercase
    review = review.lower()

    # Removing HTML entities (e.g., &amp;, &lt;, &gt;)
    review = re.sub(r"&\w+;", "", review)

    # Removing URLs
    review = re.sub(r"http\S+|www.\S+", "", review)

    # Removing special characters and punctuation
    review = re.sub(r"[^\w\s]", "", review)

    # Removing all non-ASCII characters
    review = re.sub(r"[^\x00-\x7F]+", "", review)

    # Removing leading and trailing whitespaces
    review = review.strip()

    # Replacing multiple spaces with a single space
    review = re.sub(r"\s+", " ", review)

    # Replacing two or more dots with one
    review = re.sub(r"\.{2,}", " ", review)

    # Lemmatization using spaCy
    doc = nlp(review)
    lemmatized_review = " ".join(
        [token.lemma_ if token.lemma_ != "-PRON-" else token.text for token in doc]
    )

    return lemmatized_review


def preprocess_data(data):
    """
    Preprocess a DataFrame containing text reviews and
    extract various text-based features.

    This function applies preprocessing to text reviews
    in a DataFrame, including removing stopwords, performing
    sentiment analysis, and extracting text-based features
    such as word counts, punctuation counts, and more.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing text
    reviews in the 'comment' column.

    Returns:
    pandas.DataFrame: The input DataFrame with added columns
    for preprocessed text, sentiment analysis results, and
    text-based features.

    """
    # Apply the preprocessing function to the 'comment' column
    data["review_clean"] = data["comment"].apply(preprocess_review)

    # Removing stopwords
    stop_words = set(stopwords.words("english"))
    data["review_clean"] = data["review_clean"].apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words)
    )

    # Sentiment analysis and polarity
    data["sentiment"] = data["comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
    data["sentiment_clean"] = data["review_clean"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    # Other text-based features
    data["count_word"] = data["review_clean"].apply(lambda x: len(str(x).split()))
    data["count_unique_word"] = data["review_clean"].apply(
        lambda x: len(set(str(x).split()))
    )
    data["count_letters"] = data["review_clean"].apply(lambda x: len(str(x)))
    data["count_punctuations"] = data["comment"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )
    data["count_words_upper"] = data["comment"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    data["count_words_title"] = data["comment"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()])
    )
    data["count_stopwords"] = data["comment"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in stop_words])
    )
    data["mean_word_len"] = data["review_clean"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()])
    )

    return data


def preprocess_topic(df):
    """
    Preprocess a DataFrame containing medication
    and disease information.

    This function performs various data preprocessing
    steps on the input DataFrame to prepare it for further analysis.
    It removes unnecessary columns, selects comments
    related to Crohn's Disease and Ulcerative Colitis,
    extracts the drug and disease information, and
    removes a specific row that may trigger OpenAI's policy.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing
    medication and disease information.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame
    ready for further analysis.

    """
    # Drop text_index column
    df.drop(columns="text_index", inplace=True)

    # Select only comments for Crohn's Disease and Ulcerative Colitis
    df = df.loc[
        df["medication"].str.contains("Crohn")
        | df["medication"].str.contains("Ulcerative")
    ]

    # Drop rows where medication is not specified
    df = df.loc[~df["medication"].str.contains("For")]

    # Separate drug from disease
    df["drug"] = df["medication"].str.split("for").str[0]
    df["disease"] = np.where(
        df["medication"].str.contains("Crohn"),
        "Crohn's Disease",
        np.where(df["medication"].str.contains("Ulcerative"), "Ulcerative Colitis", ""),
    )
    df.drop(columns="medication", inplace=True)

    # Drop row n°3 (triggering OpenAI's policy)
    df = df.drop(3)

    return df


def postprocess_topic(df):
    """
    Postprocess a DataFrame containing topic
    extraction results.

    This function takes a DataFrame with a
    'topics' column and applies post-processing
    to extract specific topics of interest
    based on the content of the 'topics' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame
    containing topic extraction results. It should
    have a 'topics' column where topics have
    been extracted using a suitable method.

    Returns:
    pandas.DataFrame: The input DataFrame with
    additional columns for specific topics
    extracted from the 'topics' column."""

    # Apply topic extraction function to each gpt_analysis
    df["no_side_effects"] = df["topics"].str.contains(" 0|Ex") & ~df[
        "topics"
    ].str.contains(", 0|0,")
    df["fatigue"] = (
        df["topics"].str.contains("1") & ~df["topics"].str.contains("10|11|12|13|14")
    ) | df["topics"].str.contains("1,")
    df["diarrhea"] = (
        df["topics"].str.contains("2") & ~df["topics"].str.contains("12")
    ) | (df["topics"].str.contains("2,") & ~df["topics"].str.contains("12,"))
    df["arthralgia"] = (
        df["topics"].str.contains("3") & ~df["topics"].str.contains("13")
    ) | (df["topics"].str.contains("3,") & ~df["topics"].str.contains("13,"))
    df["headaches"] = (
        df["topics"].str.contains("4") & ~df["topics"].str.contains("14")
    ) | (df["topics"].str.contains("4,") & ~df["topics"].str.contains("14,"))
    df["nausea"] = df["topics"].str.contains("5")
    df["rash"] = df["topics"].str.contains("6")
    df["hair loss"] = df["topics"].str.contains("7")
    df["constipation"] = df["topics"].str.contains("8")
    df["mental_health_issues"] = df["topics"].str.contains("9")
    df["leg_cramps"] = df["topics"].str.contains("10")
    df["heart_blood_pressure_issues"] = df["topics"].str.contains("11")
    df["liver_kidney_pain"] = df["topics"].str.contains("12")
    df["weight_loss"] = df["topics"].str.contains("13")
    df["weight_gain"] = df["topics"].str.contains("14")
    return df
