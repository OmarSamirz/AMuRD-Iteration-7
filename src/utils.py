
import re
import torch
import json
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from models import (
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
)

import re
from typing import List 

from constants import ALL_STOPWORDS, GPC_PATH

def remove_strings(text: str, strings: List[str]) -> str:
    for s in strings:
        s = str(s)
        if s in text:
            text = text.replace(s, "")

    return text

def remove_numbers(text: str, remove_string: bool = False) -> str:
    text = text.split()
    text = [t for t in text if not re.search(r"\d", t)] if remove_string else [re.sub(r"\d+", "", t) for t in text]

    return " ".join(text)

def remove_stopwords(text: str):
    text = text.split()
    text = [t for t in text if t not in ALL_STOPWORDS or t == "can"]

    return " ".join(text)

def remove_punctuations(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)

    return " ".join(text.strip().split()) 

def remove_special_chars(text: str) -> str:
    text = re.sub(r"[-_/\\|]", " ", text)  

    return " ".join(text.strip().split())


def clean_text(row) -> str:
    text = row["Item_Name"]
    brand = row["Brand"]
    text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    # text = remove_stopwords(text)
    # if unit not in text and text == "":
    #     text += unit

    return  text

def load_embedding_model(config_path: str):
    with open(config_path, "r") as f:

        config_dict = json.load(f)
    try:
        config = SentenceEmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = SentenceEmbeddingModel(config)

    return model

def join_non_empty(*args):
    return " ".join([str(a).strip() for a in args if pd.notna(a) and str(a).strip()])
    
def load_gpc_to_classes(GPC_PATH):
    df = pd.read_excel(GPC_PATH)

    df["class_name"] = (
        df["BrickTitle"].fillna("") + " - " +
        df["AttributeTitle"].fillna("") + " - " +
        df["AttributeValueTitle"].fillna("")
    )

    df["description"] = df.apply(lambda row: join_non_empty(
        row["BrickDefinition_Includes"],
        row["BrickDefinition_Excludes"],
        row["AttributeDefinition"],
        row["AttributeValueDefinition"]
    ), axis=1)

    df_new = df[["class_name", "description"]]

    return df_new
