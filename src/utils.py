import torch
import torch.nn.functional as F
import pandas as pd

import re
import json
from typing import List, Tuple

from constants import ALL_STOPWORDS, ALL_BRANDS, GPC_PATH
from modules.models import SentenceEmbeddingModel, SentenceEmbeddingConfig

def remove_brand_name(text: str) -> str:
    for brand in ALL_BRANDS:
        if brand in text:
            text = text.replace(brand, "")
            break

    return text

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

    return " ".join(text.strip().split()).lower()

def clean_text(row) -> str:
    text = row["Item_Name"]
    brand = row["Brand"]
    text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_brand_name(text)
    # text = remove_stopwords(text)
    # if unit not in text and text == "":
        # text += unit

    return text.lower()

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
    
def load_gpc_to_classes():
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

def cluster_topk_classes(cluster_embeddings: List[List[float]], classes_embeddings: List[List[float]], k: int) -> int:
    cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
    classes_embeddings = F.normalize(classes_embeddings, p=2, dim=1)

    scores = (cluster_embeddings @ classes_embeddings.T)

    topk_classes = torch.topk(scores, k=k)

    return topk_classes[1]

