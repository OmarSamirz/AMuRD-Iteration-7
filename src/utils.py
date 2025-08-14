
import re
import json
from typing import List 

from constants import ALL_STOPWORDS, ALL_BRANDS
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

def clean_text(text: str) -> str:
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_brand_name(text)
    # text = remove_stopwords(text)
    # if unit not in text and text == "":
        # text += unit

    return  text.lower()

def load_embedding_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = SentenceEmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = SentenceEmbeddingModel(config)

    return model
