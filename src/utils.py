
import re
from typing import List 

from constants import ALL_STOPWORDS

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

def clean_text(row) -> str:
    text = row.product_name
    brand = row.brand_name
    text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    # text = remove_stopwords(text)
    # if unit not in text and text == "":
    #     text += unit

    return  text

from constants import GPC_PATH



