
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
    text = row.Item_Name
    brand = row.Brand
    unit = str(row.Unit)
    text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    if unit not in text and text == "":
        text += unit

    return  text

def get_text(*fields):
    return " ".join([f.strip() for f in fields if f and f.strip()])

def insert_classes(db, schema):
    for l1 in schema:
        for l2 in l1.get("Childs", []):
            for l3 in l2.get("Childs", []):
                for brick in l3.get("Childs", []):  # Brick
                    brick_name = brick.get("Title", "")
                    brick_desc = get_text(brick.get("Definition", ""), brick.get("DefinitionExcludes", ""))

                    for attr in brick.get("Childs", []):  # Attribute Type
                        attr_name = attr.get("Title", "")
                        attr_desc = get_text(attr.get("Definition", ""), attr.get("DefinitionExcludes", ""))

                        for val in attr.get("Childs", []):  # Attribute Value
                            val_name = val.get("Title", "")
                            val_desc = get_text(val.get("Definition", ""), val.get("DefinitionExcludes", ""))

                            class_name = f"{brick_name} - {attr_name} - {val_name}"
                            description = get_text(brick_desc, attr_desc, val_desc)

                            query = f"""
                                INSERT INTO amurd.classes (class_name, description)
                                VALUES ('{class_name.replace("'", "''")}', '{description.replace("'", "''")}')
                            """
                            db.execute_query(query)

