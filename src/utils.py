import json
from pathlib import Path
from teradata_db import TeradataDatabase  # Assuming your class is saved here

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

