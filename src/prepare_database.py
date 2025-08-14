import pandas as pd
import teradataml as tdml
from teradataml import *

import json

from modules.logger import logger
from modules.db import TeradataDatabase
from utils import clean_text, load_embedding_model
from constants import TEST_DATA_PATH, E5_LARGE_INSTRUCT_CONFIG_PATH


def load_table_from_database(td_db: TeradataDatabase, table_name: str) -> tdml.DataFrame:
    tdf = td_db.execute_query(f"SELECT * FROM amurd.{table_name}")
    tdf = DataFrame(tdf)

    return tdf

def save_to_database(tdf: tdml.DataFrame, table_name: str, database_name: str) -> None:
    logger.info(f"The types in the dataframe in the database: {tdf.dtypes}")
    copy_to_sql(tdf, table_name, database_name, if_exists="replace")

def convert_to_embeddings(tdf: tdml.DataFrame) -> tdml.DataFrame:
    df = tdf.to_pandas()
    product_id = df["product_id"].tolist()
    product_name = df["product_name"].tolist()

    model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)

    embeddings = model.get_embeddings(product_name, "query")
    embeddings = embeddings.cpu().tolist()
    blobs = [json.dumps(embedding) for embedding in embeddings]

    df_product_embeddings = tdml.DataFrame({
        "product_id": product_id,
        "embeddings": blobs
    })

    return df_product_embeddings

def main():
    print("hello")
    td_db = TeradataDatabase()
    td_db.connect()
    tdf = load_table_from_database(td_db, "products")
    df = convert_to_embeddings(tdf)
    save_to_database(df, "product_embeddings", "amurd")


if __name__ == "__main__":
    main()