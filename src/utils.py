import pandas as pd


def load_gpc_to_classes(GPC_PATH):
    df = pd.read_excel(GPC_PATH)

    df["class_name"] = (
        df["BrickTitle"].fillna("") + " - " +
        df["AttributeTitle"].fillna("") + " - " +
        df["AttributeValueTitle"].fillna("")
    )

    def join_non_empty(*args):
        return " ".join([str(a).strip() for a in args if pd.notna(a) and str(a).strip()])

    df["description"] = df.apply(lambda row: join_non_empty(
        row["BrickDefinition_Includes"],
        row["BrickDefinition_Excludes"],
        row["AttributeDefinition"],
        row["AttributeValueDefinition"]
    ), axis=1)