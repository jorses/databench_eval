import pandas as pd
from datasets import load_dataset, Dataset


def load_qa(**kwargs) -> Dataset:
    return load_dataset(
        "cardiffnlp/databench", **{"name": "qa", "split": "train", **kwargs}
    )


def load_table(name):
    return pd.read_parquet(
        f"hf://datasets/cardiffnlp/databench/data/{name}/all.parquet"
    )


def load_sample(name):
    return pd.read_parquet(
        f"hf://datasets/cardiffnlp/databench/data/{name}/sample.parquet"
    )


def example_generator(row: dict) -> str:
    dataset = row["dataset"]
    question = row["question"]
    df = load_table(dataset)
    return f"""
# TODO: complete the following function in one line. It should give the answer to: How many rows are there in this dataframe? 
def example(df: pd.DataFrame) -> int:
    df.columns=["A"]
    return df.shape[0]

# TODO: complete the following function in one line. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row["type"]}:
    df.columns = {list(df.columns)}
    return"""


def extract_first_line(
    response: str,
):
    return response


def example_postprocess(response: str, dataset: str):
    df = load_table(dataset)
    lead = """
def answer(df):
    return """
    exec(
        "global ans\n" + lead + response.split("\n")[0].strip() + f"\nans = answer(df)"
    )
    return ans
