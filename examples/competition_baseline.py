import pandas as pd
import numpy as np
import subprocess
import shlex
import zipfile

from datasets import Dataset
from databench_eval import Runner, Evaluator

"""
* This is the baseline evaluation for the January SemEval 2025 Competition.
* It scores ~26% in DataBench and ~27% in DataBench lite.
* To run this file, make use of the answers.zip file as well as the data files
you were provided with in the competition phase.
* It only makes use of a quantized version of a tiny 3B model, which can be run locally
on most consumer hardware using only CPU. No finetuning or further configuration is needed.

The test part will be available in the DataBench HuggingFace page soon.
"""


# this makes use of https://huggingface.co/TheBloke/stable-code-3b-GGUF
# and https://github.com/ggerganov/llama.cpp
def call_gguf_model(prompts):
    results = []
    for p in prompts:
        escaped = p.replace('"', '\\"')
        cmd = f'llama-cli -m ../models/stable-code-3b.Q4_K_M.gguf -p "{escaped}" -c 1024 -n 128 -ngl -1'
        args = shlex.split(cmd)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            results.append(result.stdout)
        except Exception as e:
            results.append(f"__CODE_GEN_ERROR__")

    return results


def example_generator(row: dict) -> str:
    dataset = row["dataset"]
    question = row["question"]
    df = load_sample(dataset)
    return f'''
You are a pandas code generator. Your goal is to complete the function provided.
* You must not write any more code apart from that.
* You only have access to pandas and numpy.
* Pay attention to the type formatting .
* You cannot read files from disk.
* The answer should be short and concise, in the format I specify.
* DO NOT do anything else other than filling the function provided.
* Answer in one of the following formats, depending on the question
    1. True/False (do not answer with np.True_ or np.False_, but rather True or False)
    2. with a value from the dataframe, (category/number)
    3. with a list of values (either categories or numbers)

import pandas as pd
import numpy as np

# This is an example
def example(df: pd.DataFrame):
    """Returns the answer to the question: How many rows are there in the dataframe? """
    df.columns = {list(df.columns)}
    return df.shape[0]

# This is the question you have to answer
def answer(df: pd.DataFrame):
    """Returns the answer to the question: {question} """
    df.columns = {list(df.columns)}
    return'''


def example_postprocess(response: str, dataset: str, loader):
    try:
        df = loader(dataset)
        global ans
        lead = """
def answer(df):
    return """
        exec_string = (
            lead
            + response.split("return")[2]
            .split("\n")[0]
            .strip()
            .replace("[end of text]", "")
            + "\nans = answer(df)"
        )
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(exec_string, local_vars)

        ans = local_vars["ans"]
        if isinstance(ans, pd.Series):
            ans = ans.tolist()
        elif isinstance(ans, pd.DataFrame):
            ans = ans.iloc[:, 0].tolist()
        return ans.split("\n")[0] if "\n" in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"


# Load the main QA data from three .txt files into a DataFrame
qa_df = pd.DataFrame()

# Read the answers from the .txt files into separate lists
with open("answers.txt", "r") as f:
    answers = f.read().splitlines()

with open("answers_lite.txt", "r") as f:
    sample_answers = f.read().splitlines()

with open("semantics.txt", "r") as f:
    semantics = f.read().splitlines()

# Combine the lists into a DataFrame

# Load the dataset column from the specified file
qa_df = pd.read_csv(
    "test_qa.csv" # file provided in the competition
)
qa_df["answer"] = answers
qa_df["sample_answer"] = sample_answers
qa_df["type"] = semantics


# Convert to Dataset
qa = Dataset.from_pandas(qa_df.head(100))
evaluator = Evaluator(qa=qa)


def load_table(dataset):
    return pd.read_parquet(f"./{dataset}/all.parquet")


def load_sample(dataset):
    return pd.read_parquet(f"./{dataset}/sample.parquet")


runner = Runner(
    model_call=call_gguf_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, load_table
    ),
    qa=qa,
    batch_size=10,
)

runner_lite = Runner(
    model_call=call_gguf_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, load_sample
    ),
    qa=qa,
    batch_size=10,
)

responses = runner.run(save="predictions.txt")
responses_lite = runner_lite.run(save="predictions_lite.txt")
print(f"DataBench accuracy is {evaluator.eval(responses)}")
print(f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}")


with zipfile.ZipFile("submission.zip", "w") as zipf:
    zipf.write("predictions.txt")
    zipf.write("predictions_lite.txt")

print("Created submission.zip containing predictions.txt and predictions_lite.txt")
