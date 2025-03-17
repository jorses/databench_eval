import pandas as pd
import numpy as np
import asyncio
import nest_asyncio

from databench_eval import Runner, Evaluator
from databench_eval.utils import load_qa
from openai import AsyncOpenAI

nest_asyncio.apply()

"""
* This is the "proprietary" baseline evaluation for the April 2025 Iberlef Presta Dev Phase.
* It scores ~49% on the DataBenchSPA dataset.
* It makes use of the OpenAI API to generate code for the questions.
* It creates a `predictions.txt` that can then be used as submission.
* You are encouraged to load the parquets locally, as it dramatically speeds up the process,
but you can also load them from the HuggingFace datasets. Change the paths accordingly.
* Running time total should be <10s
* There is also an example "column generator" that you might use to aid in filtering questions beforehand,
to aid in small-sized context windows.
"""

OPEN_AI_KEY = "YOUR_API_KEY_HERE"


def call_model_sync(prompts):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(call_model(prompts))

async def call_model(prompts):
    client = AsyncOpenAI(api_key=OPEN_AI_KEY)

    results = []

    async def process_prompt(p):
        try:
            completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": p}],
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "__CODE_GEN_ERROR__"

    tasks = [process_prompt(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

def column_generator(row: dict) -> str:
    dataset = row["dataset"]
    question = row["question"]
    # alternatively, you can load the parquet from the HuggingFace datasets
    df = pd.read_parquet(f"hf://datasets/SINAI/databenchSPA/{dataset}/all.parquet")
    return f'''
From these columns: {df.columns.tolist()}, select the column(s) to answer this question: {question}.
* The format expected is something that can be parsed as a python list of strings, containing the list of the columns.
* DO NOT ANSWER ANYTHING ELSE APPART FROM THIS; NOT EVEN A SINGLE CHARACTER. DO NOT WRAP YOUR RESPONSE in ```or anything.
* Try to use as few columns as possible while being able to answer the question. It's always one or two columns, never more.
'''

def prompt_generator(row: dict) -> str:
    dataset = row["dataset"]
    question = row["question"]
    df = pd.read_parquet(f"./databenchSPA/{dataset}/all.parquet")
    return f'''
You are a pandas code generator. Your goal is to complete the function provided.
* You must not write any more code apart from that.
* You only have access to pandas and numpy.
* Pay attention to the type formatting .
* You cannot read files from disk.
* The answer should be short and concise, in the format I specify.
* DO NOT do anything else other than filling the function provided.
* DO NOT wrap your code in ```python``` marks. Answer in plain text.
* Answer in one of the following formats, depending on the question
1. True/False (do not answer with np.True_ or np.False_, but rather True or False)
2. with a value from the dataframe, (category/number)
3. with a list of values (either categories or numbers)

import pandas as pd
import numpy as np
import openai
import os
import asyncio

# This is an example
def example(df: pd.DataFrame):
"""Returns the answer to the question: How many rows are there in the dataframe? """
df.columns = {list(df.columns)}
return df.shape[0]

# This is the question you have to answer
def answer(df: pd.DataFrame):
"""Returns the answer to the question: {question} """
df.columns = {list(df.columns)}
return '''


def example_postprocess(response: str, dataset: str, loader):
    try:
        df = loader(dataset)
        global ans
        lead = """
def answer(df):
    return """
        exec_string = (
            lead
            + response
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


qa_dev = load_qa(lang="ES", name="iberlef", split="dev")
runner = Runner(
    model_call=call_model_sync,
    prompt_generator=prompt_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, 
        lambda x: pd.read_parquet(f"./databenchSPA/{dataset}/all.parquet")
    ),
    qa=qa_dev,
    batch_size=100,
)
responses = runner.run(save="predictions.txt")
evaluator = Evaluator(qa=qa_dev)
print(f"DataBenchSPA accuracy is {evaluator.eval(responses)}")