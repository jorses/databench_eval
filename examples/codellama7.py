import pandas as pd
import subprocess
import shlex
import zipfile

from datasets import Dataset
from databench_eval import Runner, Evaluator, utils
from llama_cpp import Llama

def call_gguf_model(prompts):
    results = []
    for p in prompts:
        escaped = p.replace('"', '\\"')
        cmd = f'llama-cli -m ./models/codellama-7b.Q4_K_M.gguf -p "{escaped}" -c 1024 -n 128 -ngl -1 --temp 0.1'
        args = shlex.split(cmd)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            results.append(result.stdout)
        except Exception as e:
            results.append(f"__CODE_GEN_ERROR__: {e}")

    return results


def example_generator(row: dict) -> str:
    """ IMPORTANT: 
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = load_sample_csv(dataset)
    return f'''
You are a pandas code generator. Your goal is to complete the function provided.
* You must not write any more code apart from that.
* You only have access to pandas and numpy.
* Pay attention to the type formatting .
* You cannot read files from disk.

import pandas as pd
import numpy as np

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
            + response.split("return")[1].split("\n")[0].strip().replace("[end of text]", "")
            + "\nans = answer(df)"
        )
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(exec_string, local_vars)

        ans = local_vars['ans']
        if isinstance(ans, pd.Series):
            ans = ans.tolist()
        elif isinstance(ans, pd.DataFrame):
            ans = ans.iloc[:, 0].tolist()
        return ans.split('\n')[0] if '\n' in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"


qa = utils.load_qa(name="qa") #.select(range(10))
runner_lite = Runner(
    model_call=call_gguf_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, loader=load_sample_csv
    ),
    qa=qa,
    batch_size=10,
)
evaluator = Evaluator(qa=qa)
responses_lite = runner_lite.run(save="predictions_lite_nosem.txt")
print(f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True) * 100} %") # ~30 %