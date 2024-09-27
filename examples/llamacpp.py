import pandas as pd
import subprocess
import shlex
import zipfile

from datasets import Dataset
from databench import Runner, Evaluator, utils


# this makes use of https://huggingface.co/TheBloke/stable-code-3b-GGUF
# and https://github.com/ggerganov/llama.cpp
def call_gguf_model(prompts):
    results = []
    for p in prompts:
        escaped = p.replace('"', '\\"')
        cmd = f'llama-cli -m ./models/stable-code-3b.Q4_K_M.gguf -p "{escaped}" -c 1024 -n 128'
        args = shlex.split(cmd)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            results.append(result.stdout)
        except Exception as e:
            results.append(f"__CODE_GEN_ERROR__: {e.stderr}")

    return results


def example_generator(row: dict) -> str:
    """IMPORTANT:
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = utils.load_table(dataset)
    return f"""
# TODO: complete the following function in one line. It should give the answer to: How many rows are there in this dataframe? 
def example(df: pd.DataFrame) -> int:
    df.columns=["A"]
    return df.shape[0]

# TODO: complete the following function in one line. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row["type"]}:
    df.columns = {list(df.columns)}
    return"""


def example_postprocess(response: str, dataset: str, loader):
    try:
        df = loader(dataset)
        lead = """
def answer(df):
    return """
        exec(
            "global ans\n"
            + lead
            + response.split("return")[2]
            .split("\n")[0]
            .strip()
            .replace("[end of text]", "")
            + f"\nans = answer(df)"
        )
        # no true result is > 1 line atm, needs 1 line for txt format
        return ans.split("\n")[0] if "\n" in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"


qa = utils.load_qa(name="semeval", split="dev")
qa = Dataset.from_pandas(pd.DataFrame(qa))
evaluator = Evaluator(qa=qa)

runner = Runner(
    model_call=call_gguf_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, utils.load_table
    ),
    qa=qa,
    batch_size=10,
)

runner_lite = Runner(
    model_call=call_gguf_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, utils.load_sample
    ),
    qa=qa,
    batch_size=10,
)

responses = runner.run(save="predictions.txt")
responses_lite = runner_lite.run(save="predictions_lite.txt")
print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.16
print(
    f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}"
)  # ~0.08


with zipfile.ZipFile("submission.zip", "w") as zipf:
    zipf.write("predictions.txt")
    zipf.write("predictions_lite.txt")

print("Created submission.zip containing predictions.txt and predictions_lite.txt")
