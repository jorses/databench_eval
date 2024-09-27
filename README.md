# DataBench

This repo contains a simple evaluation framework for the DataBench suite for QA over Tabular Data.
It is intended to provide a common framework and language to train and evaluate approaches to the task.

Since the task is very open and a number of decisions need to be taken in order to standardize for in
very different approaches, I've developed a simple framework so that only three decisions need to be taken:
* How to **build the prompt** given a question and the dataset that contains the answer
* The actual **model call** (with batch support)
* The **evaluation function** since what constitutes a valid answer is very system-dependant.

We also streamline prompt generation and processing, so we have a good setup to iterate.

## Installation

````
pip install databench-eval
````

## Basic Usage

```python
from databench_eval import Runner, Evaluator

def model_call(prompts: list[str]) -> list[str]:
    """ Call your model on a batch of prompts here. """
    return "'mock response'" * len(prompts)

responses = Runner(model_call).run(prompts)

acc = Evaluator().eval(responses)
acc_lite = Evaluator().eval(responses, lite=True)
```

## Runner
The runner is the class that calls the model.

### QA
If not QA is provided, it will download the full QA set from databench_eval.
You can choose any QA subset you might like, for example
```python
from databench_eval import Runner, Evaluator
from databench_eval.utils import load_qa

qa = load_qa(name="semeval", split="dev")

Runner(model_call, qa=qa_dev).run()
```

### Prompt Generation
Instead of prompts, you might pass a `prompt_generator` function to the Runner.
This receives a row from the qa dataset and is
expected to return a string containing the prompt.

This might be useful to quickly prototype or test out new prompts.
For example, testing the effect of knowing (or not) the semantic beforehand in
the function calls.

```python
from databench_eval.utils import load_table
def example_generator(row: dict) -> str:

    df = load_table(row["dataset"])
    question = row["question"]
    semantic = row["type"]
    return f'''
    # You must complete the following function
    def answer(df: pd.DataFrame) -> {semantic}:
        """Returns the answer to {question}"""
        df.columns = {list(df.columns)}
        return'''
custom_runner = Runner(model_call, prompt_generator=prompt_generator)
```

### Postprocessing Function

Similarly, you might implement a preprocess or postprocessing function for the model responses.
For example, let's say we have a model with a tendency to just continue rambling after providing us with the answer, and we've observed the result is greatly enhanced by just taking the first line.

By default, no postprocessing is implemented.

```python
from databench_eval import Runner

def custom_postp(response: str, dataset: str):
    return response.split("\n")[0]

custom_runner = Runner(model_call, postprocess=custom_postp)
```

The actual returns of the model_calls will be stored in Runner.raw_responses,
while the 

### Number of batches

By default, the Runner processes the dataset in batches of 10. You can change this by setting the `batch_size` parameter when initializing it.

### Save to File

You can save the model responses to a txt file. The format will be that used during the SemEval 2025 competition,
one result per line.
```
Runner.run(..., save="/path/to/responses.txt")
```

## Evaluator

In the end how accurate or useful a model is to you heavily depends on what you want to do with the model responses.
For example, someone looking to implement a solution that couples to a wider automated system where output format is key might want their responses
to have a perfect very specific format, while other applications like chatbots would get away with any kind of format as long as a human could
understand it.

### Compare Function
We have provided a basic evaluation function that is meant to serve as base evaluator.
This will be the function used to evaluate results in the Task 8 of SemEval 2025.

```python
# Evaluator.default_compare

def default_compare(value, truth, semantic=None):
    """ Default evaluation function. """
    return str(value).strip() == str(truth).strip()
```

The eval function of evaluator can be overridden, either by monkey-patching or by using the `compare` argument in the Evaluator instantiation.

As a very simple example of the kind of behaviour that I'm talking about before, let's say that we would like for the order of list-like questions to be valid regardless of their order.

```python
from databench_eval import Evaluator

def custom_compare(value, truth, semantic=None):
    """ Custom evaluation function. """
    if "list" in semantic:
        return sorted(value) == sorted(truth)
    else:
        return str(value) == str(truth)

custom_evaluator = Evaluator(compare=custom_compare)
```


### Read from file

You can also read from a file by passing the path to it.
The format expected will be that used during the SemEval 2025 competition,
one result per line.
```python
Evaluator().eval(..., save="path/to/file.txt")
```

### Number of batches

By default, the Evaluator processes the dataset in batches of 10. You can change this by setting the `batch_size` parameter when initializing it.


## Utils

Although decoupled from the main classes, I've included a couple of useful functions in here.
The main functions are:

* load_qa: load the qa sets from HuggingFace
* load_table: loads a data table from a given id (the "dataset" column of the QA table) from HuggingFace

HF is supposed to download each file only once, then just read from the cache, but in my own testing it has been
faster to just download the files locally and then passing the 
Still I've decided to include the utils since they are convenient for easier testing.

## Examples

Check out the `examples` folder for more complex showcases of this library.
`llamacpp.py` in particular carries out a full semeval 2025 task 8 submission.

## Contact

The easiest way to reach me is at jorgeosesgrijalba@gmail.com