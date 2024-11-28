from typing import Callable, List, Union, Optional
from .utils import load_qa
from datasets import Dataset
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        compare: Optional[Callable] = None,
        qa: Optional[Dataset] = None,
        batch_size: int = 10,
        **kwargs,
    ):
        self.compare = compare if compare else self.default_compare
        self.qa = qa if qa is not None else load_qa(**kwargs)

    def default_compare(self, value, truth, semantic):
        semantic = semantic.strip()
        if semantic == "boolean":
            return str(value).strip() == str(truth).strip()
        elif semantic == "category":
            return str(value).strip() == str(truth).strip()
        elif semantic == "number":
            try:
                value_cleaned = ''.join(char for char in str(value) if char.isdigit() or char in ['.', '-'])
                truth_cleaned = ''.join(char for char in str(truth) if char.isdigit() or char in ['.', '-'])
                return round(float(value_cleaned), 2) == round(float(truth_cleaned), 2)
            except:
                return False
        elif semantic == "list[category]":
            try:
                value_list = [item.strip().strip("'").strip('"') for item in str(value).strip('[]').split(',')]
                truth_list = [item.strip().strip("'").strip('"') for item in str(truth).strip('[]').split(',')]
                if len(value_list) != len(truth_list):
                    return False
                
                return set(value_list) == set(truth_list)
            except Exception as exc:
                return False

        elif semantic == "list[number]":
            try:
                value_list = sorted(round(float(''.join(c for c in v.strip() if c.isdigit() or c in ['.', '-'])), 2) for v in str(value).strip('[]').split(',') if v.strip())
                truth_list = sorted(round(float(''.join(c for c in t.strip() if c.isdigit() or c in ['.', '-'])), 2) for t in str(truth).strip('[]').split(',') if t.strip())
                
                if len(value_list) != len(truth_list):
                    return False
                
                return set(value_list) == set(truth_list)
            except Exception as exc:
                return False

        else:
            raise Exception(f"Semantic not supported: {semantic}")

    def eval(
        self,
        responses: Union[List[str], str],
        lite: bool = False,
    ) -> float:
        if isinstance(responses, str):
            with open(responses, "r") as f:
                responses = f.read().splitlines()

        correct = 0
        truths = self.qa["answer"] if not lite else self.qa["sample_answer"]
        evals = []
        for response, truth, semantic in tqdm(zip(responses, truths, self.qa["type"]), total=len(truths)):
            truthy = self.compare(response, truth, semantic)
            if self.compare(response, truth, semantic):
                correct += 1
            evals.append(truthy)
        self.evals = evals
        return correct / len(truths)
