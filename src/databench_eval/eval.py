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

    def default_compare(self, value, truth, semantic=None):
        return str(value).strip() == str(truth).strip()

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

        for response, truth in tqdm(zip(responses, truths), total=len(truths)):
            if self.compare(response, truth, "lite" if lite else None):
                correct += 1

        return correct / len(truths)
