from typing import Callable, List, Optional
from .utils import load_qa
from datasets import Dataset
from tqdm import tqdm


class Runner:
    def __init__(
        self,
        model_call: Callable,
        prompt_generator: Optional[Callable] = None,
        postprocess: Optional[Callable] = None,
        qa: Optional[Dataset] = None,
        batch_size: int = 10,
        **kwargs,
    ):
        self.model_call = model_call
        self.prompt_generator = prompt_generator
        if postprocess is not None:
            self.postprocess = postprocess

        self.raw_responses = []
        self.responses = []
        self.prompts = []
        self.qa: Dataset = qa if qa is not None else load_qa(**kwargs)
        self.batch_size = batch_size

    def process_prompt(self, prompts, datasets):
        raw_responses = self.model_call(prompts)
        responses = [
            self.postprocess(response=raw_response, dataset=dataset)
            for raw_response, dataset in zip(raw_responses, datasets)
        ]

        self.prompts.extend(prompts)
        self.raw_responses.extend(raw_responses)
        self.responses.extend(responses)

    def run(
        self,
        prompts: Optional[list[str]] = None,
        save: Optional[str] = None,
    ) -> List[str]:
        if prompts is not None:
            if len(prompts) != len(self.qa):
                raise ValueError("n_prompts != n_qa")

            for i in tqdm(range(0, len(prompts), self.batch_size)):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_datasets = self.qa[i : i + self.batch_size]["dataset"]
                self.process_prompt(batch_prompts, batch_datasets)
        else:
            if self.prompt_generator is None:
                raise ValueError("Generator must be provided if prompts are not.")
            for i in tqdm(range(0, len(self.qa), self.batch_size)):
                batch_rows = self.qa.select(
                    range(i, min(i + self.batch_size, len(self.qa)))
                )
                batch_prompts = [self.prompt_generator(row) for row in batch_rows]
                batch_datasets = [row["dataset"] for row in batch_rows]
                self.process_prompt(batch_prompts, batch_datasets)

        if save is not None:
            self.save_responses(save)
        return self.responses

    def save_responses(self, save_path: str) -> None:
        with open(save_path, "w") as f:
            for response in self.responses:
                f.write(str(response) + "\n")

    def postprocess(self, response, dataset):
        return response
