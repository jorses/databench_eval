import pytest
import sys
import os

# Add the src directory to the system path to load the files directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from databench_eval.eval import Evaluator
from databench_eval.run import Runner
from databench_eval.utils import example_generator, example_postprocess
from datasets import Dataset

qa = Dataset.from_dict(
    {
        "answer": ["C", "C"],
        "question": ["A", "B"],
        "dataset": ["001_Forbes", "001_Forbes"],
        "type": ["category", "category"],
        "sample_answer": ["C", "C"],
    }
)
evaluator = Evaluator(qa=qa)


def test_evaluator_with_mocked_responses():
    responses = qa["answer"]
    score = evaluator.eval(responses)
    assert score == 1.0, "Evaluator should return 1.0 for all correct responses"


def test_runner_with_example_generator():
    runner = Runner(
        model_call=lambda x: ["C"] * len(x),
        prompt_generator=example_generator,
        qa=qa,
        batch_size=2,
    )
    prompts = ["AA", "BB"]
    responses = runner.run(prompts=prompts)
    assert len(responses) == len(qa), "Runner should generate responses for all prompts"


def test_runner_and_evaluator_integration():
    runner = Runner(
        model_call=lambda x: ["C"] * len(x),
        prompt_generator=example_generator,
        qa=qa,
        batch_size=2,
    )
    prompts = ["AA", "BB"]
    responses = runner.run(prompts=prompts)
    score = evaluator.eval(responses)
    assert (
        score == 1.0
    ), "Integration of Runner and Evaluator should return 1.0 for all correct responses"


# New test cases based on README.md behaviors


def test_evaluator_with_incorrect_responses():
    responses = ["A", "B"]
    score = evaluator.eval(responses)
    assert score == 0.0, "Evaluator should return 0.0 for all incorrect responses"


def test_runner_with_different_model_call():
    runner = Runner(
        model_call=lambda x: ["A"] * len(x),
        prompt_generator=example_generator,
        qa=qa,
        batch_size=2,
    )
    prompts = ["AA", "BB"]
    responses = runner.run(prompts=prompts)
    assert responses == ["A", "A"], "Runner should generate 'A' for all prompts"


def test_evaluator_with_mixed_responses():
    responses = ["C", "A"]
    score = evaluator.eval(responses)
    assert score == 0.5, "Evaluator should return 0.5 for half correct responses"


def test_runner_with_postprocessing():
    runner = Runner(
        model_call=lambda x: ["C_post\n"] * len(x),
        prompt_generator=example_generator,
        qa=qa,
        postprocess=lambda response, dataset: response.split("\n")[0],
        batch_size=2,
    )
    prompts = ["AA", "BB"]
    responses = runner.run(prompts=prompts)
    assert responses == [
        "C_post",
        "C_post",
    ], "Runner should apply postprocessing to responses"


def test_custom_compare_function():
    def custom_compare(value, truth, semantic):
        if semantic is None:
            return str(value) == str(truth)
        if semantic == "boolean":
            return str(value).lower() == str(truth).lower()
        elif "list" in semantic:
            return sorted(value) == sorted(truth)
        else:
            return str(value) == str(truth)

    custom_evaluator = Evaluator(qa=qa, compare=custom_compare)
    responses = ["C", "C"]
    score = custom_evaluator.eval(responses)
    assert score == 1.0, "Custom evaluator should return 1.0 for all correct responses"


def test_runner_save_to_file(tmp_path):
    file_path = tmp_path / "responses.txt"
    runner = Runner(
        model_call=lambda x: ["C"] * len(x),
        prompt_generator=example_generator,
        qa=qa,
        batch_size=2,
    )
    prompts = ["AA", "BB"]
    runner.run(prompts=prompts, save=file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
    assert lines == ["C\n", "C\n"], "Runner should save responses to file correctly"


def test_evaluator_with_lite_mode():
    responses = qa["sample_answer"]
    score = evaluator.eval(responses, lite=True)
    assert (
        score == 1.0
    ), "Evaluator should return 1.0 for all correct responses in lite mode"


def test_runner_with_empty_prompts():
    runner = Runner(
        model_call=lambda x: ["C"] * len(x),
        prompt_generator=example_generator,
        qa=qa,
        batch_size=2,
    )
    prompts = []
    with pytest.raises(ValueError):
        runner.run(prompts=prompts)
