import pytest
from databench_eval.eval import Evaluator
from databench_eval.utils import load_qa

def test_boolean_comparisons():
    evaluator = Evaluator()
    test_cases = [
        (True, "True", True),
        ("True", True, True),
        ("[True]", "True", True),
        ("False", "False", True),
        (" True ", "True", True),
        ("false", "False", True),
        ("True", "yes", True),
        ("False", "no", True),
        ("True", "no", False),
        ("False", "yes", False),
        ("true", "yes", True),
        ("false", "no", True),
        ("yes", "yes", True),
        ("no", "no", True),
        (None, "True", False),
    ]
    
    for value, truth, expected in test_cases:
        assert evaluator.default_compare(value, truth, "boolean") == expected, \
            f"Failed boolean comparison: {value} vs {truth}"

def test_category_string_comparisons():
    evaluator = Evaluator()
    test_cases = [
        ("Barack Obama", "Barack Obama", True),
        (" Barack Obama ", "Barack Obama", True),
        ("T-rex", "T-rex", True),
        ("", "", True),
        (" ", "", True),
        ("Different", "Strings", False),
        (None, "String", False),
    ]
    
    for value, truth, expected in test_cases:
        assert evaluator.default_compare(value, truth, "category") == expected, \
            f"Failed category string comparison: {value} vs {truth}"

def test_category_date_comparisons():
    evaluator = Evaluator()
    test_cases = [
        ("2016-12-22 00:00:00+00:00", "2016-12-22", True),
        ("2016-12-22T00:00:00Z", "2016-12-22", True),
        ("22/12/2016", "2016-12-22", True),
        ("Dec 22, 2016", "2016-12-22", True),
        ("2016-12-22", "2016-12-23", False),
        ("Invalid Date", "2016-12-22", False),
        ("2016-13-45", "2016-12-22", False),  # Invalid date
    ]
    
    for value, truth, expected in test_cases:
        assert evaluator.default_compare(value, truth, "category") == expected, \
            f"Failed category date comparison: {value} vs {truth}"

def test_number_comparisons():
    evaluator = Evaluator()
    test_cases = [
        ("123.45", 123.45, True),
        ("$123.45", "123.45", True),
        ("-123.45", "-123.45", True),
        ("1.234", "1.23", True),  # Tests rounding
        ("$1,234.56", "1234.56", True),
        ("invalid", "123.45", False),
        (None, "123.45", False),
    ]
    
    for value, truth, expected in test_cases:
        assert evaluator.default_compare(value, truth, "number") == expected, \
            f"Failed number comparison: {value} vs {truth}"

def test_list_category_comparisons():
    evaluator = Evaluator()
    test_cases = [
        ("['a', 'b', 'c']", "['c', 'b', 'a']", True),
        ("a,b,c", "c,b,a", True),
        ("[a, b, c]", "[c, b, a]", True),
        ("['a', 'b']", "['a', 'b', 'c']", False),  # Different lengths
        ("invalid", "['a', 'b']", False),
        (None, "['a', 'b']", False),
        ("[nan]", "['']", True),
        # Test with dates
        ("['2023-01-01', '2023-01-02']", "['2023-01-02', '2023-01-01']", True),
        ("['2023-01-01T00:00:00Z', '2023-01-02 00:00:00+00:00']", "['2023-01-02', '2023-01-01']", True),
        ("['2023-01-01']", "['2023-01-02']", False),
    ]
    
    for value, truth, expected in test_cases:
        assert evaluator.default_compare(value, truth, "list[category]") == expected, \
            f"Failed list comparison: {value} vs {truth}"

def test_list_number_comparisons():
    evaluator = Evaluator()
    # Additional test cases for list[number]
    number_test_cases = [
        ("[1, 2, 3]", "[3, 2, 1]", True),
        ("1,2,3", "3,2,1", True),
        ("[1, 2]", "[1, 2, 3]", False),  # Different lengths
        ("invalid", "[1, 2]", False),
        (None, "[1, 2]", False),
        ("[1.1, 2.2]", "[2.2, 1.1]", True),
        ("[1.0, 2.0]", "[2.0, 1.0]", True),
        ("[1.0]", "[2.0]", False),
        ("[0.995, 0.4924, 0.3099, 0.281]", "[0.99, 0.49, 0.30, 0.28]", True)
    ]
    
    for value, truth, expected in number_test_cases:
        assert evaluator.default_compare(value, truth, "list[number]") == expected, \
            f"Failed list[number] comparison: {value} vs {truth}"

def test_edge_cases():
    evaluator = Evaluator()
    test_cases = [
        (None, None, "category", True),
        ("", "", "category", True),
        (" ", "", "category", True),
        (123, "123", "category", True),
        (123.45, "123.45", "number", True),
        ("[]", "[]", "list[category]", True),
    ]
    
    for value, truth, semantic, expected in test_cases:
        assert evaluator.default_compare(value, truth, semantic) == expected, \
            f"Failed edge case: {value} vs {truth} with semantic {semantic}"

def test_eval_with_semeval_dev_set():
    qa = load_qa()
    evaluator = Evaluator(qa=qa)
    
    responses = qa["answer"]
    evals = []
    for response, truth, semantic in zip(responses, qa["answer"], qa["type"]):
        truthy = evaluator.default_compare(response, truth, semantic)
        evals.append(truthy)
        if not truthy:
            print(f"First failing case:")
            print(f"Response: {response}")
            print(f"Truth: {truth}") 
            print(f"Semantic type: {semantic}")
            break
            
    score = evaluator.eval(responses)
    assert score == 1.0, "Evaluator should return 1.0 when comparing QA set against itself"

def test_eval_with_semeval_dev_set():
    qa = load_qa()
    evaluator = Evaluator(qa=qa)
    
    responses = qa["sample_answer"]
    evals = []
    for response, truth, semantic in zip(responses, qa["sample_answer"], qa["type"]):
        truthy = evaluator.default_compare(response, truth, semantic)
        evals.append(truthy)
        if not truthy:
            print(f"First failing case:")
            print(f"Response: {response}")
            print(f"Truth: {truth}") 
            print(f"Semantic type: {semantic}")
            break
            
    score = evaluator.eval(responses, lite=True)
    assert score == 1.0, "Evaluator should return 1.0 when comparing QA set against itself in lite mode"
