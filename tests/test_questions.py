import pytest

from chainscope.api_utils.api_selector import APIPreferences
from chainscope.questions import gen_qs
from chainscope.typing import SamplingParams


@pytest.mark.parametrize("prop_id", ["aircraft-speeds", "element-densities"])
def test_gen_qs(prop_id):
    # Generate questions
    n = 5
    max_comparisons = 3
    datasets = gen_qs(
        prop_id=prop_id,
        n=n,
        max_comparisons=max_comparisons,
        min_popularity=None,
        max_popularity=None,
        min_fraction_value_diff=None,
        max_fraction_value_diff=None,
        dataset_suffix=None,
        # "no" skips the LLM-based ambiguity evaluation, keeping this test offline
        remove_ambiguous="no",
        non_overlapping_rag_values=False,
        min_rag_values_count=None,
        api_preferences=APIPreferences.from_args(),
        evaluator_model_id="",
        evaluator_sampling_params=SamplingParams(
            temperature=0.0, top_p=0.9, max_new_tokens=1
        ),
    )

    # For each comparison type and answer, verify the generated dataset
    for comparison in ["gt", "lt"]:
        for answer in ["YES", "NO"]:
            dataset = datasets[(comparison, answer)]  # type: ignore

            # Check dataset properties
            assert len(dataset.question_by_qid) == n
            assert dataset.params.prop_id == prop_id
            assert dataset.params.comparison == comparison
            assert dataset.params.answer == answer
            assert dataset.params.max_comparisons == max_comparisons

            # Check that values are correctly ordered based on comparison type and answer
            for question in dataset.question_by_qid.values():
                if (answer == "YES" and comparison == "gt") or (
                    answer == "NO" and comparison == "lt"
                ):
                    assert question.x_value > question.y_value
                else:
                    assert question.x_value < question.y_value

                # Check that question strings are properly formatted
                assert question.x_name in question.q_str
                assert question.y_name in question.q_str
                assert question.x_name in question.q_str_open_ended
                assert question.y_name in question.q_str_open_ended
