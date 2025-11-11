import pytest

def assert_schema(result):
    assert isinstance(result, dict), "Model output must be a dict"
    assert "label" in result, "Expected 'label' in result"
    assert "score" in result, "Expected 'score' in result"
    assert isinstance(result["score"], float), "Expected 'score' to be a float"
    assert 0.0 <= result["score"] <= 1.0, "Score should be between 0 and 1"

def test_sentiment_output_schema(roberta, positive_text):
    result = roberta.predict(positive_text)
    assert_schema(result)


@pytest.mark.parametrize(
    "text_fixture, expected_label",
    [
        ("positive_text", "positive"),
        ("negative_text", "negative"),
        ("neutral_text", "neutral"),
    ],
)
def test_correct_sentiment_roberta(roberta, request, text_fixture, expected_label):
    text = request.getfixturevalue(text_fixture)
    result = roberta.predict(text)
    assert_schema(result)
    assert result["label"].lower() == expected_label, (
        f"Expected '{expected_label}', got '{result['label']}'"
    )

def test_correct_sentiment_roberta_long_text(long_text, roberta):
    result = roberta.predict(long_text)
    assert_schema(result)
    assert result["label"].lower() == "neutral"
