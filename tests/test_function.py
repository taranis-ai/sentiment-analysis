import pytest


def assert_schema(result):
    assert isinstance(result, dict), "Model output must be a dict"
    assert "label" in result, "Expected 'label' in result"
    assert "score" in result, "Expected 'score' in result"
    assert isinstance(result["score"], float), "Expected 'score' to be a float"
    assert 0.0 <= result["score"] <= 1.0, "Score should be between 0 and 1"


@pytest.mark.parametrize("model_fixture", ["roberta", "longformer"])
@pytest.mark.asyncio
async def test_sentiment_output_schema(request, model_fixture, positive_text):
    model = request.getfixturevalue(model_fixture)
    result = await model.predict(positive_text)
    assert_schema(result)


@pytest.mark.parametrize(
    "text_fixture, expected_label",
    [
        ("positive_text", "positive"),
        ("negative_text", "negative"),
        ("neutral_text", "neutral"),
    ],
)
@pytest.mark.asyncio
async def test_correct_sentiment_roberta(roberta, request, text_fixture, expected_label):
    text = request.getfixturevalue(text_fixture)
    result = await roberta.predict(text)
    assert_schema(result)
    assert result["label"].lower() == expected_label, (
        f"Expected '{expected_label}', got '{result['label']}'"
    )


@pytest.mark.asyncio
async def test_long_text_roberta_fails(long_text, roberta):
    """
    Roberta should raise an exception on very long text inputs (>1000 words).
    """
    with pytest.raises(Exception):
        await roberta.predict(long_text)


@pytest.mark.asyncio
async def test_long_text_longformer_succeeds(long_text, longformer):
    """
    Longformer should handle long texts successfully.
    """
    result = await longformer.predict(long_text)
    assert_schema(result)
    assert result["label"].lower() in {"positive", "negative", "neutral"}
