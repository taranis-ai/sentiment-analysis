import pytest
from sentiment_analysis.roberta import Roberta
from sentiment_analysis.longformer import Longformer


@pytest.fixture(scope="session")
def roberta():
    yield Roberta()

@pytest.fixture(scope="session")
def longformer():
    yield Longformer()


@pytest.fixture(scope="session")
def positive_text():
    return (
        "Wow, what an amazing surprise! I absolutely loved it—the experience was delightful, "
        "and I couldn't stop smiling. Everything exceeded my expectations, and I feel grateful and excited."
    )

@pytest.fixture(scope="session")
def negative_text():
    return (
        "I had the worst day ever. My alarm didn’t go off, I missed the bus, and it rained all day. "
        "At work, I spilled coffee on my laptop, and to top it off, I lost my wallet. "
        "Could it get any worse?"
    )

@pytest.fixture(scope="session")
def neutral_text():
    return (
        "The meeting is scheduled for 3:00 PM in Conference Room B. "
        "Please review the Q2 report beforehand and bring your laptop. "
        "Snacks and water will be available."
    )

@pytest.fixture(scope="session")
def long_text():
    base_paragraph = (
        "This paragraph is intentionally repetitive and neutral. "
        "It describes routine observations about data processing, analysis pipelines, "
        "daily workflows, and procedural operations. "
        "There are no emotional cues, and the tone remains consistent and factual. "
        "The goal is to provide a large text block for model evaluation, ensuring that "
        "the sequence length exceeds typical short-text limits. "
        "Each repetition increases the total word count and helps simulate "
        "a long document scenario for robust testing. "
    )

    return " ".join([base_paragraph] * 20)
