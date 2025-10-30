import os
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
       return (
                "The weather today was absolutely stunning."
                "I woke up to a sky painted in vibrant hues of orange and pink as the sun rose, and the air was crisp and refreshing."
                "My morning jog through the park was invigorating, with birds singing cheerfully and a gentle breeze rustling the autumn leaves." 
                "Later, I treated myself to a cup of freshly brewed coffee and a warm croissant at a cozy café. The barista was friendly, and the atmosphere was relaxing."
                "It's been a perfect day so far, filled with simple joys that remind me how beautiful life can be when we pause to appreciate it."
        )