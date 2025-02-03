import os
import pytest
import json
from sentiment_analysis.roberta_sentiment import RobertaSentiment


@pytest.fixture(scope="session")
def news_items():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    story_json = os.path.join(dir_path, "story_list.json")
    with open(story_json) as f:
        data = json.load(f)
    yield [item["content"] for cluster in data for item in cluster["news_items"] if "content" in item]


@pytest.fixture(scope="session")
def analyzer():
    yield RobertaSentiment()


@pytest.fixture(scope="session")
def example_text():
    yield "I had the worst day ever. My alarm didn’t go off, I missed the bus, and it rained all day. At work, I spilled coffee on my laptop, and to top it off, I lost my wallet. Could it get any worse?."


@pytest.fixture(scope="session")
def short_text():
    yield "Wow, what an amazing surprise!"


@pytest.fixture(scope="session")
def long_text():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    long_text = os.path.join(dir_path, "long_text.txt")
    with open(long_text) as f:
        yield f.read()
