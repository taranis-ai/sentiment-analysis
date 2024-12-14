from sentiment_analysis.sentiment_analysis_multimodel import SentimentAnalysis


def test_analyze_sentiment_basic(example_text: str, analyzer: SentimentAnalysis):
    result = analyzer.analyze_sentiment(example_text)
    assert "label" in result, "Expected 'label' in result"
    assert "score" in result, "Expected 'score' in result"
    assert isinstance(result["score"], float), "Expected score to be a float"


def test_analyze_sentiment_with_test_data(news_items, analyzer: SentimentAnalysis):
    for content in news_items:
        result = analyzer.analyze_sentiment(content)
        assert "label" in result, f"Expected 'label' in result for content: {content[:30]}"
        assert "score" in result, f"Expected 'score' in result for content: {content[:30]}"
