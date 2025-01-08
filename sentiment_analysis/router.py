from flask import Flask, Blueprint, jsonify, request
from flask.views import MethodView

from sentiment_analysis.sentiment_analysis_multimodel import SentimentAnalyzer


class SentimentView(MethodView):
    def __init__(self, analyzer: SentimentAnalyzer) -> None:
        super().__init__()
        self.analyzer = analyzer
    def post(self):
        data = request.get_json() or {}
        text = data.get("text", "")

        result = self.analyzer.analyze_sentiment(text)

        return jsonify({"sentiment": result})

def init(app: Flask, analyzer: SentimentAnalyzer):
    app.url_map.strict_slashes = False
    sentiment_bp = Blueprint("sentiment", __name__)

    sentiment_bp.add_url_rule(
        "/",
        view_func=SentimentView.as_view("sentiment", analyzer=analyzer),
        methods=["POST"]
    )
    app.register_blueprint(sentiment_bp, url_prefix="/sentiment")