from flask import Flask, Blueprint, jsonify, request
from flask.views import MethodView

from sentiment_analysis.sentiment_analysis_multimodel import SentimentAnalysis


class AnalyzeText(MethodView):
    def __init__(self, analyzer: SentimentAnalysis) -> None:
        super().__init__()
        self.analyzer = analyzer

    def post(self):
        try:
            data = request.get_json()
            text = data.get("text", "")
            if not text.strip():
                return jsonify({"error": "Empty or unparseable text provided"}), 400
            sentiment = self.analyzer.analyze_sentiment(text)
            return jsonify({"sentiment": sentiment})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


def init(app: Flask, analyzer: SentimentAnalysis):
    app.url_map.strict_slashes = False

    analyze_bp = Blueprint("predict", __name__)
    analyze_bp.add_url_rule("/", view_func=AnalyzeText.as_view("analyze", analyzer=analyzer))
    app.register_blueprint(analyze_bp)
