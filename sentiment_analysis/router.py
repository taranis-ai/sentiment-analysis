from flask import Flask, Blueprint, jsonify, request
from flask.views import MethodView

from sentiment_analysis.predictor import Predictor
from sentiment_analysis.predictor_factory import PredictorFactory
from sentiment_analysis.decorators import api_key_required


class AnalyzeText(MethodView):
    def __init__(self, processor: Predictor) -> None:
        super().__init__()
        self.processor = processor

    @api_key_required
    def post(self):
        try:
            data = request.get_json()
            text = data.get("text", "")
            if not text.strip():
                return jsonify({"error": "Empty or unparseable text provided"}), 400
            sentiment = self.processor.predict(text)
            return jsonify({"sentiment": sentiment})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


class HealthCheck(MethodView):
    def get(self):
        return jsonify({"status": "ok"})


class ModelInfo(MethodView):
    def __init__(self, processor: Predictor):
        super().__init__()
        self.processor = processor

    def get(self):
        return jsonify(self.processor.modelinfo)


def init(app: Flask):
    analyzer = PredictorFactory()
    app.url_map.strict_slashes = False

    analyze_bp = Blueprint("predict", __name__)
    analyze_bp.add_url_rule("/", view_func=AnalyzeText.as_view("analyze", processor=analyzer))
    analyze_bp.add_url_rule("/health", view_func=HealthCheck.as_view("health"))
    analyze_bp.add_url_rule("/modelinfo", view_func=ModelInfo.as_view("modelinfo", processor=analyzer))
    app.register_blueprint(analyze_bp)
