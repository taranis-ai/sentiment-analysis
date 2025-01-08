from flask import Flask
from sentiment_analysis import router
from sentiment_analysis.sentiment_analysis_multimodel import SentimentAnalyzer

def create_app():
    app = Flask(__name__)
    app.config.from_object("sentiment_analysis.config.Config")

    with app.app_context():
        init_app(app)

    return app

def init_app(app: Flask):
    analyzer = SentimentAnalyzer()
    router.init(app, analyzer)
