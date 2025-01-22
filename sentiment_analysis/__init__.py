from flask import Flask
from sentiment_analysis import router
from sentiment_analysis.sentiment_analysis_multimodel import SentimentAnalysis


def create_app():
    app = Flask(__name__)
    app.config.from_object("sentiment_analysis.config.Config")

    with app.app_context():
        init(app)

    return app


def init(app: Flask):
    analyzer = SentimentAnalysis()
    router.init(app, analyzer)


if __name__ == "__main__":
    create_app().run()
