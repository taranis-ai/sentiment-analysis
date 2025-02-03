from sentiment_analysis.config import Config
from sentiment_analysis.predictor import Predictor


class PredictorFactory:
    """
    Factory class that dynamically instantiates and returns the correct Predictor
    based on the configuration. This approach ensures that only the configured model
    is loaded at startup.
    """

    def __new__(cls, *args, **kwargs) -> Predictor:
        if Config.MODEL == "roberta":
            from sentiment_analysis.roberta_sentiment import RobertaSentiment

            return RobertaSentiment(*args, **kwargs)
        elif Config.MODEL == "longformer":
            from sentiment_analysis.longformer_sentiment import LongformerSentiment

            return LongformerSentiment(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported NER model: {Config.MODEL}")
