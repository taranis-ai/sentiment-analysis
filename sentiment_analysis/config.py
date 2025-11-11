from typing import Literal
from taranis_base_bot.config import CommonSettings


class Settings(CommonSettings):
    MODEL: Literal["roberta"] = "roberta"
    PACKAGE_NAME: str = "sentiment_analysis"
    HF_MODEL_INFO: bool = True
    PAYLOAD_SCHEMA: dict[str, dict] = {"text": {"type": "str", "required": True}}



Config = Settings()
