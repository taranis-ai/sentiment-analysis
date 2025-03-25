# Taranis AI Sentiment Analysis

This project integrates sentiment analysis into [Taranis AI](https://github.com/taranis-ai/taranis-ai), allowing for the classification of news items as **positive**, **negative**, or **neutral** using transformer models. The API intelligently chooses the appropriate model based on the text length, utilizing **XLM-RoBERTa** for shorter texts and **Longformer** for longer texts.

## Features

- **Multi-language Support** with XLM-RoBERTa and Longformer.
- **Pre-trained Models** for high-accuracy sentiment classification.
- **Flask API** for external calls to the sentiment analysis.

## Setup

### Prerequisites

- Python 3.10+

### Installation

To install dependencies:

```bash
uv venv
uv sync
```

### Running the API

To start the Flask API:

```bash
flask run
# or
granian app

# or
docker run -p 5000:5000 ghcr.io/taranis-ai/taranis-sentiment-bot:latest
```

### Example API Call

To test the API with a POST request, use curl:

```bash
curl -X POST http://127.0.0.1:5000/ \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an example sentence to analyze sentiment."}'
```

### Example Response

```json
{
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.94
  }
}
```

## License

EUROPEAN UNION PUBLIC LICENCE v. 1.2
