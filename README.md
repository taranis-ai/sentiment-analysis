# Taranis AI sentiment_analysis Bot

This project integrates sentiment analysis into [Taranis AI](https://github.com/taranis-ai/taranis-ai), allowing for the classification of news items as **positive**, **negative**, or **neutral** using transformer models. 

Available models:
- roberta (https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) - *Default*
- longformer (https://huggingface.co/allenai/longformer-base-4096)


## Pre-requisites

- uv - https://docs.astral.sh/uv/getting-started/installation/
- docker (for building container) - https://docs.docker.com/engine/

Create a python venv and install the necessary packages for the bot to run.

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras --dev
```

## Usage

You can run your bot locally with

```bash
quart run --port 5500
# or
granian --interface asgi app --port 5500
```

You can set configs either via a `.env` file or by setting environment variables directly.
available configs are in the `config.py`
You can select the model via the `MODEL` env var. E.g.:

```bash
MODEL=roberta flask run
```


## Docker

You can also create a Docker image out of this bot. For this, you first need to build the image with the build_container.sh

You can specify which model the image should be built with the MODEL environment variable. If you omit it, the image will be built with the default model.

```bash
MODEL=<model_name> ./build_container.sh
```

then you can run it with:

```bash
docker run -p 5500:8000 <image-name>:<tag>
```

If you encounter errors, make sure that port 5500 is not in use by another application.


## Test the bot

Once the bot is running, you can send test data to it on which it runs its inference method:

```bash
> curl -X POST http://127.0.0.1:5500 -H "Content-Type: application/json"  -d '{"text": "This product is really really nice"}'
> {"label":"Positive","score":0.9088153839111328}
```

```bash
> curl -X POST http://127.0.0.1:5500 -H "Content-Type: application/json"  -d '{"text": "This product is not the best, but you can use it"}'
> {"label":"Neutral","score":0.361157089471817}
```

You can also set up authorization via the `API_KEY` env var. In this case, you need to send the API_KEY as an Authorization header:

```bash
> curl -X POST http://127.0.0.1:5500/  -H "Authorization: Bearer api_key" -H "Content-Type: application/json"   -d '{"text": "This is an example for NER, about the ACME Corporation which is producing Dy#namite in Acme City, which is in Australia and run by Mr. Wile E. Coyote."}'
> {"ACME Corporation":"Organization","Acme City":"Location","Australia":"Location","Dynamite":"Product","NER":"Organization","Wile E. Coyote":"Person"}
```

```bash
> curl -X POST http://127.0.0.1:5000 -H "Content-Type: application/json" -H "Authorization: Bearer api_key"  -d '{"text": "This product is really dumb. You should not buy it"}'
> {"label":"Negative","score":0.9581084251403809}
```

## Development

If you want to contribute to the development of this bot, make sure you set up your pre-commit hooks correctly:

- Install pre-commit (https://pre-commit.com/)
- Setup hooks: `> pre-commit install`


## License

EUROPEAN UNION PUBLIC LICENCE v. 1.2
