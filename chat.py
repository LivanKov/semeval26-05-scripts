import argparse
import json
import os
from typing import Literal

from huggingface_hub import InferenceClient
from pydantic import BaseModel

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen3-VL-8B-Instruct"
GPT = "openai/gpt-oss-20b"

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict JSON generator. "
    "Return one JSON object with keys 'id' and 'prediction'. "
    "'id' must match the integer provided by the user. "
    "'prediction' must be an integer between 1 and 5 (inclusive). "
    "Do not add extra text before or after the JSON."
)


class PredictionResponse(BaseModel):
    id: int
    prediction: Literal[1, 2, 3, 4, 5]


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "PredictionResponse",
        "schema": PredictionResponse.model_json_schema(),
        "strict": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=LLAMA
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token. Falls back to the HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def request_prediction(
    client: InferenceClient,
    *,
    model_id: str,
    system_prompt: str,
) -> PredictionResponse:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Return the JSON prediction for the following sample.\n"
                f"Sample id: {19}\n"
                "Prediction must be an integer from 1 to 5.\n"
                "Sample text:\n"
                "```\n"
                "Nothing"
                "```"
            ),
        },
    ]

    response = client.chat_completion(
        model=model_id,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    content = response.choices[0].message.content
    payload = json.loads(content)
    return PredictionResponse(**payload)


def main() -> None:
    args = parse_args()
    client = InferenceClient(
        provider="cerebras",  # or use "auto" for automatic selection
        api_key=args.token
    )
    prediction = request_prediction(
        client,
        model_id=args.model_id,
        system_prompt = DEFAULT_SYSTEM_PROMPT
    )
    print(prediction.model_dump_json())


if __name__ == "__main__":
    main()
