"""
chat.py
--------
This script evaluates the probability that a given "judged meaning" for a
homonym is correct in a given sentence/context. For each item in
`data/dev_majority.json`, it sends a chat-style completion to a chosen
model via the Hugging Face Inference API and writes the model's prediction
into `output/gpt.jsonl` in JSONL format (one object per line).

Behavior:
- Builds a short conversation with a system and a user message. The user
    message contains the JSON entry to evaluate.
- Asks the model to respond in a JSON schema format defined by
    the `PredictionResponse` Pydantic model.
- Parses the response and normalizes it to a canonical `id` and
    `prediction` (1-5). If parsing fails, the function falls back to simple
    regex extraction.

This file only contains the logic to send the request and parse the
responses; it does not perform any local model inference.
"""

import argparse
import json
from typing import Literal

from pathlib import Path

from huggingface_hub import InferenceClient
from pydantic import BaseModel

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen3-VL-8B-Instruct"
GPT = "openai/gpt-oss-20b"
LLAMA_2 = "meta-llama/Llama-3.2-3B-Instruct"
WSD_FINETUNED = "afafasdfasfa/wsd-llama3.2-lora"

DEFAULT_SYSTEM_PROMPT = (
    "Are an expert in word-sense disambiguation, able to precisely interpret the meaning of words within a sentence, regardless of the context surrounding them"
    "You will be given a json object containing multiple strings."
    "'homonym' is a word with a potentially ambigious meaning depending on its context."
    "'judged meaning' tells you how the word is interpreted within this example."
    "'precontext' provides a general context in which the homonym will be used."
    "'sentence' provides the actual sentence within that context, that contains the homonym"
    "'ending' provides an ending to the situation originally described within 'precontext'"
    "'sample_id' references the id of the sample and is not important for the evaluation."
    "'example_sentence' provides a general example of a sentence in which this word might be used and can also be used for evaluation"
    "Your task is to grade the probability of 'judged_meaning' actually being the right one within the given context. You have to grade it on a scale of 1 (very unlikely to be true) to 5(very likely to be true)."
    "You are given the majority vote of the participants. You should be the closest possible to that majority vote. If it's unclear for a human, you should reflect it."
)

# Pydantic model used to validate/represent responses from the model.
# `PredictionResponse` expects an `id` string and a `prediction` integer
# literal between 1 and 5.



class PredictionResponse(BaseModel):
    id: str
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
    """Parse CLI arguments.

    --model-id: model identifier to send to HF Inference API (default: GPT)
    --token: HF API token, if not provided the client will use the HF_TOKEN
             environment variable according to the SDK's behavior.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=LLAMA_2,
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
    json_str: str,
    obj_nr: str
) -> PredictionResponse:
    """Send a chat request to the provided `client` and return a
    validated `PredictionResponse`.

    The function attempts to parse strict JSON first, then normalizes a
    few alternative JSON shapes (e.g. `{"501": 3}` or keys like
    `object_number`/`likelihood`). If JSON parsing fails, it falls back to
    regex-based extraction of `id` and `prediction` from the raw content.
    """

    # Prepare conversation messages: system-level instructions and user
    # content containing the JSON object to evaluate.
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Use the already provided information and the following json object, in order to evaluate the probability of the judged meaning of the word being correct within the given scenario. "
                       f"Grade the likelihood of it being correct on a scale from 1 to 5. "
                       f"Provide the response as a json containing two elements: the number/id of the provided object returned as a string and the likelihood as a number between 1 and 5. "
                       f"Strictly adhere to the given rules and the set structured output. "
                       f"Object number/id: {obj_nr} "
                       f"JSON object: {json_str}",
        },
    ]

    # Call to the Hugging Face Inference API to obtain a JSON Schema
    # formatted response. `RESPONSE_FORMAT` instructs the provider to
    # follow the `PredictionResponse` schema when possible.
    response = client.chat_completion(
        model=model_id,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    content = response.choices[0].message.content.strip()
    try:
        payload = json.loads(content)
        
        # If the model returns a dictionary with a single numeric key,
        # interpret it as {id: prediction}.
        if isinstance(payload, dict) and len(payload) == 1 and list(payload.keys())[0].isdigit():
            only_key = list(payload.keys())[0]
            payload = {"id": only_key, "prediction": payload[only_key]}
        # Check alternative keys that some models may use
        elif isinstance(payload, dict):
            if "object_number" in payload and "likelihood" in payload:
                payload = {
                    "id": str(payload["object_number"]),
                    "prediction": int(payload["likelihood"])
                }
        else:
            pass
    except Exception:
        import re
        # Fall back to lax regex extraction if the response is not valid
        # JSON or is otherwise malformed. This will attempt to find patterns
        # like 'id: 501' and 'prediction: 3' in the returned content.
        id_match = re.search(r'"?id"?\s*[:=]\s*"?(?P<id>\d+)"?', content)
        pred_match = re.search(r'"?prediction"?\s*[:=]\s*"?(?P<pred>[1-5])"?', content)
        
        if not (id_match and pred_match):
            raise ValueError(f"Could not parse response: {content}")
        payload = {
        "id": id_match.group("id"),
        "prediction": int(pred_match.group("pred"))
    }
    return PredictionResponse(**payload)


def read_json(path: str | Path):
    """Read and parse a JSON file from `path`.

    Accepts either a string or a `Path` object and returns the parsed JSON
    object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def main() -> None:
    """Main entrypoint.

    Reads the data, configures the HF client, iterates over the dataset,
    requests predictions for each entry, and appends them to
    `output/gpt.jsonl`.
    """

    # Load the evaluation dataset
    data = read_json("data/dev_majority.json")
    #print(data["501"])

    args = parse_args()
    client = InferenceClient(
        provider="novita",
        api_key=args.token
    )

    for key,entry in data.items():
        # Convert the entry back to a JSON string to include within the
        # user message that will be sent to the model.
        json_str = json.dumps(entry)

        prediction = request_prediction(
            client,
            model_id=args.model_id,
            json_str=json_str,
            obj_nr=key,
        )
        # Ensure the output directory exists and append predictions to
        # the JSONL file. Each prediction is a serialized `PredictionResponse`.
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "gpt.jsonl"
        with out_file.open("a", encoding="utf-8") as f:
            f.write(prediction.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
