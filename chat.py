import argparse
import json
from typing import Literal

from pathlib import Path

from huggingface_hub import InferenceClient
from pydantic import BaseModel

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen3-VL-8B-Instruct"
GPT = "openai/gpt-oss-20b"

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
)



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
    json_str: str,
    obj_nr: str
) -> PredictionResponse:
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

    response = client.chat_completion(
        model=model_id,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    content = response.choices[0].message.content
    payload = json.loads(content)
    return PredictionResponse(**payload)


def read_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def main() -> None:
    data = read_json("data/dev.cleaned.minified.json")
    #print(data["501"])

    args = parse_args()
    client = InferenceClient(
        provider="cerebras",
        api_key=args.token
    )

    for key,entry in data.items():
        json_str = json.dumps(entry)

        prediction = request_prediction(
            client,
            model_id=args.model_id,
            json_str=json_str,
            obj_nr=key,
        )
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "output_llama.jsonl"
        with out_file.open("a", encoding="utf-8") as f:
            f.write(prediction.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
