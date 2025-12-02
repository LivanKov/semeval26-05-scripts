import argparse
import json
from typing import Literal

from pathlib import Path

from huggingface_hub import InferenceClient
from pydantic import BaseModel

import time
import httpx

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
    "You are given the majority vote of the participants. You should be the closest possible to that majority vote. If it's unclear for a human, you should reflect it."
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
        default=GPT
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token. Falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
    "--run-id",
    type=str,
    default="1",
    help="Name or number to append to output file (e.g. 1, 2, 3).",
    )
    
    return parser.parse_args()

def safe_request_prediction(func, max_retries=5, **kwargs):
    """Retry wrapper to handle HF API disconnects, parsing errors, and rate limits."""

    for attempt in range(1, max_retries + 1):
        try:
            return func(**kwargs)

        except (httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.WriteError) as e:
            print(f"[WARN] Network error on attempt {attempt}: {e}")
            
        except Exception as e:
            print(f"[WARN] Unexpected error on attempt {attempt}: {e}")

        sleep_time = 2 * attempt
        print(f"Retrying in {sleep_time} seconds…")
        time.sleep(sleep_time)

    raise RuntimeError(f"Failed after {max_retries} attempts.")


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
    content = response.choices[0].message.content.strip()
    try:
        response = client.chat_completion(
            model=model_id,
            messages=messages,
            response_format=RESPONSE_FORMAT,
        )
        content = response.choices[0].message.content.strip()
        payload = json.loads(content)
    except Exception:
        # Retry once WITHOUT schema constraints
        retry_messages = messages + [
            {"role": "system", "content": "ONLY output a JSON object with keys 'id' and 'prediction'. Nothing else."}
        ]
        response = client.chat_completion(
            model=model_id,
            messages=retry_messages,
        )
        content = response.choices[0].message.content.strip()

        # Try parsing again
        try:
            payload = json.loads(content)
        except Exception:
            # FINAL FALLBACK → try to extract id & prediction manually
            import re
            id_match = re.search(r'"?id"?\s*[:=]\s*"?(?P<id>\d+)"?', content)
            pred_match = re.search(r'"?prediction"?\s*[:=]\s*"?(?P<pred>[1-5])"?', content)

            if not (id_match and pred_match):
                print(f"WARNING: Could not parse response for id {obj_nr}. Raw content:\n{content}\nSkipping.")
                return None

            payload = {
                "id": id_match.group("id"),
                "prediction": int(pred_match.group("pred")),
            }

    # Normalize shapes like {"430": 1}
    if isinstance(payload, dict) and len(payload) == 1 and list(payload.keys())[0].isdigit():
        key = list(payload.keys())[0]
        payload = {"id": key, "prediction": payload[key]}
        
    # Normalize alternative JSON formats returned by Qwen / Llama
    if isinstance(payload, dict):

        # Case 1: { "123": 5 }
        if len(payload) == 1:
            k, v = list(payload.items())[0]
            if str(k).isdigit():
                payload = {"id": str(k), "prediction": int(v)}

        # Case 2: { "object_number": "...", "likelihood": ... }
        elif ("object_number" in payload and "likelihood" in payload):
            payload = {
                "id": str(payload["object_number"]),
                "prediction": int(payload["likelihood"])
            }

        # Case 3: model returns {"id": ..., "prediction": ...} correctly
        # → do nothing

    return PredictionResponse(**payload)


def read_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def main() -> None:
    data = read_json("data/dev_majority.json")
    #print(data["501"])

    args = parse_args()
    client = InferenceClient(
        ##provider="cerebras",
        api_key=args.token
    )

    for key,entry in data.items():
        json_str = json.dumps(entry)

        prediction = safe_request_prediction(
            request_prediction,
            client=client,
            model_id=args.model_id,
            json_str=json_str,
            obj_nr=key,
        )
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        run_id = args.run_id
        model_name = args.model_id.split("/")[-1].replace("-", "_")
        out_file = out_dir / "gpt3.jsonl"
        with out_file.open("a", encoding="utf-8") as f:
            f.write(prediction.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
