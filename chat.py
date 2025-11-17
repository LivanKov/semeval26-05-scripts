import argparse
import os
import sys
from typing import Dict, List
from huggingface_hub import InferenceClient

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen3-VL-8B-Instruct"
GPT = "openai/gpt-oss-20b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature forwarded to the API (default: 0.6).",
    )

    parser.add_argument(
        "--token",
        help="Hugging Face access token. Falls back to the HF_TOKEN environment variable.",
    )

    return parser.parse_args()


def build_client(model_id: str, token: str) -> InferenceClient:
    if not token:
        raise SystemExit(
            "Pass token as an argument using --token"
        )
    return InferenceClient(model=model_id, token=token)


def request_completion(
    client: InferenceClient,
    messages: List[Dict[str, str]],
    *,
    model_id: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> str:
    kwargs = dict(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not stream:
        response = client.chat_completion(**kwargs)
        choice = response.choices[0]
        return choice.message.content or ""

    response_text: List[str] = []
    for chunk in client.chat_completion(stream=True, **kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        response_text.append(delta.content)
        print(delta.content, end="", flush=True)
    print() 
    return "".join(response_text)


def chat_loop(args: argparse.Namespace) -> None:
    token = args.token or os.getenv("HF_TOKEN")
    client = build_client(args.model_id, token)

    messages: List[Dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    while True:
        try:
            user_msg = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession terminated.")
            break

        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_msg})
        print("Assistant> ", end="", flush=True)

        try:
            assistant_reply = request_completion(
                client,
                messages,
                model_id=args.model_id,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=not args.no_stream,
            )
        except Exception as exc:
            messages.pop()
            print(f"\n[Inference error: {exc}]", file=sys.stderr)
            continue

        if args.no_stream:
            print(assistant_reply)

        messages.append({"role": "assistant", "content": assistant_reply})


def main() -> None:
    args = parse_args()
    chat_loop(args)


if __name__ == "__main__":
    main()
