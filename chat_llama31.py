#!/usr/bin/env python3
"""Interactive CLI for chatting with Meta-Llama-3.1-8B-Instruct via Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

try:
    from huggingface_hub import InferenceClient
except ImportError as exc:  # pragma: no cover - makes the missing dep obvious
    raise SystemExit(
        "huggingface-hub is required. Install it with `pip install huggingface-hub`."
    ) from exc


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chat with Meta-Llama-3.1-8B-Instruct using the Hugging Face Inference API. "
            "Obtain an access token from https://huggingface.co/settings/tokens and "
            "either export it as HF_TOKEN or pass it via --token."
        )
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model repo on Hugging Face to use (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise and helpful assistant.",
        help="System prompt injected at the beginning of the chat.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature forwarded to the API (default: 0.6).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per reply (default: 512).",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token. Falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses (entire reply prints once ready).",
    )
    return parser.parse_args()


def build_client(model_id: str, token: str) -> InferenceClient:
    if not token:
        raise SystemExit(
            "A Hugging Face access token is required. "
            "Set the HF_TOKEN environment variable or pass --token."
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
    print()  # newline after the streamed reply
    return "".join(response_text)


def chat_loop(args: argparse.Namespace) -> None:
    token = args.token or os.getenv("HF_TOKEN")
    client = build_client(args.model_id, token)

    messages: List[Dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print(
        "Chat session started. Type 'exit' or 'quit' (or press Ctrl-D) to leave.\n",
        flush=True,
    )

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
        except Exception as exc:  # pragma: no cover - surfacing inference issues
            messages.pop()  # drop the failed user turn
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
