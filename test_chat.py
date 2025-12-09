import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="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"
)

completion = client.chat.completions.create(
    model="chuanli11/Llama-3.2-3B-Instruct-uncensored",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)