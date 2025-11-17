import os
import json

from huggingface_hub import InferenceClient
from pydantic import BaseModel
from typing import Literal

# Set your Hugging Face token
# export HF_TOKEN="your_token_here"

def analyze_paper_structured():
    """Complete example of structured output for research paper analysis."""
    
    # Initialize the client
    client = InferenceClient(
        provider="cerebras",  # or use "auto" for automatic selection
        api_key="",
    )
    
    # Example paper text (you can replace this with any research paper)
    paper_text = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex recurrent 
    or convolutional neural networks that include an encoder and a decoder. The best 
    performing models also connect the encoder and decoder through an attention mechanism. 
    We propose a new simple network architecture, the Transformer, based solely on 
    attention mechanisms, dispensing with recurrence and convolutions entirely. 
    Experiments on two machine translation tasks show these models to be superior 
    in quality while being more parallelizable and requiring significantly less time to train.
    
    Introduction: Recurrent neural networks, long short-term memory and gated recurrent 
    neural networks in particular, have been firmly established as state of the art approaches 
    in sequence modeling and transduction problems such as language modeling and machine translation.
    """
    
    # Define the response format (JSON Schema)
    class PaperAnalysis(BaseModel):
        title: str
        number_of_letters_in_the_title: Literal[1, 2, 3, 4, 5]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "PaperAnalysis",
            "schema": PaperAnalysis.model_json_schema(),
            "strict": True,
        },
    }
    
    # Define your messages
    messages = [
        {
            "role": "system", 
            "content": "Extract paper title and number of letters in the summary."
        },
        {
            "role": "user", 
            "content": paper_text
        }
    ]
    
    # Generate structured output
    response = client.chat_completion(
        messages=messages,
        response_format=response_format,
        model="Qwen/Qwen3-32B",
    )
    
    # The response is guaranteed to match your schema
    structured_data = response.choices[0].message.content
    
    # Parse and display results
    analysis = json.loads(structured_data)

    print(analysis)
    
    print(f"Title: {analysis['title']}")
    print(f"Abstract Summary: {analysis['number_of_letters_in_the_title']}")

if __name__ == "__main__":
    # Make sure you have set your HF_TOKEN environment variable
    analyze_paper_structured()