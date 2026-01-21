import os
import base64
import torch
from openai import OpenAI
from typing import List
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv

load_dotenv()

class Qwen3VLEmbedder:
    """
    A helper class to load and run the Qwen3-VL-Embedding-2B model locally.
    This class is called by the get_image_and_text_embedding function.
    """
    def __init__(self, model_name_or_path: str):
        print(f"Loading embedding model: {model_name_or_path}...")
        # We use bfloat16 to save memory (requires ~5GB VRAM or RAM)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

    def process(self, context: List[dict]):
        # The processor handles mixed inputs (text or image paths)
        inputs = self.processor(text=[c.get('text', '') for c in context], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # We take the mean of the last hidden states to create a single vector
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().float().tolist()
        return embeddings

def image_to_base64(image_path: str):
    """Utility to convert local image to base64 for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_and_text_embedding(context: List[dict]):
    """
    As described in the article:
    This function takes a mixed context of images and text and 
    converts it into a unified embedding representation.
    """
    # Specify the model path
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"

    # Initialize the Qwen3VLEmbedder model
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)

    # Process the context to get embeddings
    image_and_text_embeddings = model.process(context)

    return image_and_text_embeddings

def get_image_description(image_path: str):
    """
    As described in the article:
    Sends image to RunPod vLLM to get a text description, 
    then converts that description into a text embedding.
    """
    base64_image = image_to_base64(image_path)

    # Modify OpenAI's API key and API base to use vLLM's API server on RunPod
    openai_api_key = os.getenv("RUNPOD_API_KEY", "sk-EMPTY")
    openai_api_base = os.getenv("RUNPOD_API_BASE")
    
    client = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image. always respond as a paragraph. no bullet points"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
            ],
        }],
    )

    image_description = response.choices[0].message.content
    print(f"\n[Generated Description]: {image_description}\n")

    # Pass description into the multimodal embedding function
    text_embeddings = get_image_and_text_embedding([{
        'text': image_description,
    }])

    return image_description, text_embeddings[0]