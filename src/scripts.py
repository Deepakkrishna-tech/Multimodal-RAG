import os
import uuid
import sys
from datetime import datetime
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client import models
from dotenv import load_dotenv

# Import the logic from your other file
from image_and_text_embedder import get_image_description, get_image_and_text_embedding

load_dotenv()

class QdrantUtil:
    def __init__(
            self,
            url: str = "http://localhost:6333",
            collection_name: str = "multimodal_rag"
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(
            self,
            image_vector_size: int,
            text_vector_size: int
    ):
        print(f"Initializing collection: {self.collection_name}")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors={
                "image_embedding": models.VectorParams(
                    size=image_vector_size,
                    distance=models.Distance.COSINE
                ),
                "text_embedding": models.VectorParams(
                    size=text_vector_size,
                    distance=models.Distance.COSINE
                ),
            },
        )

    def upsert_item(
            self,
            image_embedding: List[float],
            text_embedding: List[float],
            image_name: str,
            image_desc: str,
            extra_metadata: Dict = None
    ):
        payload = {
            "image_name": image_name,
            "text": image_desc,
            "date": datetime.utcnow().isoformat(),
        }

        if extra_metadata:
            payload.update(extra_metadata)

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
            },
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def multivector_search(
            self,
            text_query_vector: List[float],
            image_query_vector: List[float],
            limit: int = 10
    ):
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=models.Prefetch(
                query=text_query_vector,
                using="text_embedding",
            ),
            query=image_query_vector,
            using="image_embedding",
            limit=limit,
            with_payload=True,
        )
        return results

# --- MAIN EXECUTION LOOP ---
# This part uses the class above to run the actual RAG pipeline
if __name__ == "__main__":
    # 1. Initialize Qdrant
    qdrant = QdrantUtil(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    # Qwen3-VL-Embedding-2B produces 1536-dimensional vectors
    qdrant.create_collection(image_vector_size=1536, text_vector_size=1536)
    
    # 2. Define the image to process
    img_path = "img.png" 
    
    if os.path.exists(img_path):
        print(f"Step 1: Analyzing {img_path}...")
        # Returns the text description AND the embedding for that text
        image_description, text_emb = get_image_description(img_path)
        
        print("Step 2: Generating raw image embedding...")
        # Get the embedding for the visual content itself
        image_emb = get_image_and_text_embedding([{'image': img_path}])[0]
        
        print("Step 3: Saving to Qdrant...")
        qdrant.upsert_item(
            image_embedding=image_emb,
            text_embedding=text_emb,
            image_name="test_image_001",
            image_desc=image_description
        )
        
        print("\n--- DONE ---")
        print(f"Stored image with description: {image_description[:50]}...")
    else:
        print(f"Error: {img_path} not found in root directory.")