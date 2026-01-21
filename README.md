# Multimodal RAG with Qwen3-VL & Qdrant

This project implements a **production-grade Multimodal Retrieval-Augmented Generation (RAG)** system using the **Qwen3-VL ecosystem** and **Qdrant**. It moves beyond text-only RAG by treating images as a primary source of knowledge, utilizing **multi-vector storage** to bridge visual and textual semantics for richer, more accurate retrieval and generation.

---

## 🚀 Architecture

The pipeline follows a clean, decoupled flow:

1. **Visual Understanding**:  
   Raw images are processed by **Qwen3-VL-8B-Instruct** (hosted via **vLLM on RunPod**) to extract deep semantic interpretations and generate rich textual descriptions.

2. **Joint Embedding**:  
   Both the original image and its generated description are encoded into **1536-dimensional vectors** using the **Qwen3-VL-Embedding-2B** model—capturing complementary modalities in a shared embedding space.

3. **Multi-Vector Storage**:  
   Each knowledge entry is stored in **Qdrant** as a single point with **dual named vectors**:  
   - `image_embedding`  
   - `text_embedding`  

4. **Prefetch + Re-Rank Retrieval**:  
   Search uses a **two-stage process**:  
   - **Stage 1 (Prefetch)**: Text query retrieves top-K candidates using `text_embedding`.  
   - **Stage 2 (Re-Rank)**: Candidates are re-scored using `image_embedding` for visual alignment, ensuring high-fidelity multimodal relevance.

---

## 🛠️ Tech Stack

| Component               | Technology                                      |
|------------------------|-------------------------------------------------|
| **Vision-Language Model** | Qwen3-VL-8B-Instruct (via vLLM on RunPod)      |
| **Embedding Model**       | Qwen3-VL-Embedding-2B (local, Hugging Face Transformers) |
| **Vector Database**       | Qdrant (Docker)                                 |
| **Orchestration**         | Python + OpenAI-compatible SDK                  |
| **Deployment**            | RunPod (for inference), Local Docker (for Qdrant) |

---

## 📁 Project Structure

```text
.
├── src/
│   ├── image_and_text_embedder.py  # Handles VL understanding + dual embedding
│   └── scripts.py                  # Qdrant setup, ingestion, and multimodal search
├── .env                            # Environment variables (API keys, endpoints)
├── requirements.txt                # Python dependencies
├── img.png                         # Sample test image
└── README.md                       # You're here!
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/qwen-multimodal-rag.git
cd qwen-multimodal-rag
```

### 2. Configure Environment Variables
Create a `.env` file:
```env
RUNPOD_API_BASE=https://your-pod-id-8000.proxy.runpod.net/v1
RUNPOD_API_KEY=your_runpod_api_key
QDRANT_URL=http://localhost:6333
```

> 💡 **Note**: Replace `your-pod-id` and `your_runpod_api_key` with your actual RunPod deployment details.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Qdrant Locally
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### 5. Ingest Data & Run Demo
```bash
python3 src/scripts.py
```
This script will:
- Load `img.png`
- Generate a description using Qwen3-VL-8B
- Embed both image and text
- Store them as a multi-vector point in Qdrant
- Perform a sample multimodal query

---

## 🔍 Applications & Use Cases

### **Use Case 1: Enterprise Document Intelligence**

A strong production use case for this architecture is **enterprise document intelligence**, where images are first-class citizens. Think of domains like **insurance claims**, **medical records**, **aerospace maintenance logs**, or **compliance audits**. In these systems, a large portion of knowledge lives inside **scanned documents, photos, diagrams, X-rays, handwritten notes, or equipment images**.

This architecture allows every image to be:
- Ingested and semantically understood by a vision-language model,
- Embedded into a unified multimodal vector space,
- Stored with rich metadata in Qdrant.

When a user asks:  
> *“Show similar past claims with visible structural damage”*  
or  
> *“Find reports that contain wiring issues like this image,”*  

the system retrieves relevant historical cases using **both textual intent and visual similarity**. The language model then reasons over the retrieved context to generate **grounded explanations or summaries**.

✅ **Production Impact**:  
- Reduces manual review time  
- Improves decision consistency  
- Enables search over visual evidence previously inaccessible to text-only RAG

---

### **Use Case 2: Multimodal Knowledge Assistants for Engineering & Manufacturing**

In industries like **aerospace**, **automotive**, and **industrial manufacturing**, engineers constantly work with **schematics, component images, failure photos, and inspection visuals** alongside technical documentation.

With this architecture:
- Images from inspections or design reviews are ingested and embedded along with their AI-generated descriptions.
- Engineers can query using natural language:  
  > *“Find past issues similar to this crack pattern”*  
  or  
  > *“Retrieve components visually similar to this assembly with known failures.”*

The **multi-vector search** uses one modality (e.g., text) to narrow candidates and another (e.g., image) to precisely rank results—critical at enterprise scale. The retrieved context is passed to the language model to:
- Explain root causes  
- Suggest corrective actions  
- Summarize historical failure patterns

✅ **Production Impact**:  
- Faster troubleshooting  
- Better reuse of tacit visual knowledge  
- A single assistant that understands **both what engineers see and what they ask**

---

## 🧪 Extending the System

- **Add more images**: Modify `scripts.py` to loop over a directory.
- **Batch ingestion**: Extend `image_and_text_embedder.py` for parallel processing.
- **Hybrid scoring**: Tune fusion weights between text and image similarity scores.
- **Deploy Qdrant remotely**: Update `QDRANT_URL` in `.env`.

---

## 🙌 Acknowledgements

- [Qwen Team](https://qwenlm.github.io/) for the powerful Qwen3-VL models  
- [Qdrant](https://qdrant.tech/) for efficient multi-vector support  
- [RunPod](https://runpod.io/) for seamless GPU inference hosting  

---

> ✨ **Multimodal RAG isn’t just about adding images—it’s about unifying perception and language.**
