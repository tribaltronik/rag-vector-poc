import os
import sys
import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


SAMPLE_DOCS = [
    {
        "filename": "python_history.txt",
        "content": """Python History and Overview

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, Python has become one of the most popular programming languages in the world.

Key Features of Python:
1. Easy to Learn and Read: Python's syntax is designed to be readable and clean.
2. Versatile: Used in web development, data science, AI, automation, and more.
3. Large Community: Extensive libraries and frameworks available.
4. Cross-platform: Works on Windows, Mac, Linux, and other systems.

Python in Data Science and AI:
Python has become the dominant language for data science and machine learning due to libraries like NumPy, Pandas, TensorFlow, and PyTorch.""",
    },
    {
        "filename": "machine_learning_intro.txt",
        "content": """Introduction to Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. 

Types of Machine Learning:
1. Supervised Learning: Learning from labeled data (classification, regression)
2. Unsupervised Learning: Finding patterns in unlabeled data (clustering, dimensionality reduction)
3. Reinforcement Learning: Learning through trial and error with rewards

Popular ML Algorithms:
- Linear Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks
- Deep Learning

Applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.""",
    },
    {
        "filename": "rag_explanation.txt",
        "content": """Retrieval-Augmented Generation (RAG)

RAG is an AI framework for improving LLM responses by referencing authoritative knowledge bases outside its training data.

How RAG Works:
1. User asks a question
2. Question is converted to a vector embedding
3. Similar documents are retrieved from a vector database
4. Retrieved context is combined with the question
5. LLM generates an answer using this augmented prompt

Benefits of RAG:
- Reduces hallucinations
- Provides source citations
- Allows updating knowledge without retraining
- Works with private/proprietary documents

Popular vector databases include Qdrant, Pinecone, Weaviate, and Chroma.""",
    },
]


def upload_document(filename: str, content: str):
    import io

    files = {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")}
    try:
        response = httpx.post(f"{API_BASE_URL}/ingest/", files=files, timeout=60.0)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Indexed {data['chunks_created']} chunks from {filename}")
            return True
        else:
            print(f"✗ Failed to upload {filename}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error uploading {filename}: {str(e)}")
        return False


def main():
    print("Seeding sample documents...")
    print(f"API URL: {API_BASE_URL}")

    health_check = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
    if health_check.status_code != 200:
        print("Error: API is not available. Please start the services first.")
        sys.exit(1)

    for doc in SAMPLE_DOCS:
        upload_document(doc["filename"], doc["content"])

    print("\nSeeding complete! You can now query the documents.")


if __name__ == "__main__":
    main()
