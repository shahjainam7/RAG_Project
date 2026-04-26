from src.embed import embed_text, bm25, bm25_params_path
from src.pinecone_db import get_index
from src.embed import load_data
import time

# Load data from the problems.json file
data = load_data("data/problems.json")
print(f"Loaded {len(data)} problems from JSON.")

# Fit BM25 on the corpus
print("Fitting BM25 on the corpus...")
corpus = [item.get("content_to_embed", item["text"]) for item in data]
bm25.fit(corpus)
bm25.dump(bm25_params_path)
print("BM25 fitted and parameters saved.")

# Get the Pinecone index
index = get_index()

vectors = []

print("Generating dense and sparse embeddings...")
for i, item in enumerate(data):
    content = item.get("content_to_embed", item["text"])
    
    # Use item index as ID for simplicity
    vectors.append({
        "id": str(i),
        "values": embed_text(content),
        "sparse_values": bm25.encode_documents(content),
        "metadata": {
            "topics": item["topics"],
            "complexity": item["complexity"],
            "text": item["text"]
        }
    })

# Upload in batches (IMPORTANT for large datasets)
batch_size = 100
print(f"Uploading {len(vectors)} vectors in batches of {batch_size}...")

for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch)
    print(f"Uploaded batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

print("✅ Data successfully uploaded to Pinecone")
