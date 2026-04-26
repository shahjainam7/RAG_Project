import json
import re

def parse_doc(doc):
    topics = re.search(r"Topics:\n(.*?)\n", doc)
    complexity = re.search(r"Expected Time Complexity:\n(.*)", doc)
    
    # Extract Title and Description to create a cleaner, denser text for embedding
    title_match = re.search(r"Problem Title: (.*?)\n", doc)
    title = title_match.group(1).strip() if title_match else ""
    
    desc_match = re.search(r"Description:\n(.*?)\n\n(?:Topics:|Constraints:|Hints:)", doc, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else doc
    
    content_to_embed = f"{title}: {description}" if title else description

    return {
        "text": doc,
        "content_to_embed": content_to_embed,
        "topics": topics.group(1).split(", ") if topics else [],
        "complexity": complexity.group(1).strip() if complexity and complexity.group(1) else "Unknown"
    }

def load_data(path):
    with open(path) as f:
        data = json.load(f)

    return [parse_doc(item["document"]) for item in data]


    
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone_text.sparse import BM25Encoder
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize BM25 Encoder
bm25 = BM25Encoder.default()
bm25_params_path = "data/bm25_params.json"
if os.path.exists(bm25_params_path):
    bm25.load(bm25_params_path)

def embed_text(text):
    return model.encode(text).tolist()

def embed_sparse(text):
    return bm25.encode_queries(text)

def rerank(query, documents, top_k=5):
    # Create pairs of (query, document_text)
    pairs = [[query, doc.get("metadata", {}).get("text", "")] for doc in documents]
    
    # Get scores from the cross-encoder
    scores = reranker_model.predict(pairs)
    
    # Sort documents by their score in descending order
    scored_docs = zip(scores, documents)
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    # Return the top_k documents
    return [doc for score, doc in sorted_docs[:top_k]]