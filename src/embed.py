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


    
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text).tolist()