from .embed import embed_text
from .pinecone_db import get_index

def retrieve(query, k=5):
    index = get_index()

    results = index.query(
        vector=embed_text(query),
        top_k=k,
        include_metadata=True
    )

    return results["matches"]