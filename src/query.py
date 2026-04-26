from .embed import embed_text, embed_sparse, rerank
from .pinecone_db import get_index

def retrieve(query, k=5, fetch_k=20):
    index = get_index()

    # Generate dense and sparse vectors
    dense_vec = embed_text(query)
    sparse_vec = embed_sparse(query)

    # Initial hybrid retrieval fetching a larger candidate pool
    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=fetch_k,
        include_metadata=True
    )

    matches = results["matches"]
    
    # Reranking using Cross-Encoder
    if matches:
        reranked_matches = rerank(query, matches, top_k=k)
        return reranked_matches
    
    return []