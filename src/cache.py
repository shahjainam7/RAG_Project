import numpy as np
from src.embed import embed_text

class SemanticCache:
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.cache = []
        self.embeddings = []

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def check(self, query):
        if not self.embeddings:
            return None

        query_embedding = np.array(embed_text(query))
        
        # Calculate cosine similarity with all cached embeddings
        similarities = [self._cosine_similarity(query_embedding, cached_emb) for cached_emb in self.embeddings]
        
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]

        if max_sim >= self.threshold:
            print(f"Cache HIT (Similarity: {max_sim:.4f})")
            return self.cache[max_idx]
            
        print(f"Cache MISS (Max Similarity: {max_sim:.4f})")
        return None

    def store(self, query, response_data):
        query_embedding = np.array(embed_text(query))
        self.embeddings.append(query_embedding)
        self.cache.append(response_data)
        
semantic_cache = SemanticCache(threshold=0.92)
