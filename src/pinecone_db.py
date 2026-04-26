from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()



# Initialize Pinecone with the modern API
api_key=os.getenv("API_KEY")
# api_key = "pcsk_2WMiqn_pCx5f921gipdFbTHQrEUNSqPua37Zn5mCmajVWFqxU7znFZDp1ScwkcvKckg1r"
pc = Pinecone(api_key=api_key)

index_name = "leetcode-rag-v2"

def get_index():
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)