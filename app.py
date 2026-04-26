from fastapi import FastAPI
from src.query import retrieve
from src.aggregator import aggregate, get_context
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure for OpenRouter
# Note: OpenRouter is compatible with the OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_llm_answer(query, context):
    prompt = f"""
    You are a high-level competitive programming coach. 
    Analyze the user's problem description based ONLY on the provided retrieved examples.

    User Query: {query}

    Reference Examples:
    {context}

    Your task:
    1. Identify the single most OPTIMAL approach/method to solve this.
    2. Determine the most efficient Time Complexity.

    Return the response in this EXACT short format:
    OPTIMAL METHOD: [The primary algorithm or data structure, e.g., Sliding Window, Dijkstra's, etc.]
    TIME COMPLEXITY: [e.g., O(n log n)]
    WHY: [One sentence explaining why this is optimal for the given constraints.]
    """

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

@app.post("/predict")
def predict(problem_description: str):
    # 1. Input Query -> Embedding -> Pinecone Retrieve
    matches = retrieve(problem_description)
    
    # 2. Get metadata context
    context = get_context(matches)
    
    # 3. LLM Final Answer (Classifier + Aggregator role combined in LLM)
    llm_answer = get_llm_answer(problem_description, context)
    
    # Also keep the structured data if needed
    topics, complexity = aggregate(matches)

    return {
        "structured_data": {
            "topics": topics,
            "time_complexity": complexity
        },
        "llm_answer": llm_answer
    }
