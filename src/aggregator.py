from collections import Counter

def aggregate(matches):
    topics = []
    complexities = []

    for m in matches:
        metadata = m.get("metadata", {})
        topics.extend(metadata.get("topics", []))
        comp = metadata.get("complexity")
        # Filter out 'Unknown' to get a more accurate expected complexity
        if comp and comp != "Unknown":
            complexities.append(comp)

    # Only return topics that appear in at least 40% of the retrieved results to ensure relevance
    counts = Counter(topics)
    threshold = len(matches) * 0.4
    top_topics = [t for t, count in counts.items() if count >= threshold]
    
    # Fallback to the single most common if none meet the threshold
    if not top_topics and topics:
        top_topics = [counts.most_common(1)[0][0]]
    
    if not complexities:
        top_complexity = "Unknown"
    else:
        # Get the most common complexity that is not None/Empty
        common_complexities = Counter(complexities).most_common(1)
        top_complexity = common_complexities[0][0] if common_complexities else "Unknown"

    return top_topics, top_complexity

def get_context(matches):
    context = ""
    for i, m in enumerate(matches):
        metadata = m.get("metadata", {})
        context += f"Problem {i+1}:\n{metadata.get('text', '')}\n\n"
    return context.strip()