
import re



def classify_question(question: str) -> dict:
    """
    Classify a question by its apparent complexity
    and select a processing route.
    """

    question_lower = question.lower().strip()

    # Route 1: Complex questions — check these first
    complex_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifference between\b",
        r"\badvantages and disadvantages\b",
        r"\bpros and cons\b",
        r"\banalyze\b",
        r"\banalyse\b",
        r"\bexplain how\b",
        r"\bin detail\b",
        r"\bwhy\b",
        r"\bhow do i\b",
        r"\bhow to\b",
        r"\bhow can i\b",
        r"\bsteps to\b",
        r"\btutorial\b",
    ]

    if any(re.search(pattern, question_lower) for pattern in complex_patterns):
        return {
            "complexity": "complex",
            "route": "advanced",
            "reason": "The question appears to require deeper explanation or analysis.",
        }

    # Route 2: Simple questions
    simple_patterns = [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bdefine\b",
        r"\bwho is\b",
        r"\bwho are\b",
        r"\bmeaning of\b",
    ]

    if any(re.search(pattern, question_lower) for pattern in simple_patterns):
        return {
            "complexity": "simple",
            "route": "lightweight",
            "reason": "The question appears to request a basic definition or fact.",
        }

    # Route 3: Default
    return {
        "complexity": "moderate",
        "route": "standard",
        "reason": "The question does not match a simple or complex rule.",
    }



if __name__ == "__main__":
    test_questions = [
    "What is Python?",
    "Define an HTTP GET request.",
    "Compare FastAPI and Flask.",
    "Explain how semantic caching works.",
    "Why is semantic search useful?",
    "How do I create an API using FastAPI?",
    "What is the difference between HTTP PUT and PATCH?",
    ]

    print("\n--- SmartRAG Question Router Tests ---")

    for question in test_questions:
        result = classify_question(question)

        print(f"\nQuestion: {question}")
        print(f"Complexity: {result['complexity']}")
        print(f"Route: {result['route']}")
        print(f"Reason: {result['reason']}")
