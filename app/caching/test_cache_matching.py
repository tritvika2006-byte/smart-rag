
from app.caching.semantic_cache import get_question_intent

test_cases = [
    (
        "What is FastAPI?",
        "Explain FastAPI in simple words.",
    ),
    (
        "What is FastAPI?",
        "Compare FastAPI and Flask.",
    ),
    (
        "What is FastAPI?",
        "What are the disadvantages of FastAPI?",
    ),
    (
        "What is FastAPI?",
        "What are the advantages of FastAPI?",
    ),
]

print("\n--- Cache Compatibility Tests ---")

for cached_question, new_question in test_cases:
    cached_intent = get_question_intent(cached_question)
    new_intent = get_question_intent(new_question)

    compatible = cached_intent == new_intent

    print(f"\nCached question: {cached_question}")
    print(f"New question:    {new_question}")
    print(f"Cached intent:   {cached_intent}")
    print(f"New intent:      {new_intent}")
    print(f"Compatible:      {compatible}")
