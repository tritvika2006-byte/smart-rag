
from app.caching.semantic_cache import get_question_intent


test_questions = [
    "What is FastAPI?",
    "Explain FastAPI in simple words.",
    "Compare FastAPI and Flask.",
    "What are the disadvantages of FastAPI?",
    "What are the advantages of FastAPI?",
    "How do I build an API using FastAPI?",
    "Why should I use FastAPI?",
]


print("\n--- Question Intent Tests ---")

for question in test_questions:
    intent = get_question_intent(question)

    print(f"Question: {question}")
    print(f"Intent:   {intent}")
    print()
