from app.caching.semantic_cache import (
    initialize_cache,
    save_to_cache,
    search_cache,
)
from app.retrieval.embeddings import model


def main():
    # Create the database table if necessary
    initialize_cache()

    # Create an embedding for a question
    question = "What is FastAPI?"

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    # Save a sample answer
    save_to_cache(
        question=question,
        question_embedding=question_embedding,
        answer="FastAPI is a Python framework for building APIs.",
    )

    print("Sample question saved to cache.")

    # Test a semantically similar question
    new_question = "Explain FastAPI in simple words."

    new_embedding = model.encode(
        new_question,
        normalize_embeddings=True,
    )

    result = search_cache(
    query_embedding=new_embedding,
    query=new_question,
    similarity_threshold=0.75,
)

    print("\n--- Cache Lookup Result ---")

    if result:
        print("Cache hit!")
        print(f"Matched question: {result['question']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Cached answer: {result['answer']}")
    else:
        print("Cache miss. Run the normal RAG pipeline.")


if __name__ == "__main__":
    main()