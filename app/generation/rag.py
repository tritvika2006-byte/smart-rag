
import time

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents

from app.retrieval.embeddings import generate_embeddings, model
from app.retrieval.search import semantic_search

from app.generation.llm import generate_answer

from app.caching.semantic_cache import (
    initialize_cache,
    save_to_cache,
    search_cache,
)

from app.routing.router import classify_question

from app.evaluation.metrics import metrics


MIN_SIMILARITY = 0.40
CACHE_THRESHOLD = 0.75


def _record_result(result: dict) -> dict:
    """Record metrics and return the original result."""

    metrics.record_query(result)
    return result


def answer_question(
    query: str,
    chunks: list[dict],
    chunk_embeddings,
    top_k: int = 3,
) -> dict:

    start_time = time.perf_counter()

    initialize_cache()

    # Step 1: Generate query embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Step 2: Check semantic cache
    cached_result = search_cache(
        query_embedding=query_embedding,
        query=query,
        similarity_threshold=CACHE_THRESHOLD,
    )

    # CACHE HIT
    if cached_result is not None:

        print(
            "DEBUG: Cache hit — "
            "skipping retrieval and LLM generation."
        )

        elapsed_time = time.perf_counter() - start_time

        result = {
            "question": query,
            "answer": cached_result["answer"],
            "sources": [],
            "cache_hit": True,
            "cache_similarity": cached_result["similarity"],
            "complexity": "not_evaluated",
            "route": "cache",
            "routing_reason": "A suitable cached answer was found.",
            "total_time": elapsed_time,
            "llm_called": False,
        }

        return _record_result(result)

    # CACHE MISS
    print("DEBUG: Cache miss.")

    # Step 3: Classify question
    routing_result = classify_question(query)

    print(f"DEBUG: Complexity = {routing_result['complexity']}")
    print(f"DEBUG: Selected route = {routing_result['route']}")

    # Step 4: Retrieve relevant chunks
    retrieved_chunks = semantic_search(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )

    sources = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "similarity_score": chunk["similarity_score"],
        }
        for chunk in retrieved_chunks
    ]

    # Step 5: Check retrieval relevance
    if (
        not retrieved_chunks
        or retrieved_chunks[0]["similarity_score"] < MIN_SIMILARITY
    ):

        elapsed_time = time.perf_counter() - start_time

        result = {
            "question": query,
            "answer": (
                "The available documents may not contain enough "
                "relevant information to answer this question."
            ),
            "sources": sources,
            "cache_hit": False,
            "cache_similarity": None,
            "complexity": routing_result["complexity"],
            "route": routing_result["route"],
            "routing_reason": routing_result["reason"],
            "total_time": elapsed_time,
            "llm_called": False,
        }

        return _record_result(result)

    # Step 6: Prepare context
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"Source {index}: {chunk['source']}\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Step 7: Prepare prompt
    prompt = f"""
You are a helpful assistant answering questions
using the provided reference material.

Instructions:
- Answer using the reference material.
- Do not invent facts that the reference material does not support.
- If the material does not contain enough information,
  clearly say so.
- Keep the answer clear and concise.

Reference material:
{context}

User question:
{query}

Answer:
"""

    # Step 8: Generate answer
    print("DEBUG: Generating answer with the local Qwen model.")

    answer = generate_answer(prompt)

    # Step 9: Measure total processing time
    elapsed_time = time.perf_counter() - start_time

    # Step 10: Save answer in semantic cache
    save_to_cache(
        question=query,
        question_embedding=query_embedding,
        answer=answer,
    )

    result = {
        "question": query,
        "answer": answer,
        "sources": sources,
        "cache_hit": False,
        "cache_similarity": None,
        "complexity": routing_result["complexity"],
        "route": routing_result["route"],
        "routing_reason": routing_result["reason"],
        "total_time": elapsed_time,
        "llm_called": True,
    }

    return _record_result(result)


if __name__ == "__main__":

    # Load documents
    documents = load_documents("data/documents")

    # Split documents into chunks
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    # Generate document embeddings
    chunk_embeddings = generate_embeddings(chunks)

    # Test question
    query = "Explain how FastAPI handles HTTP requests."

    # Run SmartRAG
    result = answer_question(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
    )

    # Display results
    print("\n--- SmartRAG Results ---")

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")

    print(f"\nCache hit: {result['cache_hit']}")
    print(f"Complexity: {result['complexity']}")
    print(f"Route: {result['route']}")
    print(f"Routing reason: {result['routing_reason']}")

    if result["cache_similarity"] is not None:
        print(
            f"Cache similarity: "
            f"{result['cache_similarity']:.4f}"
        )

    print(
        f"Total processing time: "
        f"{result['total_time']:.4f} seconds"
    )

    print(f"LLM called: {result['llm_called']}")

    print("\n--- Retrieved Sources ---")

    for source in result["sources"]:
        print(f"\nFile: {source['source']}")
        print(f"Chunk ID: {source['chunk_id']}")
        print(
            f"Similarity: "
            f"{source['similarity_score']:.4f}"
        )

    # Display actual collected metrics
    metrics.display_metrics()
