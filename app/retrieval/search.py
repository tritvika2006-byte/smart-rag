from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.embeddings import generate_embeddings, model


def semantic_search(
    query: str,
    chunks: list[dict],
    chunk_embeddings,
    top_k: int = 3,
) -> list[dict]:
    """
    Find the most relevant document chunks for a query
    using cosine similarity.
    """

    # 1. Convert the user's query into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # 2. Calculate similarity between the query and every chunk
    similarities = chunk_embeddings @ query_embedding

    # 3. Rank chunks from highest to lowest similarity
    ranked_indices = similarities.argsort()[::-1]

    # 4. Return the top-k most relevant chunks
    results = []

    for index in ranked_indices[:top_k]:
        result = chunks[int(index)].copy()

        result["similarity_score"] = float(similarities[index])

        results.append(result)

    return results


if __name__ == "__main__":

    # Load documents and create chunks
    documents = load_documents("data/documents")

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    # Generate embeddings for all chunks
    chunk_embeddings = generate_embeddings(chunks)

    # Ask a test question
    query = "How does FastAPI validate incoming request data?"

    # Search for relevant chunks
    results = semantic_search(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
    )

    # Display the results
    print(f"\nQuery: {query}")
    print("\n--- Search Results ---")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Text: {result['text'][:250]}...")