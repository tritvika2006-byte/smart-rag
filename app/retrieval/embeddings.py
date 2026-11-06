
from sentence_transformers import SentenceTransformer

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents


# 1. Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(chunks: list[dict]):
    """
    Convert text chunks into numerical vectors.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


if __name__ == "__main__":

    # 2. Load documents
    documents = load_documents("data/documents")

    # 3. Split documents into chunks
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    # 4. Generate embeddings
    embeddings = generate_embeddings(chunks)

    # 5. Display results
    print("\n--- Embedding Results ---")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    # Display the first vector's first five values
    print("\nFirst embedding (first 5 values):")
    print(embeddings[0][:5])
