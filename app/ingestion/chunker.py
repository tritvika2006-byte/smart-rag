def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    chunk_size: maximum approximate chunk length in characters
    overlap: number of characters shared between adjacent chunks
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Split loaded documents into chunks while preserving source metadata.
    """

    all_chunks = []

    for document in documents:
        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(text_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{document['document_id']}_{index}",
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    return all_chunks


if __name__ == "__main__":
    from app.ingestion.loader import load_documents

    documents = load_documents("data/documents")

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Characters: {len(chunk['text'])}")
        print(f"Text preview: {chunk['text'][:150]}")