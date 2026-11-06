from pathlib import Path


def load_documents(folder_path: str) -> list[dict]:
    """
    Load all text files from a directory.

    Each document contains:
    - document_id
    - source
    - text
    """

    folder = Path(folder_path)
    documents = []

    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        document = {
            "document_id": file_path.stem,
            "source": file_path.name,
            "text": text,
        }

        documents.append(document)

    return documents


if __name__ == "__main__":
    documents = load_documents("data/documents")

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        print(f"\nSource: {document['source']}")
        print(f"Characters: {len(document['text'])}")