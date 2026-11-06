
import sqlite3
import json
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATABASE_PATH = Path("data/semantic_cache.db")


# --------------------------------------------------
# Database Connection
# --------------------------------------------------

def get_connection():
    """
    Create a connection to the SQLite database.
    """

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(DATABASE_PATH)


# --------------------------------------------------
# Initialize Cache
# --------------------------------------------------

def initialize_cache():
    """
    Create the semantic cache table if it doesn't exist.
    """

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                question_embedding TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


# --------------------------------------------------
# Save Question and Answer
# --------------------------------------------------

def save_to_cache(
    question: str,
    question_embedding,
    answer: str,
):
    """
    Save a question, its embedding, and its answer.
    """

    embedding_json = json.dumps(
        question_embedding.tolist()
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO semantic_cache (
                question,
                question_embedding,
                answer
            )
            VALUES (?, ?, ?)
            """,
            (
                question,
                embedding_json,
                answer,
            ),
        )


# --------------------------------------------------
# Identify Question Intent
# --------------------------------------------------

def get_question_intent(question: str) -> str:
    """
    Identify the broad intent of a question using simple rules.

    This is a lightweight safeguard, not a complete
    natural-language understanding system.
    """

    question = question.lower().strip()

    # Comparison questions
    if re.search(
        r"\b(compare|comparison|versus|vs\.?)\b",
        question,
    ):
        return "comparison"

    # Disadvantages and limitations
    if re.search(
        r"\b(disadvantages?|drawbacks?|limitations?|cons)\b",
        question,
    ):
        return "disadvantages"

    # Advantages and benefits
    if re.search(
        r"\b(advantages?|benefits?|pros)\b",
        question,
    ):
        return "advantages"

    # How-to questions
    if re.search(
        r"\b(how do i|how to|how can i|steps to|tutorial)\b",
        question,
    ):
        return "how_to"

    # Questions about reasons
    if re.search(
        r"\b(why|reason|reasons)\b",
        question,
    ):
        return "why"

    # Definition and explanation questions
    if re.search(
        r"\b(what is|what are|who is|who are|define|explain)\b",
        question,
    ):
        return "definition"

    # Unknown question type
    return "other"


# --------------------------------------------------
# Search Semantic Cache
# --------------------------------------------------

def search_cache(
    query_embedding,
    query: str,
    similarity_threshold: float = 0.75,
):
    """
    Find a cached answer whose question has a compatible
    broad intent and sufficient embedding similarity.

    Return the best matching entry or None.
    """

    # Determine the intent of the new question
    query_intent = get_question_intent(query)

    # Load cached entries from SQLite
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, question, question_embedding, answer
            FROM semantic_cache
            """
        ).fetchall()

    # Nothing has been cached yet
    if not rows:
        return None

    best_match = None
    best_similarity = -1.0

    # Compare the new question against cached questions
    for row in rows:
        cache_id, cached_question, embedding_json, answer = row

        # Reject entries with a different broad question intent
        cached_intent = get_question_intent(cached_question)

        if cached_intent != query_intent:
            continue

        # Convert the stored JSON embedding back into a NumPy array
        cached_embedding = np.array(
            json.loads(embedding_json),
            dtype=np.float32,
        )

        # The embeddings are normalized, so the dot product
        # is equivalent to cosine similarity.
        similarity = float(
            np.dot(query_embedding, cached_embedding)
        )

        # Keep the most similar compatible question
        if similarity > best_similarity:
            best_similarity = similarity

            best_match = {
                "cache_id": cache_id,
                "question": cached_question,
                "answer": answer,
                "similarity": similarity,
            }

    # Return a match only if it passes the threshold
    if (
        best_match is not None
        and best_similarity >= similarity_threshold
    ):
        return best_match

    return None


# --------------------------------------------------
# Run This File Directly
# --------------------------------------------------

if __name__ == "__main__":

    initialize_cache()

    print("Semantic cache database initialized.")
    print(f"Database location: {DATABASE_PATH}")
