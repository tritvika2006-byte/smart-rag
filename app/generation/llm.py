
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def generate_answer(prompt: str, model_name: str = MODEL_NAME) -> str:
    """
    Send a prompt to a selected local Ollama model
    and return its generated answer.
    """

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()


if __name__ == "__main__":
    test_prompt = "Explain HTTP GET requests in two sentences."

    answer = generate_answer(test_prompt)

    print("\n--- Ollama Response ---")
    print(answer)
