import time
import requests


class OllamaClient:
    """
    Simple local Ollama LLM client.

    Uses the Ollama HTTP API to generate answers
    from a supplied context.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host.rstrip("/")

    def generate(
        self,
        query: str,
        context: str
    ) -> dict:
        """
        Generate an answer using the supplied context.

        Returns:
            response
            latency_ms
        """

        prompt = f"""
Answer the user's question using only the context below.

Question:
{query}

Context:
{context}

Answer:
""".strip()

        start_time = time.perf_counter()

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        end_time = time.perf_counter()

        response.raise_for_status()

        data = response.json()

        return {
            "response": data.get(
                "response",
                ""
            ).strip(),
            "latency_ms": (
                end_time - start_time
            ) * 1000
        }