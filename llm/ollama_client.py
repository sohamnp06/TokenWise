import time

import requests


class OllamaClient:
    """
    Local Ollama LLM client.

    Sends a prompt to Ollama and returns:

        - generated response
        - latency
        - prompt token count
        - generated token count
        - total token count
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
        Generate an answer using only the supplied context.

        Returns
        -------
        dict
            response
            latency_ms
            prompt_tokens
            completion_tokens
            total_tokens
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not context or not context.strip():
            raise ValueError(
                "Context cannot be empty."
            )

        prompt = f"""
Answer the user's question using only the context below.

If the answer is not explicitly supported by the context,
say that the context does not provide enough information.

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

        prompt_tokens = data.get(
            "prompt_eval_count",
            0
        )

        completion_tokens = data.get(
            "eval_count",
            0
        )

        total_tokens = (
            prompt_tokens
            +
            completion_tokens
        )

        return {
            "response": data.get(
                "response",
                ""
            ).strip(),

            "latency_ms": (
                end_time - start_time
            ) * 1000,

            "prompt_tokens": (
                prompt_tokens
            ),

            "completion_tokens": (
                completion_tokens
            ),

            "total_tokens": (
                total_tokens
            )
        }