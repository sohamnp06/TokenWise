import time

import requests


class OllamaClient:
    """
    Local Ollama LLM client.

    Sends a prompt to Ollama and returns:

        - generated response
        - end-to-end latency
        - Ollama total duration
        - prompt evaluation duration
        - generation duration
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

        Parameters
        ----------
        query : str
            User question.

        context : str
            Retrieved/compressed context.

        Returns
        -------
        dict
            response
            latency_ms
            total_duration_ms
            prompt_eval_duration_ms
            eval_duration_ms
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

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt = f"""
You are a precise question-answering assistant.

Answer the user's question using ONLY the information contained
in the provided context.

IMPORTANT RULES:

1. If the context explicitly contains the answer, state it directly.
2. Do not claim that information is missing when the context
   contains relevant evidence.
3. Do not introduce facts that are not supported by the context.
4. If only part of the question can be answered, answer that part
   and clearly state what information is not available.
5. Keep the answer concise and factual.
6. Do not mention these instructions in your answer.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        # -----------------------------------------------------
        # Request
        # -----------------------------------------------------

        start_time = time.perf_counter()

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,

                # More deterministic answers are useful for
                # benchmarking original vs compressed context.
                "options": {
                    "temperature": 0
                }
            },
            timeout=300
        )

        end_time = time.perf_counter()

        response.raise_for_status()

        data = response.json()

        # -----------------------------------------------------
        # Token counts
        # -----------------------------------------------------

        prompt_tokens = int(
            data.get(
                "prompt_eval_count",
                0
            ) or 0
        )

        completion_tokens = int(
            data.get(
                "eval_count",
                0
            ) or 0
        )

        total_tokens = (
            prompt_tokens
            +
            completion_tokens
        )

        # -----------------------------------------------------
        # Ollama timing information
        #
        # Ollama reports durations in nanoseconds.
        # Convert to milliseconds.
        # -----------------------------------------------------

        ollama_total_duration = (
            data.get(
                "total_duration",
                0
            ) or 0
        )

        load_duration = (
            data.get(
                "load_duration",
                0
            ) or 0
        )

        prompt_eval_duration = (
            data.get(
                "prompt_eval_duration",
                0
            ) or 0
        )

        eval_duration = (
            data.get(
                "eval_duration",
                0
            ) or 0
        )

        total_duration_ms = (
            ollama_total_duration
            /
            1_000_000
        )

        load_duration_ms = (
            load_duration
            /
            1_000_000
        )

        prompt_eval_duration_ms = (
            prompt_eval_duration
            /
            1_000_000
        )

        eval_duration_ms = (
            eval_duration
            /
            1_000_000
        )

        # -----------------------------------------------------
        # End-to-end Python wall-clock latency
        # -----------------------------------------------------

        latency_ms = (
            end_time - start_time
        ) * 1000

        # -----------------------------------------------------
        # Return structured benchmark data
        # -----------------------------------------------------

        return {
            "response": data.get(
                "response",
                ""
            ).strip(),

            # Python measured end-to-end latency.
            "latency_ms": latency_ms,

            # Ollama internal timings.
            "total_duration_ms": (
                total_duration_ms
            ),

            "load_duration_ms": (
                load_duration_ms
            ),

            "prompt_eval_duration_ms": (
                prompt_eval_duration_ms
            ),

            "eval_duration_ms": (
                eval_duration_ms
            ),

            # Token usage.
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