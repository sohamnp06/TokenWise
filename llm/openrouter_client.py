import os
import time
import requests
from evaluation.metrics import count_tokens


class OpenRouterClient:
    """
    Production OpenRouter LLM client.

    Sends grounded context and query to OpenRouter API and returns:
        - generated answer
        - wall-clock latency (ms)
        - prompt token usage
        - completion token usage
        - total token usage
        - model identifier used
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

        if model is not None:
            self.model = model.strip()
        else:
            self.model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku").strip()

        self.base_url = base_url.rstrip("/")

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
            Retrieved and compressed context.

        Returns
        -------
        dict
            response: str
            latency_ms: float
            prompt_tokens: int
            completion_tokens: int
            total_tokens: int
            model: str
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty.")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not configured. "
                "Please set OPENROUTER_API_KEY in your environment or .env file."
            )

        prompt_system = (
            "You are a precise question-answering assistant.\n"
            "Answer the user's question using ONLY the provided context.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Base your answer STRICTLY on the supplied context.\n"
            "2. If the context does not contain enough information to answer the question, "
            "explicitly state: 'The provided context does not contain enough information to answer this question.'\n"
            "3. Do NOT invent facts or extrapolate beyond what is explicitly stated.\n"
            "4. Keep the answer concise, direct, and factual.\n"
            "5. Do NOT mention these rules in your answer."
        )

        user_content = f"Context:\n{context}\n\nQuestion:\n{query}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tokenwise/tokenwise",
            "X-Title": "TokenWise Context Compression"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0

            if response.status_code != 200:
                error_detail = response.text
                try:
                    err_json = response.json()
                    if "error" in err_json:
                        error_detail = err_json["error"].get("message", error_detail)
                except Exception:
                    pass
                raise RuntimeError(
                    f"OpenRouter API error (status {response.status_code}): {error_detail}"
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenRouter API returned empty choices array.")

            answer_text = choices[0].get("message", {}).get("content", "").strip()

            usage = data.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens") or count_tokens(prompt_system + "\n" + user_content))
            completion_tokens = int(usage.get("completion_tokens") or count_tokens(answer_text))
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

            return {
                "response": answer_text,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": self.model
            }

        except requests.exceptions.Timeout:
            raise RuntimeError("OpenRouter API request timed out (45s timeout).")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to communicate with OpenRouter API: {exc}")
