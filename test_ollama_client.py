from llm.ollama_client import OllamaClient


print("\n")
print("=" * 70)
print("                    OLLAMA CLIENT TEST")
print("=" * 70)


client = OllamaClient(
    model="llama3.2:latest"
)


query = (
    "What is renewable energy?"
)


context = """
Renewable energy is energy generated from
naturally replenishing resources such as sunlight,
wind, and water.
"""


print("\nRunning Llama 3.2...\n")


result = client.generate(
    query=query,
    context=context
)


print("=" * 70)
print("                         RESULT")
print("=" * 70)


print(
    "\nAnswer:"
)

print(
    result["response"]
)


print(
    f"\nLatency: "
    f"{result['latency_ms']:.2f} ms"
)


print(
    f"Prompt tokens: "
    f"{result['prompt_tokens']}"
)


print(
    f"Completion tokens: "
    f"{result['completion_tokens']}"
)


print(
    f"Total tokens: "
    f"{result['total_tokens']}"
)


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()