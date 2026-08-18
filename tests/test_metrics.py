from evaluation.metrics import (
    count_tokens,
    compression_ratio,
    tokens_saved
)


original_text = """
Renewable energy production increased significantly
between 2020 and 2025. Solar generation increased
by 42% during this period.
"""


compressed_text = """
Solar generation increased by 42% between 2020 and 2025.
"""


original_tokens = count_tokens(
    original_text
)

compressed_tokens = count_tokens(
    compressed_text
)

saved = tokens_saved(
    original_tokens,
    compressed_tokens
)

ratio = compression_ratio(
    original_tokens,
    compressed_tokens
)


print("\n===== TOKEN METRICS TEST =====\n")

print(
    f"Original tokens:   {original_tokens}"
)

print(
    f"Compressed tokens: {compressed_tokens}"
)

print(
    f"Tokens saved:      {saved}"
)

print(
    f"Compression:       {ratio:.2f}%"
)

print(
    "\n==============================\n"
)