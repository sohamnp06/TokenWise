import tiktoken


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------
#
# cl100k_base is commonly used for token estimation with
# modern OpenAI-style LLM tokenization.
#
# We use it consistently throughout TokenWise so that
# original context and compressed context are measured
# using the same tokenizer.
# ---------------------------------------------------------

encoder = tiktoken.get_encoding(
    "cl100k_base"
)


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text.

    Parameters
    ----------
    text : str
        Text whose tokens should be counted.

    Returns
    -------
    int
        Number of tokens.
    """

    if not text:
        return 0

    return len(
        encoder.encode(text)
    )


def compression_ratio(
    original_tokens: int,
    compressed_tokens: int
) -> float:
    """
    Calculate percentage of tokens removed.

    Formula:

        compression_ratio =
            (1 - compressed / original) * 100

    Example:

        Original = 1000 tokens
        Compressed = 300 tokens

        Result = 70%

    Parameters
    ----------
    original_tokens : int
        Number of tokens before compression.

    compressed_tokens : int
        Number of tokens after compression.

    Returns
    -------
    float
        Percentage of tokens saved.
    """

    if original_tokens <= 0:
        return 0.0

    saved = (
        1
        -
        (
            compressed_tokens
            /
            original_tokens
        )
    )

    return saved * 100


def tokens_saved(
    original_tokens: int,
    compressed_tokens: int
) -> int:
    """
    Calculate the absolute number of tokens removed.

    Parameters
    ----------
    original_tokens : int
        Tokens before compression.

    compressed_tokens : int
        Tokens after compression.

    Returns
    -------
    int
        Number of tokens saved.
    """

    return max(
        0,
        original_tokens - compressed_tokens
    )