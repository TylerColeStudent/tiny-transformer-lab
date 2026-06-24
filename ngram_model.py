from collections import Counter
import random

from tokeniser import original_text, char_to_id, id_to_char, encode_text, decode_text


def sample_next_token(input_tokens: tuple[int, ...], ngram_counts: dict) -> int:
    """Choose a next token to follow a sequence of input tokens based on frequencies.

    Args:
        input_tokens: A tuple of token IDs representing the current context.
        ngram_counts: A dictionary mapping token sequences to Counter objects
            containing next-token frequencies.

    Returns:
        The selected integer token ID to follow the sequence of input token IDs.

    Raises:
        ValueError: If the sequence of input tokens did not appear in the training text.
        RuntimeError: If sampling fails despite counts being available.
    """
    if input_tokens not in ngram_counts:
        raise ValueError(
            f"No next-token counts available for the input token(s): {input_tokens}"
            f"(characters: {decode_text(list(input_tokens))})"
        )

    next_token_counts = ngram_counts[input_tokens]

    # Use weighted random sampling to choose the next token based on frequencies.
    rand = random.randint(1, sum(next_token_counts.values()))
    for next_token, count in next_token_counts.most_common():
        rand -= count
        if rand <= 0:
            return next_token

    raise RuntimeError(
        f"Failed to sample a next token for the input token(s): {input_tokens} "
        f"(characters: {decode_text(list(input_tokens))})"
    )


def generate_text(
    start_chars: str, length: int, context_size: int, ngram_counts: dict
) -> str:
    """Generate a string of text using observed character frequencies.

    Args:
        start_chars: A string of at least 'context_size' characters to start the text.
        length: The number of additional characters to generate.
        context_size: The number of previous characters to be considered when
            predicting the next character.
        ngram_counts: A dictionary mapping token sequences to Counter objects
            containing next-token frequencies.

    Returns:
        The initial 'start_chars' followed by the generated characters.

    Raises:
        ValueError: If 'start_chars' contains less than 'context_size' characters.
        ValueError: If a character in 'start_chars' is not in the known character set.
        ValueError: If any generated context was not seen in the training text.
    """
    if len(start_chars) < context_size:
        raise ValueError("Not enough initial context provided.")

    tokens = [char_to_id(char) for char in start_chars]
    context = tokens[-context_size:]  # Keep only the most recent context tokens.
    output = start_chars

    for _ in range(length):
        # Use a sliding context window.
        context.append(sample_next_token(tuple(context), ngram_counts))
        context.pop(0)
        output += id_to_char(context[-1])

    return output


def get_ngram_counts(text: str, context_size: int) -> dict:
    """Create a dictionary mapping token sequences to next-token frequencies.

    Args:
        text: A string to be used as the training text.
        context_size: The number of previous tokens used as context. For example,
            1 gives a bigram model, and 2 gives a trigram model.

    Returns:
        A dictionary mapping context tuples to Counter objects containing
            next-token frequencies.

    Raises:
        ValueError: If a character in 'text' is not in the known character set.
    """
    encoded_text = encode_text(text)
    ngram_counts = {}

    # Slide a context window through the text and count which token follows it.
    for i in range(len(encoded_text) - context_size):
        context_tokens = tuple(encoded_text[i : i + context_size])
        next_token = encoded_text[i + context_size]
        if context_tokens not in ngram_counts:
            ngram_counts[context_tokens] = Counter()
        ngram_counts[context_tokens][next_token] += 1

    return ngram_counts


if __name__ == "__main__":
    CONTEXT_SIZE = 3  # 1 for bigram, 2 for trigram, etc.
    GENERATED_LENGTH = 500
    START_CHARS = "Scr"

    ngram_counts = get_ngram_counts(original_text, CONTEXT_SIZE)

    print("____Generated Text____")
    print(generate_text(START_CHARS, GENERATED_LENGTH, CONTEXT_SIZE, ngram_counts))
