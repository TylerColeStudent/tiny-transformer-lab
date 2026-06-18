from collections import Counter
import random

from tokeniser import original_text, char_to_id, id_to_char, encode_text, decode_text


def sample_next_token(input_tokens: tuple[int, ...], ngram_counts: dict) -> int:
    if input_tokens not in ngram_counts:
        raise ValueError(
            f"No next-token counts available for the input token(s): {input_tokens} (characters: {decode_text(list(input_tokens))})"
        )

    next_token_counts = ngram_counts[input_tokens]

    rand = random.randint(1, sum(next_token_counts.values()))
    output_token = None

    for next_token, count in next_token_counts.most_common():
        rand -= count
        if rand <= 0:
            output_token = next_token
            break

    if output_token is None:
        raise RuntimeError(
            f"Failed to sample a next token for the input token(s): {input_tokens} (characters: {decode_text(list(input_tokens))})"
        )

    return output_token


def generate_text(
    start_chars: str, length: int, context_size: int, ngram_counts: dict
) -> str:
    if len(start_chars) < context_size:
        raise ValueError("Not enough initial context provided.")

    tokens = [char_to_id(char) for char in start_chars]
    context = tokens[-context_size:]
    output = start_chars

    for _ in range(length):
        context.append(sample_next_token(tuple(context), ngram_counts))
        context.pop(0)
        output += id_to_char(context[-1])

    return output


def get_ngram_counts(text: str, context_size: int) -> dict:
    encoded_text = encode_text(text)
    ngram_counts = {}

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
    START_CHARS = "the"

    ngram_counts = get_ngram_counts(original_text, CONTEXT_SIZE)

    print("____Generated Text____")
    print(generate_text(START_CHARS, GENERATED_LENGTH, CONTEXT_SIZE, ngram_counts))
