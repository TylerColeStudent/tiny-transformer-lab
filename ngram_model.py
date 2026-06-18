from collections import Counter
import random

CONTEXT_SIZE = 3  # 1 for bigram, 2 for trigram, etc.
GENERATED_LENGTH = 500
START_CHARS = "the"

with open("data/input.txt", "r", encoding="utf-8") as f:
    original_text = f.read()

chars = []
for char in original_text:
    if char in chars:
        continue
    chars.append(char)

sorted_chars = sorted(chars)


def char_to_id(char: str) -> int:
    if len(char) != 1:
        raise ValueError(
            "char_to_id function expects only a single character as the argument."
        )
    if char not in sorted_chars:
        raise ValueError(f"Character {repr(char)} is not in the training data.")
    return sorted_chars.index(char)


def id_to_char(token_id: int) -> str:
    if token_id < 0 or token_id > len(sorted_chars) - 1:
        raise ValueError(
            f"id_to_char function expects an integer between 0 and {len(sorted_chars) - 1}"
        )
    return sorted_chars[token_id]


def encode_text(text: str) -> list[int]:
    return [char_to_id(char) for char in text]


def decode_text(encoded_data: list[int]) -> str:
    return "".join(id_to_char(token_id) for token_id in encoded_data)


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


ngram_counts = get_ngram_counts(original_text, CONTEXT_SIZE)

print("____Generated Text____")
print(generate_text(START_CHARS, GENERATED_LENGTH, CONTEXT_SIZE, ngram_counts))
