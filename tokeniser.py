from collections import Counter
import random

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


def sample_next_token(input_token: int, bigram_counts: dict) -> int:
    next_char_counts = bigram_counts[input_token]
    if not next_char_counts:
        raise ValueError(
            f"No next-token counts available for the input token: {input_token} (character: {repr(id_to_char(input_token))})"
        )
    rand = random.randint(1, sum(next_char_counts.values()))
    output_token = None

    for next_token, count in next_char_counts.most_common():
        rand -= count
        if rand <= 0:
            output_token = next_token
            break

    if output_token is None:
        raise RuntimeError(
            f"Failed to sample a next token for the input token: {input_token} (character: {repr(id_to_char(input_token))})"
        )

    return output_token


def generate_text(start_char: str, length: int, bigram_counts: dict) -> str:
    token = char_to_id(start_char)
    output = start_char

    for _ in range(length):
        token = sample_next_token(token, bigram_counts)
        output += id_to_char(token)

    return output


if original_text == decode_text(encode_text(original_text)):
    print("TOKENISER SUCCESSFUL")
else:
    print("TOKENISER FAILED")

print("dataset length: ", len(original_text))
print("vocabulary size: ", len(sorted_chars))
print("unique characters (alphabetised): ", sorted_chars)
print("first 50 characters in dataset: ", original_text[:50])
print("first 50 tokens: ", encode_text(original_text[:50]))
print("first 50 decoded characters: ", decode_text(encode_text(original_text[:50])))

encoded_text = encode_text(original_text)

bigram_counts = {token_id: Counter() for token_id in range(len(sorted_chars))}

for i in range(len(encoded_text) - 1):
    input_token = encoded_text[i]
    target_token = encoded_text[i + 1]
    bigram_counts[input_token][target_token] += 1

print("____Generated Text____")
print(generate_text("s", 500, bigram_counts))
