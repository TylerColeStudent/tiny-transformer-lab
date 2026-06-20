DATA_PATH = "data/input.txt"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    original_text = f.read()

chars = []
for char in original_text:
    if char in chars:
        continue
    chars.append(char)

char_set = sorted(chars)


def char_to_id(char: str) -> int:
    """Return the token ID corresponding to a single character.

    Args:
        char: A single character from the known character set.

    Returns:
        The integer token ID for the character.

    Raises:
        ValueError: If 'char' is not a single character.
        ValueError: If 'char' is not in the known character set.
    """
    if len(char) != 1:
        raise ValueError(
            "char_to_id function expects only a single character as the argument."
        )
    if char not in char_set:
        raise ValueError(f"Character {repr(char)} is not in the known character set.")
    return char_set.index(char)


def id_to_char(token_id: int) -> str:
    """Return the character corresponding to a token ID.

    Args:
        token_id: The integer token ID. Must be between 0 and len(char_set) - 1.

    Returns:
        The single-character string represented by the token ID.

    Raises:
        ValueError: If token_id is outside the valid token ID range.
    """
    if token_id < 0 or token_id > len(char_set) - 1:
        raise ValueError(
            f"token_id must be between 0 and {len(char_set) - 1}, got {token_id}."
        )
    return char_set[token_id]


def encode_text(text: str) -> list[int]:
    """Encode text into a list of token IDs.

    Args:
        text: The text to encode.

    Returns:
        A list of integer token IDs which represent the full text.

    Raises:
        ValueError: If any character in 'text' is not in the known character set.
    """
    return [char_to_id(char) for char in text]


def decode_text(encoded_data: list[int]) -> str:
    """Decode a list of token IDs into human-readable text.

    Args:
        encoded_data: The list of integer token IDs to decode.

    Returns:
        A string of characters corresponding to the list of token IDs.

    Raises:
        ValueError: If any token ID in 'encoded_data' is outside the valid token ID range.
    """
    return "".join(id_to_char(token_id) for token_id in encoded_data)


if __name__ == "__main__":
    if original_text == decode_text(encode_text(original_text)):
        print("TOKENISER SUCCESSFUL")
    else:
        print("TOKENISER FAILED")

    print("dataset length: ", len(original_text))
    print("vocabulary size: ", len(char_set))
    print("unique characters (alphabetised): ", char_set)
