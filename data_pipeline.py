import torch

DATA_PATH = "data/input.txt"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    original_text = f.read()

char_set = sorted(set(original_text))

char_to_id_map = {char: i for i, char in enumerate(char_set)}


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
    if char not in char_to_id_map:
        raise ValueError(f"Character {repr(char)} is not in the known character set.")
    return char_to_id_map[char]


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


def get_batch(
    data: torch.Tensor, context_length: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of context windows and corresponding next tokens.

    Args:
        data: A tensor of token IDs representing the text.
        context_length: The number of previous tokens used to predict the next token.
        batch_size: The number of token sequences to sample.

    Returns:
        A tuple of tensors. The first tensor contains input token sequences, with
        shape [batch_size, context_length], and the second tensor contains the
        corresponding target token sequences, with shape [batch_size, context_length].
    """
    start_indices = torch.randint(
        data.shape[0] - context_length, size=(batch_size,), device=data.device
    )
    context_offsets = torch.arange(context_length, device=data.device)

    # Broadcasting shapes:
    # [batch_size, 1] + [context_length] -> [batch_size, context_length]
    input_indices = start_indices.unsqueeze(1) + context_offsets

    # The target token is the same sequence as x, shifted once into the future.
    target_indices = input_indices + 1

    input_tensor = data[input_indices]
    target_tensor = data[target_indices]
    return (input_tensor, target_tensor)


if __name__ == "__main__":
    if original_text == decode_text(encode_text(original_text)):
        print("TOKENISER SUCCESSFUL")
    else:
        print("TOKENISER FAILED")

    print("dataset length: ", len(original_text))
    print("vocabulary size: ", len(char_set))
    print("unique characters (alphabetised): ", char_set)
