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


if __name__ == "__main__":
    if original_text == decode_text(encode_text(original_text)):
        print("TOKENISER SUCCESSFUL")
    else:
        print("TOKENISER FAILED")

    print("dataset length: ", len(original_text))
    print("vocabulary size: ", len(sorted_chars))
    print("unique characters (alphabetised): ", sorted_chars)
