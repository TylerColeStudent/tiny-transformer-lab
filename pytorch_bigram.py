import torch
import torch.nn.functional as F

from tokeniser import original_text, char_to_id, id_to_char, encode_text, char_set


def sample_next_token_from_logits(input_token: int, bigram_scores: torch.Tensor) -> int:
    """Choose a next token to follow an input token based on learned bigram logits.

    Args:
        input_token: The token ID to be used as context.
        bigram_scores: A tensor of shape [vocab_size, vocab_size] containing learned
            next-token logits for each token.

    Returns:
        The integer token ID of the selected next token.

    Raises:
        ValueError: If 'input_token' is outside the valid token ID range.
    """
    if input_token < 0 or input_token >= bigram_scores.shape[0]:
        raise ValueError(
            f"input_token must be between 0 and {bigram_scores.shape[0] - 1}, "
            f"got {input_token}"
        )

    token_scores = bigram_scores[input_token]

    # Convert raw logits to probabilities before sampling.
    token_probs = torch.softmax(token_scores, dim=0)
    output_token = torch.multinomial(token_probs, 1)
    return int(output_token.item())


def generate_text_from_logits(
    start_chars: str, length: int, bigram_scores: torch.Tensor
) -> str:
    """Generate a string of text using learned bigram logits.

    Args:
        start_chars: A string of characters to start the text.
        length: The number of additional characters to generate.
        bigram_scores: A tensor of shape [vocab_size, vocab_size] containing learned
            next-token logits for each token.

    Returns:
        The initial 'start_chars' followed by the generated text.

    Raises:
        ValueError: If the last character in 'start_chars' is not in the known
            character set.
    """
    with torch.no_grad():
        current_token = char_to_id(start_chars[-1])
        output = start_chars
        for _ in range(length):
            next_token = sample_next_token_from_logits(current_token, bigram_scores)
            output += id_to_char(next_token)
            current_token = next_token
    return output


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARN_RATE = 10
    STEPS = 500

    encoded_text = encode_text(original_text)

    data = torch.tensor(encoded_text, dtype=torch.long, device=DEVICE)

    inputs = data[:-1]
    targets = data[1:]
    vocab_size = len(char_set)

    bigram_scores = torch.zeros(
        vocab_size, vocab_size, dtype=torch.float32, device=DEVICE, requires_grad=True
    )

    print("Device: ", DEVICE)

    for step in range(1, STEPS + 1):
        bigram_scores.grad = None
        logits = bigram_scores[inputs]  # Shape: [len(encoded_text) - 1, vocab_size]
        loss = F.cross_entropy(logits, targets)

        # Calculate how each score should change to reduce the loss
        loss.backward()

        with torch.no_grad():
            if bigram_scores.grad is None:
                raise RuntimeError("bigram_scores.grad does not exist")

            # Update the logits using gradient descent.
            bigram_scores -= LEARN_RATE * bigram_scores.grad

        if step == 1 or step % 100 == 0:
            print(f"loss at step {step}: {loss.item():.3f}")

    print(generate_text_from_logits("S", 200, bigram_scores))
