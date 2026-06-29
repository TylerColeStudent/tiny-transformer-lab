import torch
import torch.nn.functional as F

from data_pipeline import original_text, char_to_id, id_to_char, encode_text, char_set


def sample_next_token_from_logits(input_token: int, bigram_logits: torch.Tensor) -> int:
    """Choose a next token to follow an input token based on learned bigram logits.

    Args:
        input_token: The token ID to be used as context.
        bigram_logits: A tensor of shape [vocab_size, vocab_size] containing learned
            next-token logits for each token.

    Returns:
        The integer token ID of the selected next token.

    Raises:
        ValueError: If 'input_token' is outside the valid token ID range.
    """
    if input_token < 0 or input_token >= bigram_logits.shape[0]:
        raise ValueError(
            f"input_token must be between 0 and {bigram_logits.shape[0] - 1}, "
            f"got {input_token}"
        )

    token_scores = bigram_logits[input_token]

    # Convert raw logits to probabilities before sampling.
    token_probs = torch.softmax(token_scores, dim=0)
    output_token = torch.multinomial(token_probs, 1)
    return int(output_token.item())


def generate_text(start_chars: str, length: int, bigram_logits: torch.Tensor) -> str:
    """Generate a string of text using learned bigram logits.

    Args:
        start_chars: A string of characters to start the text.
        length: The number of additional characters to generate.
        bigram_logits: A tensor of shape [vocab_size, vocab_size] containing learned
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
            next_token = sample_next_token_from_logits(current_token, bigram_logits)
            output += id_to_char(next_token)
            current_token = next_token
    return output


def get_batch(data: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from token data.

    Args:
        data: A tensor of token IDs representing text.
        batch_size: The desired number of input-target pairs.

    Returns:
        A tuple containing a tensor of input tokens and a tensor of target tokens.
        Both tensors have shape [batch_size].
    """
    indices = torch.randint(data.shape[0] - 1, size=(batch_size,), device=data.device)
    input_tensor = data[indices]
    target_tensor = data[indices + 1]
    return (input_tensor, target_tensor)


def estimate_loss(
    data: torch.Tensor, bigram_logits: torch.Tensor, batch_size: int, trials: int
) -> float:
    """Estimate the average loss of a bigram model.

    Args:
        data: A tensor of token IDs to sample batches from.
        bigram_logits: A tensor of shape [vocab_size, vocab_size] containing learned
            next-token logits for each token.
        batch_size: The desired number of input-target pairs.
        trials: The number of batches used to estimate loss. A higher value
            increases accuracy at the cost of computation time.

    Returns:
        The average loss across 'trials' different batches, as a floating point number.
    """
    total_loss = 0
    with torch.no_grad():
        for _ in range(trials):
            inputs, targets = get_batch(data, batch_size)
            logits = bigram_logits[inputs]
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()

    return total_loss / trials


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARN_RATE = 2
    BATCH_SIZE = 1024
    STEPS = 5000
    EVAL_TRIALS = 100

    print("Device:", DEVICE)

    encoded_text = encode_text(original_text)
    cutoff = int(0.9 * len(encoded_text))
    train_tokens = encoded_text[:cutoff]
    val_tokens = encoded_text[cutoff:]
    train_data = torch.tensor(train_tokens, dtype=torch.long, device=DEVICE)
    val_data = torch.tensor(val_tokens, dtype=torch.long, device=DEVICE)

    vocab_size = len(char_set)
    bigram_logits = torch.zeros(
        vocab_size, vocab_size, dtype=torch.float32, device=DEVICE, requires_grad=True
    )

    for step in range(1, STEPS + 1):
        bigram_logits.grad = None
        inputs, targets = get_batch(train_data, BATCH_SIZE)
        logits = bigram_logits[inputs]  # Shape: [BATCH_SIZE, vocab_size]
        loss = F.cross_entropy(logits, targets)

        # Calculate how each score should change to reduce the loss.
        loss.backward()

        with torch.no_grad():
            if bigram_logits.grad is None:
                raise RuntimeError("bigram_logits.grad does not exist")

            # Update the learned bigram logits using gradient descent.
            bigram_logits -= LEARN_RATE * bigram_logits.grad

        if step == 1 or step % 1000 == 0:
            train_loss = estimate_loss(
                train_data, bigram_logits, BATCH_SIZE, EVAL_TRIALS
            )
            val_loss = estimate_loss(val_data, bigram_logits, BATCH_SIZE, EVAL_TRIALS)
            print(f"Estimated training loss at step {step}: {train_loss:.3f}")
            print(f"Estimated validation loss at step {step}: {val_loss:.3f}")

    print("____Generated Text____")
    print(generate_text("S", 200, bigram_logits))
