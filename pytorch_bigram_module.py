import torch
import torch.nn as nn
import torch.optim as optim

from tokeniser import encode_text, original_text, char_set, char_to_id, id_to_char
from pytorch_bigram import get_batch


class BigramModel(nn.Module):
    """A simple Bigram Language Model implemented in PyTorch.

    This model predicts the next token in a sequence based on the single previous
    token using a learned embedding lookup table.

    Args:
        vocab_size: The number of unique tokens in the dataset.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Get next-token logits for a sequence of input tokens.

        Args:
            input_tokens: A tensor of shape [batch_size] of token IDs

        Returns:
            A tensor of shape [batch_size, vocab_size] containing next-token
                logits for every input token.
        """
        logits = self.token_table(input_tokens)
        return logits

    def generate(self, start_chars: str, length: int) -> str:
        """Generate text using the model's learned bigram logits.

        Args:
            start_chars: A string of characters to start the text.
            length: The number of additional characters to generate.

        Returns:
            The initial 'start_chars' followed by the generated characters.

        Raises:
            ValueError: If the last character in 'start_chars' is not in the
                known character set.
        """
        self.eval()

        # Use the same device as the model's parameters.
        device = next(self.parameters()).device
        with torch.no_grad():
            current_token = char_to_id(start_chars[-1])
            output = start_chars

            for _ in range(length):
                input_tensor = torch.tensor(current_token, device=device)
                token_scores = self(input_tensor)  # Shape: [vocab_size]

                # Convert logits to probabilities before sampling.
                token_probs = torch.softmax(token_scores, dim=0)  # Shape: [vocab_size]
                next_token = int(torch.multinomial(token_probs, 1).item())
                output += id_to_char(next_token)
                current_token = next_token

        return output


def estimate_loss(
    data: torch.Tensor,
    model: BigramModel,
    batch_size: int,
    trials: int,
    loss_func: nn.Module,
) -> float:
    """Estimate the average loss of a bigram model.

    Args:
        data: A tensor of token IDs to sample batches from.
        model: An instance of BigramModel to evaluate.
        batch_size: The number of input-target pairs in each batch.
        trials: The number of batches used to estimate loss. A higher value
            increases accuracy at the cost of computation time.
        loss_func: The loss function used to compute the losses.

    Returns:
        The average loss across 'trials' different batches, as a floating point number.
    """
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            inputs, targets = get_batch(data, batch_size)
            logits = model(inputs)
            loss = loss_func(logits, targets)
            total_loss += loss.item()

    return total_loss / trials


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARN_RATE = 2e-3
    STEPS = 5000
    BATCH_SIZE = 1024
    EVAL_TRIALS = 100

    print("Device: ", DEVICE)

    encoded_text = encode_text(original_text)
    cutoff = int(0.9 * len(encoded_text))
    train_tokens = encoded_text[:cutoff]
    val_tokens = encoded_text[cutoff:]
    train_data = torch.tensor(train_tokens, dtype=torch.long, device=DEVICE)
    val_data = torch.tensor(val_tokens, dtype=torch.long, device=DEVICE)

    vocab_size = len(char_set)

    model = BigramModel(vocab_size)
    model.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=LEARN_RATE)

    for step in range(1, STEPS + 1):
        model.train()
        optimiser.zero_grad()
        inputs, targets = get_batch(train_data, BATCH_SIZE)
        logits = model(inputs)

        loss = loss_function(logits, targets)
        loss.backward()
        optimiser.step()

        if step == 1 or step % 1000 == 0:
            train_loss = estimate_loss(
                train_data, model, BATCH_SIZE, EVAL_TRIALS, loss_function
            )
            val_loss = estimate_loss(
                val_data, model, BATCH_SIZE, EVAL_TRIALS, loss_function
            )
            print(f"Estimated training loss at step {step}: {train_loss:.3f}")
            print(f"Estimated validation loss at step {step}: {val_loss:.3f}")

    print("____Generated Text____")
    print(model.generate("S", 200))
