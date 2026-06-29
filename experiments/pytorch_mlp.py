import torch
import torch.nn as nn
import torch.optim as optim

from data_pipeline import encode_text, id_to_char, char_set, original_text


class MLPLanguageModel(nn.Module):
    """A character-level Multi-Layer Perceptron, implemented in PyTorch.

    This model predicts the next token in a sequence of 'context_length' tokens.
    It has an embedding layer which maps characters into a vector space, a hidden
    layer that learns non-linear interactions between the characters using tanh, and
    an output layer that produces next-token logits.

    Args:
        vocab_size: The number of unique characters in the dataset.
        embedding_dim: The number of dimensions of the space that the characters should
            be embedded in.
        context_length: The number of tokens in each sequence, all of which are
            considered in predicting the next token.
        hidden_size: The number of neurons in the hidden layer.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, context_length: int, hidden_size: int
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(context_length * embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Get next-token logits for a batch of token sequences.

        Args:
            input_tokens: A 2D tensor of shape [batch_size, context_length] of
                token IDs. Each row represents a single sequence of context from
                the training text.

        Returns:
            A tensor of shape [batch_size, vocab_size] containing logits for each
                possible next token, for every sequence in the batch.
        """
        x = self.embedding(input_tokens)
        # Shape: [batch_size, context_length, embedding_dim]

        x = self.hidden(x.flatten(start_dim=1))
        # Shape: [batch_size, hidden_size]

        x = torch.tanh(x)
        logits = self.output(x)
        # Shape: [batch_size, vocab_size]

        return logits

    def generate(self, start_chars: str, length: int) -> str:
        """Generate text using the model's learned logits.

        Args:
            start_chars: A string of at least 'context_length' characters to start
                the text.
            length: The number of additional characters to generate.

        Returns:
            The initial 'start_chars' followed by the generated text.

        Raises:
            ValueError: If 'start_chars' contains less than 'context_length' characters.
            ValueError: If any of the last 'context_length' characters in 'start_chars'
                is not in the known character set.
        """
        if len(start_chars) < self.context_length:
            raise ValueError(
                f"Only {len(start_chars)} starting characters received, "
                f"expected at least {self.context_length}."
            )

        self.eval()

        # Use the same device as the model's parameters.
        device = next(self.parameters()).device
        with torch.no_grad():

            # Consider the most recent tokens.
            context_tokens = encode_text(start_chars[-self.context_length :])
            output = start_chars

            for _ in range(length):
                input_tensor = torch.tensor(
                    [context_tokens], dtype=torch.long, device=device
                )
                # Shape: [1, context_length]

                next_token_logits = self(input_tensor)
                # Shape: [1, vocab_size]

                # Convert logits to probabilities before sampling.
                next_token_probs = torch.softmax(next_token_logits, dim=1)
                # Shape: [1, vocab_size]

                next_token = int(torch.multinomial(next_token_probs, 1).item())
                output += id_to_char(next_token)

                # Slide the context window forward.
                context_tokens.append(next_token)
                context_tokens.pop(0)

        return output


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
        corresponding target tokens, with shape [batch_size].
    """
    start_indices = torch.randint(
        data.shape[0] - context_length, size=(batch_size,), device=data.device
    )
    context_offsets = torch.arange(context_length, device=data.device)

    # Broadcasting shapes:
    # [batch_size, 1] + [context_length] -> [batch_size, context_length]
    input_indices = start_indices.unsqueeze(1) + context_offsets

    # The target token is the token immediately after the context window.
    target_indices = start_indices + context_length

    input_tensor = data[input_indices]
    target_tensor = data[target_indices]
    return (input_tensor, target_tensor)


def estimate_loss(
    data: torch.Tensor,
    model: MLPLanguageModel,
    batch_size: int,
    trials: int,
    loss_func: nn.Module,
) -> float:
    """Estimate the average loss of a MLP Language Model.

    Args:
        data: A tensor of token IDs to sample batches from.
        model: An instance of MLPLanguageModel to evaluate.
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
            inputs, targets = get_batch(data, model.context_length, batch_size)
            logits = model(inputs)
            loss = loss_func(logits, targets)
            total_loss += loss.item()

    return total_loss / trials


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARN_RATE = 1e-3
    STEPS = 10000
    BATCH_SIZE = 1024
    EVAL_TRIALS = 100
    EMBEDDING_DIM = 10
    CONTEXT_LENGTH = 8
    HIDDEN_SIZE = 100

    print("Device:", DEVICE)

    encoded_text = encode_text(original_text)
    cutoff = int(0.9 * len(encoded_text))
    train_tokens = encoded_text[:cutoff]
    val_tokens = encoded_text[cutoff:]
    train_data = torch.tensor(train_tokens, dtype=torch.long, device=DEVICE)
    val_data = torch.tensor(val_tokens, dtype=torch.long, device=DEVICE)

    vocab_size = len(char_set)

    model = MLPLanguageModel(vocab_size, EMBEDDING_DIM, CONTEXT_LENGTH, HIDDEN_SIZE)
    model.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=LEARN_RATE)

    for step in range(1, STEPS + 1):
        model.train()
        optimiser.zero_grad()
        inputs, targets = get_batch(train_data, model.context_length, BATCH_SIZE)
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
    print(model.generate("Scrooge ", 500))
