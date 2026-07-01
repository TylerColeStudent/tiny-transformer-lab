import torch
import torch.nn as nn
import torch.optim as optim

from data_pipeline import CharacterTokeniser, get_batch
from transformer_model import TransformerLanguageModel


def estimate_loss(
    data: torch.Tensor,
    model: TransformerLanguageModel,
    batch_size: int,
    trials: int,
    loss_func: nn.Module,
) -> float:
    """Estimate the average loss of a Transformer Language Model.

    Args:
        data: A tensor of token IDs to sample batches from.
        model: An instance of TransformerLanguageModel to evaluate.
        batch_size: The number of input-target pairs in each batch.
        trials: The number of batches used to estimate loss. A higher value
            increases accuracy at the cost of computation time.
        loss_func: The loss function used to compute the losses.

    Returns:
        The average loss across 'trials' different batches, as a floating point number.
    """
    was_training = model.training
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            inputs, targets = get_batch(data, model.context_length, batch_size)
            logits = model(inputs)
            loss = loss_func(logits.flatten(0, 1), targets.flatten(0, 1))
            total_loss += loss.item()

    if was_training:
        model.train()
    return total_loss / trials


def train_model(encoded_text: list[int], vocab_size: int) -> TransformerLanguageModel:
    """Train a transformer language model on encoded text data.

    Args:
        encoded_text: The full dataset encoded as a list of token IDs.
        vocab_size: The number of unique tokens in the tokeniser vocabulary.

    Returns:
        The fully trained TransformerLanguageModel instance.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARN_RATE = 1e-3
    STEPS = 500
    BATCH_SIZE = 256
    EVAL_TRIALS = 20
    EMBEDDING_DIM = 64
    CONTEXT_LENGTH = 64
    HIDDEN_SIZE = 256
    NUM_OF_HEADS = 4
    NUM_OF_BLOCKS = 8

    print("Device: ", DEVICE)

    cutoff = int(0.9 * len(encoded_text))
    train_tokens = encoded_text[:cutoff]
    val_tokens = encoded_text[cutoff:]
    train_data = torch.tensor(train_tokens, dtype=torch.long, device=DEVICE)
    val_data = torch.tensor(val_tokens, dtype=torch.long, device=DEVICE)

    model = TransformerLanguageModel(
        vocab_size,
        EMBEDDING_DIM,
        CONTEXT_LENGTH,
        NUM_OF_HEADS,
        HIDDEN_SIZE,
        NUM_OF_BLOCKS,
    )
    model.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=LEARN_RATE)

    for step in range(1, STEPS + 1):
        model.train()
        optimiser.zero_grad()
        inputs, targets = get_batch(train_data, model.context_length, BATCH_SIZE)
        # Both have shape [batch_size, context_length].

        logits = model(inputs)  # Shape: [batch_size, context_length, vocab_size]

        loss = loss_function(logits.flatten(0, 1), targets.flatten(0, 1))
        loss.backward()
        optimiser.step()

        if step == 1 or step % 100 == 0:
            train_loss = estimate_loss(
                train_data, model, BATCH_SIZE, EVAL_TRIALS, loss_function
            )
            val_loss = estimate_loss(
                val_data, model, BATCH_SIZE, EVAL_TRIALS, loss_function
            )
            print(f"Estimated training loss at step {step}: {train_loss:.3f}")
            print(f"Estimated validation loss at step {step}: {val_loss:.3f}")

    return model


if __name__ == "__main__":
    DATA_PATH = "data/input.txt"

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        original_text = f.read()

    tokeniser = CharacterTokeniser(original_text)
    encoded_text = tokeniser.encode_text(original_text)

    model = train_model(encoded_text, tokeniser.vocab_size)
    device = next(model.parameters()).device

    start_chars = "Scrooge "
    start_tokens = tokeniser.encode_text(start_chars)
    initial_context = torch.tensor(start_tokens, dtype=torch.long, device=device)

    print("____Generated Text____")
    print(tokeniser.decode_text(model.generate(initial_context, 500)))
