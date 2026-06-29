import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from tokeniser import encode_text, decode_text, char_set, original_text


class TransformerLanguageModel(nn.Module):
    """An autoregressive, decoder-only transformer language model built in PyTorch.

    This model embeds tokens in a high-dimensional vector space. It runs through a set
    number of blocks, where each block is split into two sections. The first section
    consists of multiple attention heads, which allow the tokens to communicate with
    each other. The second section is a simple feed-forward network, where each token
    is individually processed through a non-linear transformation. The result is then
    normalised before being passed through a linear layer and then returned as a
    3D tensor representing next-token logits.

    Args:
        vocab_size: The number of unique tokens in the dataset.
        embedding_dim: The number of dimensions of the space that the tokens should
            be embedded in.
        context_length: The number of tokens in each sequence, all of which are
            considered in predicting the next token.
        num_of_heads: The number of distinct attention heads used in each block
        hidden_size: The number of neurons in the hidden layer of each FeedForward
            network.
        num_of_blocks: The number of blocks used, where each block contains a
            MultiHeadAttention and a FeedForward network.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_of_heads: int,
        hidden_size: int,
        num_of_blocks: int,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

        self.blocks = nn.ModuleList(
            [
                Block(embedding_dim, num_of_heads, hidden_size)
                for _ in range(num_of_blocks)
            ]
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Get next-token logits for every position in the input token sequence.

        Args:
            input_tokens: A tensor of shape [batch_size, context_length] containing
                token IDs.

        Returns:
            A tensor of shape [batch_size, context_length, vocab_size] containing
            the learned next-token logits for every token in the sequence.
        """
        token_embeddings = self.token_embedding(input_tokens)
        # Shape: [batch_size, context_length, embedding_dim]

        current_context_length = min(input_tokens.shape[1], self.context_length)
        pos_indices = torch.arange(current_context_length, device=input_tokens.device)
        pos_embeddings = self.pos_embedding(pos_indices)
        # Shape: [context_length, embedding_dim]

        # Construct a single tensor combining semantic meaning with spatial position.
        x = token_embeddings + pos_embeddings
        # Shape: [batch_size, context_length, embedding_dim]

        for block in self.blocks:
            x = block(x)
        # Shape: [batch_size, context_length, embedding_dim]

        x = self.norm(x)
        logits = self.output(x)
        return logits  # Shape: [batch_size, context_length, vocab_size]

    def generate(self, context: torch.Tensor, length: int) -> str:
        """
        Generate text using the model's learned logits.

        Args:
            context: A 1D tensor of token IDs which represent the initial context
                window.
            length: The number of additional tokens to generate

        Returns:
            A single string of the context followed by the generated characters.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(length):
                if context.shape[0] > self.context_length:
                    current_context = context[-self.context_length :]
                else:
                    current_context = context
                logits = self(current_context.unsqueeze(0))

                next_token_logits = logits[:, -1, :]
                # Shape: [batch_size, vocab_size]

                # Convert logits to probabilities before sampling.
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)

                context = torch.cat([context, next_token.squeeze(dim=1)], dim=0)

        if was_training:
            self.train()
        return decode_text(context.tolist())


class Head(nn.Module):
    """
    A class representing a single attention head for a transformer language model.

    Args:
        embedding_dim: The number of dimensions of the space that the characters should
            be embedded in.
        head_dim: The number of dimensions of space used for the attention queries,
            keys, and values.
    """

    def __init__(self, embedding_dim: int, head_dim: int) -> None:
        super().__init__()

        self.head_dim = head_dim

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get the contextualised embeddings for a tensor of token embeddings.

        Args:
            embeddings: A tensor of shape [batch_size, context_length, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, context_length, head_dim] representing
            contextualised embeddings for each token in the context window.
        """
        query = self.query(embeddings)
        key = self.key(embeddings)
        value = self.value(embeddings)
        # All have shape [batch_size, context_length, head_dim].

        weights = query @ key.transpose(1, 2)
        # Shape: [batch_size, context_length, context_length]

        weights /= math.sqrt(self.head_dim)  # Reduce variance to balance softmax.

        # Causal mask: hide future tokens to prevent look-ahead.
        mask = torch.triu(
            torch.ones(
                (embeddings.shape[1], embeddings.shape[1]),
                dtype=torch.bool,
                device=embeddings.device,
            ),
            diagonal=1,
        )
        weights.masked_fill_(mask, float("-inf"))

        # Convert raw values to a probability distribution.
        weights = torch.softmax(weights, dim=-1)

        return weights @ value  # Shape: [batch_size, context_length, head_dim]


class MultiHeadAttention(nn.Module):
    """
    A class for running multiple attention heads in parallel.

    Args:
        embedding_dim: The number of dimensions of the space that the characters should
            be embedded in.
        num_of_heads: The number of distinct attention heads used in training the
            transformer language model.

    """

    def __init__(self, embedding_dim: int, num_of_heads: int) -> None:
        if embedding_dim % num_of_heads:
            raise ValueError(
                f"embedding_dim must be divisible by num_of_heads. "
                f"got: embedding_dim = {embedding_dim}, num_of_heads = {num_of_heads}"
            )

        super().__init__()

        head_dim = embedding_dim // num_of_heads
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_dim) for _ in range(num_of_heads)]
        )
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get a single tensor representing contextualised embeddings from multiple heads.

        Args:
            embeddings: A tensor of shape [batch_size, context_length, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, context_length, embedding_dim] representing
            the combined contextualised embeddings for each token in the context window.
        """
        x = [h(embeddings) for h in self.heads]
        # Shape of each element: [batch_size, context_length, head_dim]

        x = torch.cat(x, dim=-1)
        # Shape: [batch_size, context_length, embedding_dim]

        x = self.output(x)
        return x


class FeedForward(nn.Module):
    """
    A simple feed-forward neural network.

    Args:
        embedding_dim: The number of dimensions of the space that the characters should
            be embedded in.
        hidden_size: The number of neurons in the hidden layer.
    """

    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()

        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process the contextualised embeddings through a non-linear transformation.

        Args:
            embeddings: A tensor of shape [batch_size, context_length, embedding_dim]
            representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, context_length, embedding_dim] containing
            the transformed token embeddings.
        """
        x = self.hidden(embeddings)
        x = F.gelu(x)
        x = self.output(x)
        return x


class Block(nn.Module):
    """
    A single transformer block utilising a pre-norm residual architecture.

    The block consists of two sub-layers:
    1) Multi-head attention, where each token representation is updated using
        information from previous positions.
    2) A feed-forward network, where each token is individually passed through a
        non-linear transformation.
    Both sub-layers apply normalisation prior to computation, and residual connections
    are utilised to improve gradient stability during backpropagation.

    Args:
        embedding_dim: The number of dimensions of the space that the characters should
            be embedded in.
        num_of_heads: The number of attention heads in the multi-head attention part of
            the block.
        hidden_size: The number of neurons in the hidden layer of the feed-forward
            network section of the block.
    """

    def __init__(self, embedding_dim: int, num_of_heads: int, hidden_size: int) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_of_heads)
        self.feed_forward = FeedForward(embedding_dim, hidden_size)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process token embeddings through the attention and feed-forward sub-layers.

        Args:
            embeddings: A tensor of shape [batch_size, context_length, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, context_length, embedding_dim] containing
            the updated representations of the tokens.
        """
        # Pre-normalisation keeps activations on a more stable scale before attention.
        x = self.norm1(embeddings)
        x = self.multi_head_attention(x)

        # Update the existing representation rather than replacing it entirely,
        # to preserve the original information and improve gradient flow.
        residual = x + embeddings

        x = self.norm2(residual)
        x = self.feed_forward(x)
        residual = residual + x
        return residual


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
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            inputs, targets = get_batch(data, model.context_length, batch_size)
            logits = model(inputs)
            loss = loss_func(logits.flatten(0, 1), targets.flatten(0, 1))
            total_loss += loss.item()

    return total_loss / trials


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARN_RATE = 1e-3
    STEPS = 1000
    BATCH_SIZE = 256
    EVAL_TRIALS = 100
    EMBEDDING_DIM = 64
    CONTEXT_LENGTH = 64
    HIDDEN_SIZE = 256
    NUM_OF_HEADS = 4
    NUM_OF_BLOCKS = 8

    encoded_text = encode_text(original_text)
    cutoff = int(0.9 * len(encoded_text))
    train_tokens = encoded_text[:cutoff]
    val_tokens = encoded_text[cutoff:]
    train_data = torch.tensor(train_tokens, dtype=torch.long, device=DEVICE)
    val_data = torch.tensor(val_tokens, dtype=torch.long, device=DEVICE)

    vocab_size = len(char_set)

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

    s = "Marley was dead, to begin with. There is no doubt whatever about that. The register of his burial was signed by the clergyman"
    print("____Generated Text____")
    print(
        model.generate(
            torch.tensor(encode_text(s), dtype=torch.long, device=DEVICE),
            500,
        )
    )
