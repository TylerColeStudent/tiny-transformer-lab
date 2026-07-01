import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        context_length: The maximum number of tokens in the sequence used to predict
            the next token.
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
                Block(embedding_dim, num_of_heads, hidden_size, context_length)
                for _ in range(num_of_blocks)
            ]
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Get next-token logits for every position in the input token sequence.

        Args:
            input_tokens: A tensor of shape [batch_size, sequence_len] containing
                token IDs.

        Returns:
            A tensor of shape [batch_size, sequence_len, vocab_size] containing
            the next-token logits for every token in the sequence.

        Raises:
            ValueError: If the token sequences are longer than the model's maximum
                context length.
        """
        if input_tokens.shape[1] > self.context_length:
            raise ValueError(
                f"Token sequences must not be longer than {self.context_length} "
                f"tokens. Got sequence_len: {input_tokens.shape[1]}"
            )

        sequence_len = input_tokens.shape[1]

        token_embeddings = self.token_embedding(input_tokens)
        # Shape: [batch_size, sequence_len, embedding_dim]

        pos_indices = torch.arange(sequence_len, device=input_tokens.device)
        pos_embeddings = self.pos_embedding(pos_indices)
        # Shape: [sequence_len, embedding_dim]

        # Construct a single tensor combining semantic meaning with sequence position.
        x = token_embeddings + pos_embeddings
        # Shape: [batch_size, sequence_len, embedding_dim]

        for block in self.blocks:
            x = block(x)
        # Shape: [batch_size, sequence_len, embedding_dim]

        x = self.norm(x)
        logits = self.output(x)
        return logits  # Shape: [batch_size, sequence_len, vocab_size]

    def generate(self, context: torch.Tensor, length: int) -> list[int]:
        """
        Generate text using the model's learned logits.

        Args:
            context: A 1D tensor of token IDs which represent the initial context
                window.
            length: The number of additional tokens to generate

        Returns:
            A list of the initial context followed by the generated token IDs.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(length):
                current_context = context[-self.context_length :]

                logits = self(current_context.unsqueeze(0))

                next_token_logits = logits[:, -1, :]
                # Shape: [batch_size, vocab_size]

                # Convert logits to probabilities before sampling.
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)

                context = torch.cat([context, next_token.squeeze(dim=1)], dim=0)

        if was_training:
            self.train()
        return context.tolist()


class Head(nn.Module):
    """
    A class representing a single attention head for a transformer language model.

    It implements the Scaled Dot-Product Attention formula from the original
    "Attention is All You Need" paper.

    Args:
        embedding_dim: The number of dimensions of the space that the tokens should
            be embedded in.
        head_dim: The number of dimensions of space used for the attention queries,
            keys, and values.
        context_length: The maximum number of tokens in the sequence used to predict
            the next token.
    """

    mask: torch.Tensor

    def __init__(self, embedding_dim: int, head_dim: int, context_length: int) -> None:
        super().__init__()

        self.head_dim = head_dim

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones((context_length, context_length), dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Contextualise token embeddings with data from the current and preceding tokens.

        Args:
            embeddings: A tensor of shape [batch_size, sequence_len, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, sequence_len, head_dim] representing
            contextualised embeddings for each token in the context window.
        """
        sequence_len = embeddings.shape[1]
        query = self.query(embeddings)
        key = self.key(embeddings)
        value = self.value(embeddings)
        # All have shape [batch_size, sequence_len, head_dim].

        logits = query @ key.transpose(1, 2)
        # Shape: [batch_size, sequence_len, sequence_len]

        logits /= math.sqrt(self.head_dim)  # Reduce variance to balance softmax.

        # Causal mask: hide future tokens to prevent look-ahead.
        mask = self.mask[:sequence_len, :sequence_len]
        logits.masked_fill_(mask, float("-inf"))

        # Convert raw logits to a probability distribution.
        probs = torch.softmax(logits, dim=-1)

        return probs @ value  # Shape: [batch_size, sequence_len, head_dim]


class MultiHeadAttention(nn.Module):
    """
    A class for running multiple attention heads and combining their outputs.

    Args:
        embedding_dim: The number of dimensions of the space that the tokens should
            be embedded in.
        num_of_heads: The number of attention heads used in this attention layer.
        context_length: The maximum number of tokens in the sequence used to predict
            the next token.
    """

    def __init__(
        self, embedding_dim: int, num_of_heads: int, context_length: int
    ) -> None:
        if embedding_dim % num_of_heads:
            raise ValueError(
                f"embedding_dim must be divisible by num_of_heads. "
                f"got: embedding_dim = {embedding_dim}, num_of_heads = {num_of_heads}"
            )

        super().__init__()

        head_dim = embedding_dim // num_of_heads
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_dim, context_length) for _ in range(num_of_heads)]
        )
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get a single tensor representing contextualised embeddings from multiple heads.

        Args:
            embeddings: A tensor of shape [batch_size, sequence_len, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, sequence_len, embedding_dim] representing
            the combined contextualised embeddings for each token in the sequence.
        """
        x = [h(embeddings) for h in self.heads]
        # Shape of each element: [batch_size, sequence_len, head_dim]

        x = torch.cat(x, dim=-1)
        # Shape: [batch_size, sequence_len, embedding_dim]

        x = self.output(x)
        return x


class FeedForward(nn.Module):
    """
    A simple feed-forward neural network used inside a transformer block.

    Args:
        embedding_dim: The number of dimensions of the space that the tokens should
            be embedded in.
        hidden_size: The number of neurons in the hidden layer.
    """

    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()

        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply a non-linear transformation to each token embedding independently.

        Args:
            embeddings: A tensor of shape [batch_size, sequence_len, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, sequence_len, embedding_dim] containing
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
        information from the current and previous positions.
    2) A feed-forward network, where each token is individually passed through a
        non-linear transformation.
    Both sub-layers apply normalisation prior to computation, and residual connections
    are used to improve gradient stability during backpropagation.

    Args:
        embedding_dim: The number of dimensions of the space that the tokens should
            be embedded in.
        num_of_heads: The number of attention heads in the multi-head attention part of
            the block.
        hidden_size: The number of neurons in the hidden layer of the feed-forward
            network section of the block.
        context_length: The maximum number of tokens in the sequence used to predict
            the next token.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_of_heads: int,
        hidden_size: int,
        context_length: int,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim, num_of_heads, context_length
        )
        self.feed_forward = FeedForward(embedding_dim, hidden_size)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process token embeddings through the attention and feed-forward sub-layers.

        Args:
            embeddings: A tensor of shape [batch_size, sequence_len, embedding_dim]
                representing the tokens in a high-dimensionality space.

        Returns:
            A tensor of shape [batch_size, sequence_len, embedding_dim] containing
            the updated representations of the tokens.
        """
        # Pre-normalisation keeps activations on a more stable scale before attention.
        x = self.attention_norm(embeddings)
        x = self.multi_head_attention(x)

        # Update the existing representation rather than replacing it entirely,
        # to preserve the original information and improve gradient flow.
        residual = x + embeddings

        # Process each token embedding individually after normalising again.
        x = self.feed_forward_norm(residual)
        x = self.feed_forward(x)
        residual = residual + x
        return residual
