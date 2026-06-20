# Tiny Transformer Lab

Building a small GPT-style language model from first principles in Python.

Current progress:
- Character-level tokeniser
- Encode/decode text as token IDs
- Count-based bigram text generator
- Generalised to an n-gram text generator
- Weighted random next-character sampling
- Built a PyTorch bigram model trained with cross-entropy loss and manual gradient descent.

The current training text is "A Christmas Carol" by Charles Dickens, sourced from Standard Ebooks.