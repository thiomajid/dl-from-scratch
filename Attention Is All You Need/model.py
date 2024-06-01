import math

import torch
from torch import nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape seq_len x d_model.
        # positional encoding shape must match embeddings shape
        pe = torch.zeros((seq_len, d_model))

        # words position vector
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # the div term for sin and cosine PE
        # Use exp here => think of 2^n exponential form
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1000) / d_model)
        )

        # Apply sin to even positions
        # Here a multiplication because the div_term uses a negative log, which turns out to be a fraction
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding a batch_dim => (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # mul
        self.bias = nn.Parameter(torch.zeros(1))  # add

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)

        return self.alpha * x_norm + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_prob: float):
        super().__init__()

        self.lin_1 = nn.Linear(d_model, d_ff)
        self.lin_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.lin_1(x))
        return self.lin_2(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_prob: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"

        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # (batch, seq_len, d_model) => (batch, seq_len, d_model)
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
