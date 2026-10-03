"""Small causal transformer building blocks (TensorFlow / tf-keras) shared
by the Chronos-style and TimesFM-style models. Pre-layer-norm blocks,
learned positional embeddings, causal self-attention.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, length: int, d_model: int, **kw):
        super().__init__(**kw)
        self.length, self.d_model = length, d_model
        self.emb = layers.Embedding(length, d_model)

    def call(self, x):
        return x + self.emb(tf.range(self.length))[None, :, :]

    def get_config(self):
        return {**super().get_config(), "length": self.length, "d_model": self.d_model}


class CausalBlock(layers.Layer):
    """x + MHA(LN(x)) then x + FFN(LN(x)), attention masked causally."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ff1 = layers.Dense(d_ff, activation="gelu")
        self.ff2 = layers.Dense(d_model)

    def call(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, use_causal_mask=True)
        return x + self.ff2(self.ff1(self.ln2(x)))

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff}
