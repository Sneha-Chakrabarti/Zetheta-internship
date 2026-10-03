"""Chronos-style model: a small token language model over time series.

Follows Chronos's recipe (Section A5.2): scale each series by its mean
absolute value, quantise into uniform bins, and train a transformer with
a cross-entropy next-token loss. Embeddings are the mean-pooled final
hidden states, the same "pool to a fixed vector" step as the spec's
`chronos_embed`.

Differences from Amazon's Chronos, stated plainly:

- Decoder-only causal transformer (~110k parameters), not the pretrained
  T5 encoder-decoder. Nothing here is Amazon's model or weights.
- Pretrained on `corpus.py` for a few thousand steps, not on a large
  real-world corpus.
- The scale is computed over the whole 252-day window, including
  positions the next-token loss then predicts. That leaks only
  magnitude, not order, into the pretext task and makes the pretraining
  loss slightly optimistic. It does not affect embedding extraction,
  which only ever sees one window.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from .transformer import CausalBlock, PositionalEmbedding

PAD, EOS, N_SPECIAL = 0, 1, 2


class MeanScaleTokenizer:
    def __init__(self, n_bins: int = 256, limit: float = 8.0):
        self.n_bins, self.limit = n_bins, limit

    @property
    def vocab_size(self) -> int:
        return self.n_bins + N_SPECIAL

    def encode(self, x: np.ndarray):
        """x: (batch, length). Returns (tokens int32 (batch, length), scale (batch, 1))."""
        x = np.asarray(x, dtype="float64")
        scale = np.abs(x).mean(axis=1, keepdims=True) + 1e-12
        xs = np.clip(x / scale, -self.limit, self.limit - 1e-6)
        idx = ((xs + self.limit) / (2 * self.limit) * self.n_bins).astype("int32")
        return idx + N_SPECIAL, scale


def build_mini_chronos(context_len: int = 252, n_bins: int = 256, d_model: int = 64,
                        n_layers: int = 2, n_heads: int = 4, d_ff: int = 128, seed: int = 0):
    """Returns (lm, encoder) sharing all weights: lm maps tokens to
    next-token logits, encoder maps tokens to per-position hidden states
    (batch, context_len, d_model)."""
    tf.keras.utils.set_random_seed(seed)
    vocab = n_bins + N_SPECIAL
    tokens = layers.Input(shape=(context_len,), dtype="int32")
    x = layers.Embedding(vocab, d_model)(tokens)
    x = PositionalEmbedding(context_len, d_model)(x)
    for _ in range(n_layers):
        x = CausalBlock(d_model, n_heads, d_ff)(x)
    hidden = layers.LayerNormalization(epsilon=1e-5)(x)
    logits = layers.Dense(vocab)(hidden)
    return Model(tokens, logits), Model(tokens, hidden)


def make_train_step(lm, lr: float = 1e-3):
    opt = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def step(tokens):
        with tf.GradientTape() as tape:
            logits = lm(tokens, training=True)
            loss = loss_fn(tokens[:, 1:], logits[:, :-1, :])
        grads = tape.gradient(loss, lm.trainable_variables)
        opt.apply_gradients(zip(grads, lm.trainable_variables))
        return loss

    return step, opt


def lm_cross_entropy(lm, tokens: np.ndarray, batch: int = 64) -> float:
    """Mean next-token cross-entropy in nats (no gradient)."""
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum")
    total, count = 0.0, 0
    for i in range(0, len(tokens), batch):
        t = tf.constant(tokens[i:i + batch])
        logits = lm(t, training=False)
        total += float(loss_fn(t[:, 1:], logits[:, :-1, :]))
        count += int(t.shape[0]) * (t.shape[1] - 1)
    return total / count


def unigram_entropy(tokens: np.ndarray, vocab: int) -> float:
    """Cross-entropy of the best STATIC token distribution: the
    no-temporal-structure baseline the language model must beat."""
    counts = np.bincount(tokens[:, 1:].ravel(), minlength=vocab).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def embed_windows(encoder, tokens: np.ndarray, batch: int = 128) -> np.ndarray:
    """Mean-pooled final hidden states, one d_model vector per window."""
    out = []
    for i in range(0, len(tokens), batch):
        h = encoder(tf.constant(tokens[i:i + batch]), training=False)
        out.append(tf.reduce_mean(h, axis=1).numpy())
    return np.vstack(out)
