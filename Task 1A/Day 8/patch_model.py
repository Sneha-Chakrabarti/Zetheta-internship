"""TimesFM-style model: a decoder-only transformer over patches of a
series, trained to forecast the next patch's quantiles.

Follows TimesFM's shape (Section A5.3): the window is split into
fixed-length patches, each patch is linearly embedded, a causal
transformer runs over patches, and each position outputs quantile
forecasts for the following patch. Mean-pooled hidden states give the
embedding, and the quantile forecasts are the "quantile features" the
spec mentions.

Differences from Google's TimesFM, stated plainly: about 100k
parameters instead of 200M, pretrained on `corpus.py` for a short run
instead of ~100B real and synthetic time points, and no frequency
indicator. Nothing here is Google's model or weights.

Simplification: a window is normalised by its own mean and std over all
252 days during pretraining, so the scale of the patch being predicted
leaks into the normalisation. `forecast_next_patch` avoids this for
evaluation by normalising with the context only.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from .transformer import CausalBlock, PositionalEmbedding

QUANTILES = (0.1, 0.5, 0.9)


def normalize_windows(x: np.ndarray):
    """Per-window mean/std normalisation. Returns (normalised, mean, std)."""
    x = np.asarray(x, dtype="float64")
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-12
    return ((x - mean) / std).astype("float32"), mean, std


def to_patches(x: np.ndarray, patch_len: int) -> np.ndarray:
    b, length = x.shape
    assert length % patch_len == 0, "context length must be a multiple of patch_len"
    return x.reshape(b, length // patch_len, patch_len)


def build_mini_timesfm(context_len: int = 252, patch_len: int = 12, d_model: int = 64,
                        n_layers: int = 3, n_heads: int = 4, d_ff: int = 128,
                        quantiles=QUANTILES, seed: int = 0):
    """Returns (forecaster, encoder) sharing weights. forecaster maps
    patches (batch, n_patches, patch_len) to quantile forecasts of the
    NEXT patch, shape (batch, n_patches, n_quantiles, patch_len)."""
    tf.keras.utils.set_random_seed(seed)
    n_patches, nq = context_len // patch_len, len(quantiles)
    inp = layers.Input(shape=(n_patches, patch_len))
    x = layers.Dense(d_model)(inp)
    x = PositionalEmbedding(n_patches, d_model)(x)
    for _ in range(n_layers):
        x = CausalBlock(d_model, n_heads, d_ff)(x)
    hidden = layers.LayerNormalization(epsilon=1e-5)(x)
    out = layers.Dense(nq * patch_len)(hidden)
    out = layers.Reshape((n_patches, nq, patch_len))(out)
    return Model(inp, out), Model(inp, hidden)


def pinball_loss(y_true, y_pred, quantiles=QUANTILES):
    """y_true (b, n, patch_len); y_pred (b, n, nq, patch_len)."""
    q = tf.constant(quantiles, dtype=y_pred.dtype)[None, None, :, None]
    err = y_true[:, :, None, :] - y_pred
    return tf.reduce_mean(tf.maximum(q * err, (q - 1.0) * err))


def make_train_step(model, lr: float = 1e-3, quantiles=QUANTILES):
    opt = tf.keras.optimizers.Adam(lr, clipnorm=1.0)

    @tf.function
    def step(patches):
        with tf.GradientTape() as tape:
            pred = model(patches, training=True)
            loss = pinball_loss(patches[:, 1:, :], pred[:, :-1], quantiles)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return step, opt


def eval_pinball(model, patches: np.ndarray, batch: int = 128, quantiles=QUANTILES) -> float:
    total, n = 0.0, 0
    for i in range(0, len(patches), batch):
        p = tf.constant(patches[i:i + batch])
        pred = model(p, training=False)
        total += float(pinball_loss(p[:, 1:, :], pred[:, :-1], quantiles)) * int(p.shape[0])
        n += int(p.shape[0])
    return total / n


def embed_windows(encoder, patches: np.ndarray, batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(patches), batch):
        h = encoder(tf.constant(patches[i:i + batch]), training=False)
        out.append(tf.reduce_mean(h, axis=1).numpy())
    return np.vstack(out)


def forecast_next_patch(model, windows: np.ndarray, patch_len: int = 12, quantiles=QUANTILES):
    """Proper out-of-sample quantile forecast of the final `patch_len`
    days of each window from the preceding context only.

    windows: (b, context_len). The context is normalised with its own
    mean/std (no target leakage), the last patch is zero-padded (causal
    attention cannot see it), and the output at the last CONTEXT patch is
    the forecast. Returns (forecast in original units (b, nq, patch_len),
    actual (b, patch_len)), quantiles sorted to remove crossing.
    """
    windows = np.asarray(windows, dtype="float64")
    ctx, tgt = windows[:, :-patch_len], windows[:, -patch_len:]
    mean = ctx.mean(axis=1, keepdims=True)
    std = ctx.std(axis=1, keepdims=True) + 1e-12
    norm = ((windows - mean) / std).astype("float32")
    norm[:, -patch_len:] = 0.0
    pred = model(tf.constant(to_patches(norm, patch_len)), training=False).numpy()
    q = np.sort(pred[:, -2, :, :], axis=1)              # output at last context patch
    return q * std[:, :, None] + mean[:, :, None], tgt
