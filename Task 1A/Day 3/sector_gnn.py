"""Graph-structure embeddings via a lightweight GCN, Section A7.3.

Section A7.3 treats "the universe of stocks/sectors" as a graph. We have
neither individual stocks nor NSE sectoral indices (Nifty IT, Nifty Bank,
Nifty Auto and similar) in the required data panel (Section A1) - only
three cap-segment indices plus six macro/flow series. The graph built
here reuses `src.features.topology.build_return_panel`'s nine-series
panel as the node set: a real, if smaller and differently-composed, graph
rather than a sector graph. Wiring in genuine NSE sectoral indices later
means adding those series to the data schema and pointing this module at
them; the GCN architecture and the rest of this module do not change.

`SectorGCN`'s weights are fixed at construction (a seeded random init,
never trained here). That mirrors how foundation-model embeddings are
used in Section A5: a frozen feature extractor, not an end-to-end trained
component. End-to-end fine-tuning of this GCN jointly with the regime
classifier is deferred to the model-training work (Day 9 ensembling),
once there is a supervised objective to train it against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .topology import build_return_panel, TOPOLOGY_SERIES

NODE_FEATURE_DIM = 2  # [z-scored 5-day change, z-scored 21-day realised vol]


class SectorGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x.mean(dim=0)  # graph-level embedding for the day


def build_graph_edges(panel: pd.DataFrame, end_date, window: int = 63,
                       corr_threshold: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """Thresholded correlation graph over `panel`'s columns for the
    `window`-day period ending at `end_date`: an edge exists where
    |correlation| exceeds `corr_threshold`, weighted by |correlation|.
    Self-loops are excluded; GCNConv adds its own."""
    corr = panel.loc[:end_date].tail(window).corr().values
    n = corr.shape[0]
    src, dst, weight = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(corr[i, j]) > corr_threshold:
                src.append(i)
                dst.append(j)
                weight.append(abs(corr[i, j]))
    if not src:
        # No edge cleared the threshold this window: fall back to a fully
        # disconnected graph (GCNConv still runs; it degrades to per-node
        # self-transform only, which is the correct behaviour when the
        # data genuinely shows no structure that window).
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float32)
    return edge_index, edge_weight


def build_node_features(panel: pd.DataFrame, end_date) -> torch.Tensor:
    """Per-node feature vector as of `end_date`: z-scored 5-day change and
    z-scored 21-day realised vol of each of the nine panel series, using
    only history up to `end_date` for both the value and the z-score
    baseline (no look-ahead)."""
    history = panel.loc[:end_date]
    chg_5d = history.diff(5).iloc[-1]
    chg_5d_z = (chg_5d - history.diff(5).mean()) / history.diff(5).std()
    vol_21d = history.diff().rolling(21).std().iloc[-1]
    vol_21d_z = (vol_21d - history.diff().rolling(21).std().mean()) / history.diff().rolling(21).std().std()
    x = np.stack([chg_5d_z.values, vol_21d_z.values], axis=1)
    x = np.nan_to_num(x, nan=0.0)
    return torch.tensor(x, dtype=torch.float32)


def sector_gnn_embeddings(data: dict[str, pd.DataFrame], window: int = 63,
                           step: int = 5, corr_threshold: float = 0.3,
                           out_dim: int = 32, seed: int = 13) -> pd.DataFrame:
    """End-to-end: build the nine-series panel, and for every `step`-th day
    from `window` days in, construct that day's graph and node features
    and run them through a fixed-weight SectorGCN. Returns one embedding
    row per sampled date. Column names are `gcn_0000` style: individual
    dimensions of a random-projection embedding carry no standalone
    meaning, same convention as the TDA landscape columns.
    """
    panel = build_return_panel(data)
    dates = panel.index[window - 1 :: step]

    torch.manual_seed(seed)
    model = SectorGCN(in_dim=NODE_FEATURE_DIM, hidden=64, out_dim=out_dim)
    model.eval()

    rows = []
    with torch.no_grad():
        for end_date in dates:
            edge_index, edge_weight = build_graph_edges(panel, end_date, window=window,
                                                         corr_threshold=corr_threshold)
            x = build_node_features(panel, end_date)
            embedding = model(x, edge_index, edge_weight if edge_weight.numel() else None)
            rows.append(embedding.numpy())

    columns = [f"gcn_{i:04d}" for i in range(out_dim)]
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(dates), columns=columns)
