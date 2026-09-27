import numpy as np
import pandas as pd
import pytest
import torch

from src.data.loader import DataConfig, load_market_data
from src.features.engineer import engineer_regime_features, FEATURE_GROUPS
from src.features.topology import build_return_panel, rolling_correlation_matrices, topology_feature_frame
from src.features.sector_gnn import SectorGCN, build_graph_edges, build_node_features, sector_gnn_embeddings

DATA = load_market_data(DataConfig(backend="synthetic", n_years=3.0, seed=11))


def test_engineer_regime_features_shape_and_no_nans():
    feats = engineer_regime_features(DATA)
    expected_cols = sum(len(v) for v in FEATURE_GROUPS.values())
    assert feats.shape[1] == expected_cols
    assert not feats.isna().any().any()
    assert set(feats.columns) == {c for group in FEATURE_GROUPS.values() for c in group}


def test_engineer_regime_features_above_200dma_is_binary():
    feats = engineer_regime_features(DATA)
    assert set(feats["above_200dma"].unique()).issubset({0, 1})


def test_topology_feature_frame_shape_and_finite():
    tda = topology_feature_frame(DATA, window=63, step=10)
    assert tda.shape[0] > 0
    assert tda.shape[1] == 5 * 100 * 2  # n_layers * n_bins * homology_dims
    assert np.isfinite(tda.values).all()


def test_rolling_correlation_matrices_are_valid_correlation_matrices():
    panel = build_return_panel(DATA)
    matrices, dates = rolling_correlation_matrices(panel, window=63, step=20)
    assert matrices.shape[1] == matrices.shape[2] == panel.shape[1]
    # Diagonal of a correlation matrix is always 1.
    for m in matrices:
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-8)
    assert len(dates) == matrices.shape[0]


def test_sector_gcn_forward_pass_shape():
    torch.manual_seed(0)
    model = SectorGCN(in_dim=2, hidden=8, out_dim=4)
    x = torch.randn(9, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (4,)


def test_build_graph_edges_handles_no_edges_above_threshold():
    panel = build_return_panel(DATA)
    end_date = panel.index[100]
    edge_index, edge_weight = build_graph_edges(panel, end_date, window=63, corr_threshold=1.5)
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)


def test_sector_gnn_embeddings_reproducible_with_fixed_seed():
    emb1 = sector_gnn_embeddings(DATA, window=63, step=15, out_dim=8, seed=42)
    emb2 = sector_gnn_embeddings(DATA, window=63, step=15, out_dim=8, seed=42)
    pd.testing.assert_frame_equal(emb1, emb2)
    assert np.isfinite(emb1.values).all()
