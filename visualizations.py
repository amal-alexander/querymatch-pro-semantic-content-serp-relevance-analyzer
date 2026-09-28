from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go


def similarity_heatmap(matrix: np.ndarray, labels: Sequence[str]) -> go.Figure:
    rounded = np.round(matrix, 3)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdYlGn",
            text=rounded,
            texttemplate="%{text:.3f}",
            hovertemplate="%{y} ↔ %{x}<br>Cosine similarity: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Cosine"),
        )
    )
    fig.update_layout(
        title="Cosine Similarity Matrix",
        xaxis_title="Query / Page",
        yaxis_title="Query / Page",
        height=max(520, 85 * len(labels)),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def euclidean_heatmap(matrix: np.ndarray, labels: Sequence[str]) -> go.Figure:
    rounded = np.round(matrix, 3)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Viridis_r",
            text=rounded,
            texttemplate="%{text:.3f}",
            hovertemplate="%{y} ↔ %{x}<br>Euclidean distance: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Distance"),
        )
    )
    fig.update_layout(
        title="Euclidean Distance Matrix",
        xaxis_title="Query / Page",
        yaxis_title="Query / Page",
        height=max(520, 85 * len(labels)),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig
