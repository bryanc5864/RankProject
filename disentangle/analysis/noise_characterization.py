"""
Phase 2: Noise Characterization.

Train separate models on different experiments, compare representations
to demonstrate that current models encode experiment-specific noise.

Key analyses:
- CKA (Centered Kernel Alignment) between model representations
- UMAP visualization colored by experiment
- Paired-experiment representation divergence

Produces Figures 1A and 1B for the paper.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist


def extract_representations(model, sequences: np.ndarray,
                           batch_size: int = 256) -> np.ndarray:
    """Extract penultimate-layer representations from a model."""
    model.eval()
    representations = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[i:i + batch_size], dtype=torch.float32
            ).cuda()
            if hasattr(model, "encode"):
                features = model.encode(batch)
            else:
                features = model.base_model.encode(batch)

            if features.dim() == 3:
                features = features.mean(dim=1)

            representations.append(features.cpu().numpy())

    return np.concatenate(representations, axis=0)


# ---- CKA Computation ----

def linear_kernel(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def rbf_kernel_matrix(X: np.ndarray, sigma: float = None) -> np.ndarray:
    if sigma is None:
        sigma = np.median(pdist(X, "euclidean"))
    return rbf_kernel(X, gamma=1.0 / (2 * sigma**2))


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def compute_cka(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """
    Centered Kernel Alignment between two representation matrices.

    Args:
        X: [N, D1] representations from model 1
        Y: [N, D2] representations from model 2
        kernel: 'linear' or 'rbf'

    Returns:
        CKA value in [0, 1]. Higher = more similar representations.
    """
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    else:
        K = rbf_kernel_matrix(X)
        L = rbf_kernel_matrix(Y)

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


# ---- Visualization ----

def plot_umap_by_experiment(representations_dict: dict, output_path: str):
    """
    UMAP visualization colored by which experiment's model produced the representations.

    Args:
        representations_dict: {model_name: np.array[N, D]} for the SAME sequences
        output_path: path to save the figure

    Produces Figure 1A: if models learn biology, same-sequence points cluster together.
    If they learn noise, points cluster by model.
    """
    import matplotlib.pyplot as plt
    import umap

    all_reps = []
    labels = []
    for name, reps in representations_dict.items():
        all_reps.append(reps)
        labels.extend([name] * len(reps))

    all_reps = np.concatenate(all_reps, axis=0)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(all_reps)

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = list(representations_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[i]], label=label, alpha=0.5, s=10,
        )

    ax.legend(fontsize=12)
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_title(
        "Representations of Same Sequences from Models\n"
        "Trained on Different Experiments",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP plot to {output_path}")


def plot_cka_matrix(cka_matrix: np.ndarray, labels: list[str], output_path: str):
    """Plot CKA heatmap (Figure 1B)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cka_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("CKA Similarity Between Models\nTrained on Different Experiments")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CKA matrix to {output_path}")
