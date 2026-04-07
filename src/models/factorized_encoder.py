"""
Factorized Encoder for Multi-Scale Representation Decomposition

Decomposes sequence representations into:
1. z_motif: Local motif patterns (small kernel convolutions)
2. z_grammar: Motif arrangement/grammar (dilated convolutions)
3. z_composition: Global sequence composition (GC%, etc.)

Includes:
- Variational Information Bottleneck (VIB) on composition
- Adversarial GC deconfounding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dream_rnn import GradientReversalLayer


class MotifBranch(nn.Module):
    """
    Local motif detection branch.

    Uses small kernel convolutions (8bp) without dilation
    to capture local regulatory motifs like TF binding sites.
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 128,
                 output_dim: int = 64, kernel_size: int = 8, dropout: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size,
                               padding='same')
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                               padding='same')
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.pool = nn.AdaptiveMaxPool1d(1)  # Max pool to detect strongest motif
        self.fc = nn.Linear(hidden_channels, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]

        Returns:
            z_motif: Local motif representation [batch_size, output_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class GrammarBranch(nn.Module):
    """
    Motif arrangement/grammar branch.

    Uses dilated convolutions to capture spatial relationships
    between motifs (e.g., spacing, orientation patterns).
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 128,
                 output_dim: int = 64, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()

        # Increasing dilation rates to capture multi-scale dependencies
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size,
                               dilation=1, padding='same')
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                               dilation=2, padding='same')
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                               dilation=4, padding='same')
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]

        Returns:
            z_grammar: Grammar representation [batch_size, output_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class CompositionBranch(nn.Module):
    """
    Global composition branch.

    Captures sequence-level features like GC content,
    dinucleotide frequencies, etc. using global pooling.
    """

    def __init__(self, in_channels: int = 4, output_dim: int = 32):
        super().__init__()

        # Simple architecture: just pool and project
        # Captures overall nucleotide frequencies
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]

        Returns:
            z_composition: Composition representation [batch_size, output_dim]
        """
        x = self.pool(x).squeeze(-1)  # [batch, 4]
        return self.fc(x)


class VIBComposition(nn.Module):
    """
    Variational Information Bottleneck for composition branch.

    Compresses composition features through a stochastic bottleneck,
    reducing information that is purely compositional (noise-correlated).
    """

    def __init__(self, in_channels: int = 4, latent_dim: int = 16, beta: float = 0.01):
        """
        Args:
            in_channels: Input channels (4 for DNA)
            latent_dim: Dimension of bottleneck
            beta: KL divergence weight (higher = more compression)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Encoder to (mu, log_var)
        self.fc_mu = nn.Linear(in_channels, latent_dim)
        self.fc_log_var = nn.Linear(in_channels, latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at inference

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]

        Returns:
            (z, kl_loss): z is sampled latent, kl_loss is KL divergence
        """
        x = self.pool(x).squeeze(-1)  # [batch, 4]

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        z = self.reparameterize(mu, log_var)

        # KL divergence with standard normal
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return z, self.beta * kl_loss


class GCAdversary(nn.Module):
    """
    Adversarial classifier that predicts GC content.

    Used with gradient reversal to ensure motif/grammar branches
    don't encode GC information (deconfounding).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_bins: int = 10):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer size
            n_bins: Number of GC content bins (for classification)
        """
        super().__init__()
        self.n_bins = n_bins

        self.gradient_reversal = GradientReversalLayer(lambda_=1.0)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bins)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            GC bin logits [batch_size, n_bins]
        """
        reversed_features = self.gradient_reversal(features)
        return self.classifier(reversed_features)

    def set_lambda(self, lambda_: float):
        """Set gradient reversal strength."""
        self.gradient_reversal.set_lambda(lambda_)


class FactorizedEncoder(nn.Module):
    """
    Multi-scale factorized encoder with separate prediction branches.

    Decomposes representations into:
    - z_motif: Local motif patterns
    - z_grammar: Motif arrangement
    - z_composition: Global composition

    Each branch has its own prediction head, allowing analysis
    of what each representation captures.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 motif_dim: int = 64, grammar_dim: int = 64,
                 composition_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        self.motif_branch = MotifBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=motif_dim,
            dropout=dropout
        )

        self.grammar_branch = GrammarBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=grammar_dim,
            dropout=dropout
        )

        self.composition_branch = CompositionBranch(
            in_channels=in_channels,
            output_dim=composition_dim
        )

        total_dim = motif_dim + grammar_dim + composition_dim

        # Separate prediction heads
        self.motif_head = nn.Linear(motif_dim, 1)
        self.grammar_head = nn.Linear(grammar_dim, 1)
        self.composition_head = nn.Linear(composition_dim, 1)

        # Combined prediction
        self.combined_head = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_components: bool = False):
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]
            return_components: If True, return all components

        Returns:
            If return_components:
                (activity, motif_pred, grammar_pred, comp_pred, z_motif, z_grammar, z_comp)
            Else:
                activity
        """
        z_motif = self.motif_branch(x)
        z_grammar = self.grammar_branch(x)
        z_composition = self.composition_branch(x)

        # Individual predictions
        motif_pred = self.motif_head(z_motif).squeeze(-1)
        grammar_pred = self.grammar_head(z_grammar).squeeze(-1)
        comp_pred = self.composition_head(z_composition).squeeze(-1)

        # Combined prediction
        z_combined = torch.cat([z_motif, z_grammar, z_composition], dim=1)
        activity = self.combined_head(z_combined).squeeze(-1)

        if return_components:
            return activity, motif_pred, grammar_pred, comp_pred, z_motif, z_grammar, z_composition

        return activity

    def get_embeddings(self, x: torch.Tensor) -> dict:
        """Get all branch embeddings."""
        z_motif = self.motif_branch(x)
        z_grammar = self.grammar_branch(x)
        z_composition = self.composition_branch(x)

        return {
            'motif': z_motif,
            'grammar': z_grammar,
            'composition': z_composition,
            'combined': torch.cat([z_motif, z_grammar, z_composition], dim=1)
        }


class FactorizedEncoderVIB(nn.Module):
    """
    Factorized encoder with VIB on composition branch.

    The VIB bottleneck compresses composition features,
    reducing the influence of noise-correlated compositional biases.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 motif_dim: int = 64, grammar_dim: int = 64,
                 composition_dim: int = 16, dropout: float = 0.2,
                 vib_beta: float = 0.01):
        super().__init__()

        self.motif_branch = MotifBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=motif_dim,
            dropout=dropout
        )

        self.grammar_branch = GrammarBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=grammar_dim,
            dropout=dropout
        )

        self.composition_branch = VIBComposition(
            in_channels=in_channels,
            latent_dim=composition_dim,
            beta=vib_beta
        )

        total_dim = motif_dim + grammar_dim + composition_dim

        self.combined_head = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_kl: bool = True):
        """
        Args:
            x: Input sequences
            return_kl: If True, return KL divergence loss

        Returns:
            (activity, kl_loss) or just activity
        """
        z_motif = self.motif_branch(x)
        z_grammar = self.grammar_branch(x)
        z_composition, kl_loss = self.composition_branch(x)

        z_combined = torch.cat([z_motif, z_grammar, z_composition], dim=1)
        activity = self.combined_head(z_combined).squeeze(-1)

        if return_kl:
            return activity, kl_loss
        return activity


class FactorizedEncoderGCAdv(nn.Module):
    """
    Factorized encoder with adversarial GC deconfounding.

    Uses gradient reversal to ensure motif and grammar branches
    cannot predict GC content, preventing compositional shortcuts.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 motif_dim: int = 64, grammar_dim: int = 64,
                 composition_dim: int = 32, dropout: float = 0.2,
                 gc_bins: int = 10):
        super().__init__()

        self.motif_branch = MotifBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=motif_dim,
            dropout=dropout
        )

        self.grammar_branch = GrammarBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=grammar_dim,
            dropout=dropout
        )

        self.composition_branch = CompositionBranch(
            in_channels=in_channels,
            output_dim=composition_dim
        )

        # GC adversary on motif + grammar (not composition)
        self.gc_adversary = GCAdversary(
            input_dim=motif_dim + grammar_dim,
            hidden_dim=32,
            n_bins=gc_bins
        )

        total_dim = motif_dim + grammar_dim + composition_dim

        self.combined_head = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_gc_logits: bool = True):
        """
        Args:
            x: Input sequences
            return_gc_logits: If True, return GC classification logits

        Returns:
            (activity, gc_logits) or just activity
        """
        z_motif = self.motif_branch(x)
        z_grammar = self.grammar_branch(x)
        z_composition = self.composition_branch(x)

        z_combined = torch.cat([z_motif, z_grammar, z_composition], dim=1)
        activity = self.combined_head(z_combined).squeeze(-1)

        if return_gc_logits:
            # Adversarial GC prediction from motif + grammar
            z_for_gc = torch.cat([z_motif, z_grammar], dim=1)
            gc_logits = self.gc_adversary(z_for_gc)
            return activity, gc_logits

        return activity

    def set_adversary_lambda(self, lambda_: float):
        """Set gradient reversal strength for GC adversary."""
        self.gc_adversary.set_lambda(lambda_)

    @staticmethod
    def compute_gc_labels(x: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
        """
        Compute GC content bin labels for adversarial training.

        Args:
            x: One-hot sequences [batch, 4, seq_len]
            n_bins: Number of bins

        Returns:
            GC bin labels [batch]
        """
        # GC content = (G + C) / seq_len
        # Assuming encoding: A=0, C=1, G=2, T=3
        gc_count = x[:, 1, :].sum(dim=1) + x[:, 2, :].sum(dim=1)
        seq_len = x.shape[2]
        gc_content = gc_count / seq_len

        # Bin into n_bins categories
        gc_labels = (gc_content * n_bins).long().clamp(0, n_bins - 1)
        return gc_labels


class FactorizedEncoderFull(nn.Module):
    """
    Full factorized encoder with both VIB and GC adversarial.

    Combines all deconfounding techniques:
    - VIB on composition (information bottleneck)
    - GC adversarial on motif/grammar (gradient reversal)
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 motif_dim: int = 64, grammar_dim: int = 64,
                 composition_dim: int = 16, dropout: float = 0.2,
                 vib_beta: float = 0.01, gc_bins: int = 10):
        super().__init__()

        self.motif_branch = MotifBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=motif_dim,
            dropout=dropout
        )

        self.grammar_branch = GrammarBranch(
            in_channels=in_channels,
            hidden_channels=128,
            output_dim=grammar_dim,
            dropout=dropout
        )

        self.composition_branch = VIBComposition(
            in_channels=in_channels,
            latent_dim=composition_dim,
            beta=vib_beta
        )

        self.gc_adversary = GCAdversary(
            input_dim=motif_dim + grammar_dim,
            hidden_dim=32,
            n_bins=gc_bins
        )

        total_dim = motif_dim + grammar_dim + composition_dim

        self.combined_head = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_all: bool = True):
        """
        Args:
            x: Input sequences
            return_all: If True, return all auxiliary outputs

        Returns:
            If return_all: (activity, kl_loss, gc_logits)
            Else: activity
        """
        z_motif = self.motif_branch(x)
        z_grammar = self.grammar_branch(x)
        z_composition, kl_loss = self.composition_branch(x)

        z_combined = torch.cat([z_motif, z_grammar, z_composition], dim=1)
        activity = self.combined_head(z_combined).squeeze(-1)

        if return_all:
            z_for_gc = torch.cat([z_motif, z_grammar], dim=1)
            gc_logits = self.gc_adversary(z_for_gc)
            return activity, kl_loss, gc_logits

        return activity

    def set_adversary_lambda(self, lambda_: float):
        """Set gradient reversal strength."""
        self.gc_adversary.set_lambda(lambda_)
