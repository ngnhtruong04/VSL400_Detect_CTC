"""Reconstructed HandsTCNTransformerCTC model for VSL400 Sign Language Recognition.

Contract:
- Input shape: (B, T, 294) representing T frames of 42 hand joints * 7 features [x, y, z, dx, dy, dz, valid]
- Output shape: (B, T, 401) representing logits for 400 Vietnamese Glosses + 1 CTC Blank (index 400)
- Trainable parameters: 5,442,961 (Exact match with architecture_lock.json)
"""

import torch
import torch.nn as nn
from typing import Optional


class TCNBlock(nn.Module):
    """Residual Temporal Convolutional Block with 2 Conv1d layers, BatchNorm, ReLU, and LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Symmetric padding preserves temporal length T
        padding = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, T, C)
        """
        res = x
        # Transpose to (B, C, T) for 1D convolutions
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        # Residual connection and layer normalization
        return self.norm(out + res)


class HandsTCNTransformerCTC(nn.Module):
    """Locked TCN-Transformer-CTC Architecture for Hands-Only VSL Recognition."""

    def __init__(
        self,
        input_dim: int = 294,
        model_dim: int = 256,
        tcn_layers: int = 4,
        tcn_kernel_size: int = 5,
        nhead: int = 8,
        transformer_layers: int = 4,
        ff_dim: int = 768,
        num_classes: int = 401,
        dropout: float = 0.1,
        blank_bias_init: float = -1.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes

        # Linear projection + LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # 4-layer dilated Temporal Convolutional Network
        self.tcn = nn.ModuleList([
            TCNBlock(
                in_channels=model_dim,
                out_channels=model_dim,
                kernel_size=tcn_kernel_size,
                dilation=2**i,
                dropout=dropout,
            )
            for i in range(tcn_layers)
        ])

        # 4-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
        )

        # Classifier head to 401 classes (400 Glosses + 1 Blank)
        self.classifier = nn.Linear(model_dim, num_classes)

        # Optional bias initialization for CTC blank (index num_classes - 1)
        if blank_bias_init is not None:
            nn.init.zeros_(self.classifier.bias)
            self.classifier.bias.data[-1] = blank_bias_init

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (B, T, input_dim)
            src_key_padding_mask: Optional mask of shape (B, T)
        Returns:
            Logits of shape (B, T, num_classes)
        """
        # Shape validation
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input shape (B, T, {self.input_dim}), got {tuple(x.shape)}"
            )

        # Linear projection
        x = self.input_proj(x)

        # Dilated TCN blocks
        for block in self.tcn:
            x = block(x)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Classifier logits
        logits = self.classifier(x)
        return logits

