"""
Hybrid CNN-Transformer architecture for chess AI.

Combines:
1. CNN Stem with SE attention - local pattern recognition
2. Lightweight Transformer - global strategic reasoning
3. Dual output heads - policy and value
"""

import logging
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE block.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for squeeze operation
        """
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of same shape
        """
        b, c, h, w = x.size()
        
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        
        # Excitation: learn channel-wise importance
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale input by excitation weights
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual block with optional SE attention.
    """
    
    def __init__(self, channels: int, use_se: bool = True, reduction: int = 16):
        """
        Initialize residual block.
        
        Args:
            channels: Number of input/output channels
            use_se: Whether to use SE attention
            reduction: SE reduction ratio
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of same shape
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # Apply SE attention
        
        out += residual  # Residual connection
        out = self.relu(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and MLP.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            dropout: Dropout probability
        """
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, D) where N is sequence length, D is embed_dim
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class HybridChessNet(nn.Module):
    """
    Hybrid CNN-Transformer architecture for chess.
    
    Architecture:
    1. CNN Stem: 8 residual blocks with SE attention
    2. Transformer: 4 transformer blocks on downsampled features
    3. Policy Head: Upsampled features to 4096 move predictions
    4. Value Head: Global pooling to scalar value
    """
    
    def __init__(
        self,
        input_channels: int = 33,
        cnn_channels: int = 256,
        num_res_blocks: int = 8,
        transformer_embed_dim: int = 512,
        transformer_num_heads: int = 8,
        transformer_num_blocks: int = 4,
        transformer_mlp_ratio: int = 4,
        num_moves: int = 4096,
        dropout: float = 0.1
    ):
        """
        Initialize hybrid chess network.
        
        Args:
            input_channels: Number of input planes (33 for chess board encoding)
            cnn_channels: Number of channels in CNN stem
            num_res_blocks: Number of residual blocks in CNN stem
            transformer_embed_dim: Embedding dimension for transformer
            transformer_num_heads: Number of attention heads
            transformer_num_blocks: Number of transformer blocks
            transformer_mlp_ratio: MLP expansion ratio
            num_moves: Number of possible moves (4096)
            dropout: Dropout probability
        """
        super(HybridChessNet, self).__init__()
        
        self.input_channels = input_channels
        self.cnn_channels = cnn_channels
        self.transformer_embed_dim = transformer_embed_dim
        self.num_moves = num_moves
        
        # CNN Stem
        self.conv_stem = nn.Conv2d(
            input_channels, cnn_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn_stem = nn.BatchNorm2d(cnn_channels)
        self.relu_stem = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(cnn_channels, use_se=True)
            for _ in range(num_res_blocks)
        ])
        
        # Downsample for transformer (8x8 -> 4x4)
        self.downsample = nn.Conv2d(
            cnn_channels, transformer_embed_dim,
            kernel_size=2, stride=2, bias=False
        )
        self.bn_downsample = nn.BatchNorm2d(transformer_embed_dim)
        
        # Positional embedding (for 4x4 = 16 positions)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, transformer_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                transformer_embed_dim,
                transformer_num_heads,
                transformer_mlp_ratio,
                dropout
            )
            for _ in range(transformer_num_blocks)
        ])
        
        # Policy Head
        # Upsample back to 8x8
        self.policy_upsample = nn.Sequential(
            nn.ConvTranspose2d(transformer_embed_dim, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Global features for policy (from transformer output)
        self.policy_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(transformer_embed_dim, 256),
            nn.ReLU(inplace=True)
        )
        
        # Final policy output
        self.policy_fc = nn.Sequential(
            nn.Linear(32 * 8 * 8 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_moves)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(transformer_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Tanh()  # Output range: [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 33, 8, 8)
            
        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (B, 4096) unnormalized log probabilities
            - value: (B, 1) position evaluation in range [-1, 1]
        """
        # CNN Stem
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.relu_stem(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Downsample for transformer
        x_transformer = self.downsample(x)
        x_transformer = self.bn_downsample(x_transformer)
        x_transformer = F.relu(x_transformer)
        
        # Flatten spatial dimensions: (B, D, 4, 4) -> (B, 16, D)
        B, D, H, W = x_transformer.shape
        x_transformer = x_transformer.flatten(2).transpose(1, 2)  # (B, 16, D)
        
        # Add positional embedding
        x_transformer = x_transformer + self.pos_embed
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x_transformer = transformer_block(x_transformer)
        
        # Reshape back to spatial: (B, 16, D) -> (B, D, 4, 4)
        x_transformer = x_transformer.transpose(1, 2).reshape(B, D, 4, 4)
        
        # Policy Head
        policy_spatial = self.policy_upsample(x_transformer)  # (B, 32, 8, 8)
        policy_spatial_flat = policy_spatial.flatten(1)  # (B, 32*8*8)
        
        # Global features
        policy_global = self.policy_global(x_transformer)  # (B, 256)
        
        # Concatenate and output
        policy_features = torch.cat([policy_spatial_flat, policy_global], dim=1)
        policy_logits = self.policy_fc(policy_features)  # (B, 4096)
        
        # Value Head
        value = self.value_head(x_transformer)  # (B, 1)
        
        return policy_logits, value
    
    def get_policy(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get policy distribution (softmax over moves).
        
        Args:
            x: Input tensor
            temperature: Temperature for softmax (1.0 = normal, >1.0 = softer)
            
        Returns:
            Policy distribution of shape (B, 4096)
        """
        policy_logits, _ = self.forward(x)
        return F.softmax(policy_logits / temperature, dim=-1)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test architecture
    model = HybridChessNet(
        input_channels=33,
        cnn_channels=256,
        num_res_blocks=8,
        transformer_embed_dim=512,
        transformer_num_heads=8,
        transformer_num_blocks=4
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 33, 8, 8)
    policy, value = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")
    print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")
