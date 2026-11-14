"""Define the non-distilled DeiT-Tiny used by all FairAge conditions.

Build the 224/16 transformer from fundamental PyTorch layers, with twelve
width-192 blocks, three attention heads, and nine age-category logits.

Main classes:
    Attention: Project Q/K/V and apply scaled dot-product attention.
    Block: Apply pre-LayerNorm attention and MLP residual branches.
    DeiTTiny: Embed image patches and classify the final CLS token.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Apply three-head scaled dot-product attention to 197 width-192 tokens."""

    def __init__(self, dropout=0.0):
        """Create the fixed layers; baseline dropout is zero."""

        super().__init__()
        self.qkv = nn.Linear(192, 576)
        self.proj = nn.Linear(192, 192)
        self.dropout = dropout

    def forward(self, tokens):
        """Map [B,N,192] tokens to [B,N,192] mixed token features.

        One projection produces Q/K/V, each [B,3,N,64] after splitting heads.
        SDPA computes softmax(QK^T / sqrt(64))V over the N key positions;
        it applies the scale internally. Dropout is disabled in evaluation.
        Transpose/reshape rejoins the three heads before the output projection.
        """

        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, 3, 64)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    """Use pre-normalization, attention and a GELU MLP with residual additions."""

    def __init__(self, dropout=0.0):
        """Create the fixed layers; baseline dropout is zero."""

        super().__init__()
        self.norm1 = nn.LayerNorm(192, eps=1e-6)
        self.attn = Attention(dropout)
        self.norm2 = nn.LayerNorm(192, eps=1e-6)
        self.fc1 = nn.Linear(192, 768)
        self.fc2 = nn.Linear(768, 192)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        """Preserve [B,N,192] through two pre-normalized residual branches.

        Attention first mixes token positions; the 192→768→192 GELU MLP then
        transforms each token's features. Each branch is added to its input,
        keeping a direct information/gradient path through the block.
        """

        tokens = tokens + self.dropout(self.attn(self.norm1(tokens)))
        hidden = self.dropout(F.gelu(self.fc1(self.norm2(tokens))))
        return tokens + self.dropout(self.fc2(hidden))


class DeiTTiny(nn.Module):
    """Non-distilled 224/16 DeiT: 12 blocks, width 192, three heads, nine classes."""

    def __init__(self, dropout=0.0):
        """Create scratch weights; pretrained loading later replaces the backbone.

        Linear/position weights use std .02, CLS uses std 1e-6, and Linear
        biases are zero. Conv2d and LayerNorm keep PyTorch initialization.
        The same fresh nine-class head is retained when loading ImageNet.
        """

        super().__init__()
        self.patch_embed = nn.Conv2d(3, 192, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.empty(1, 197, 192))
        self.blocks = nn.ModuleList([Block(dropout) for _ in range(12)])
        self.norm = nn.LayerNorm(192, eps=1e-6)
        self.head = nn.Linear(192, 9)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)

    def forward_features(self, images):
        """Map normalized [B,3,224,224] RGB images to [B,197,192] tokens.

        Conv2d projects each non-overlapping 16×16 patch to 192 features;
        flatten/transpose preserves the 14×14 grid's order as 196 tokens.
        CLS is position zero. Learned positions precede 12 independent blocks
        and a final LayerNorm; preprocessing belongs to fairage.data.
        """

        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("Expected RGB images shaped [batch,3,224,224]")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(len(images), -1, -1), patches), 1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, images):
        """Return [B,9] age logits from CLS; cross-entropy expects these scores.

        Softmax is applied for exported probabilities, not before training loss.
        """

        return self.head(self.forward_features(images)[:, 0])
