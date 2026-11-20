"""Vision Transformer (ViT) written from scratch in PyTorch.

Every component is implemented by hand (no nn.MultiheadAttention, no timm) so
that each line maps onto an equation from Dosovitskiy et al. (2021),
"An Image is Worth 16x16 Words".
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split image into P x P patches and linearly project each one to D dims.

    A Conv2d with kernel = stride = P is *exactly* "flatten each patch and
    multiply by a (P*P*C) x D matrix" -- just computed efficiently.
    (B, C, H, W) -> (B, N, D) with N = (H/P) * (W/P)
    """

    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=192):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)


class MultiHeadSelfAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_k)) V, computed for h heads in parallel."""

    def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)    # W_Q, W_K, W_V fused into one matmul
        self.out = nn.Linear(dim, dim)        # W_O
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_attn = None                 # stored for visualisation

    def forward(self, x, keep_attn=False):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each (B, h, T, dk)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B, h, T, T)
        attn = scores.softmax(dim=-1)
        if keep_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        y = attn @ v                                            # (B, h, T, dk)
        y = y.transpose(1, 2).reshape(B, T, D)                  # concat heads
        return self.proj_drop(self.out(y))


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic depth: randomly skip a whole residual branch per sample."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


class EncoderBlock(nn.Module):
    """Pre-LN transformer block:
        z' = z  + MSA(LN(z))
        z  = z' + MLP(LN(z'))
    """

    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, proj_drop=drop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.dp = DropPath(drop_path)

    def forward(self, x, keep_attn=False):
        x = x + self.dp(self.attn(self.ln1(x), keep_attn))
        x = x + self.dp(self.mlp(self.ln2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, dim=192,
                 depth=6, heads=3, mlp_ratio=2.0, drop=0.0, drop_path=0.1,
                 use_pos=True, use_cls=True):
        super().__init__()
        self.patch = PatchEmbedding(img_size, patch_size, 3, dim)
        n = self.patch.num_patches
        self.use_cls, self.use_pos = use_cls, use_pos
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls else None
        self.pos = nn.Parameter(torch.zeros(1, n + int(use_cls), dim)) if use_pos else None
        self.drop = nn.Dropout(drop)
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            EncoderBlock(dim, heads, mlp_ratio, drop, dpr[i]) for i in range(depth))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init()

    def _init(self):
        if self.pos is not None:
            nn.init.trunc_normal_(self.pos, std=0.02)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, keep_attn=False):
        x = self.patch(x)                                    # (B, N, D)
        if self.use_cls:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)  # (B, N+1, D)
        if self.use_pos:
            x = x + self.pos
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, keep_attn)
        x = self.ln(x)
        feat = x[:, 0] if self.use_cls else x.mean(dim=1)   # CLS token or mean-pool
        return self.head(feat)


class SmallResNet(nn.Module):
    """CNN baseline of comparable size (ResNet-style, ~1M params)."""

    def __init__(self, num_classes=10, w=(64, 128, 256)):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))

        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3 = block(w[0], w[0], 1), block(w[0], w[1], 2), block(w[1], w[2], 2)
        self.s2 = nn.Sequential(nn.Conv2d(w[0], w[1], 1, 2, bias=False), nn.BatchNorm2d(w[1]))
        self.s3 = nn.Sequential(nn.Conv2d(w[1], w[2], 1, 2, bias=False), nn.BatchNorm2d(w[2]))
        self.head = nn.Linear(w[2], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(x + self.b1(x))
        x = F.relu(self.s2(x) + self.b2(x))
        x = F.relu(self.s3(x) + self.b3(x))
        return self.head(x.mean((2, 3)))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
