"""One-dimensional Diffusion Transformer (DiT) for HRRP view synthesis.

Architecture (following Peebles & Xie, ICCV 2023):

* **Patchify**: the noised HRRP and the source HRRP are stacked as two input
  channels and cut into non-overlapping patches of length ``p`` (a strided
  1-D convolution).  With 800 range cells and ``p = 8`` this gives 100 tokens.
* **Positional embedding**: fixed 1-D sine-cosine embeddings, as in ViT/DiT.
* **Backbone**: ``N`` DiT blocks with pre-norm self-attention + 4x MLP.  Every
  block uses **adaLN-Zero**: shift/scale/gate parameters are regressed from the
  conditioning vector ``c`` and the block is initialised as the identity.
* **Conditioning**: ``c = t_emb(t) + angle_pair_emb(a_src, a_tgt)``.  The
  source HRRP enters through the channel concatenation; both azimuths enter
  through the adaLN conditioning.  A learned null vector + zeroed source
  supports classifier-free guidance.
* **Decoder**: adaptive final layer norm + linear unpatchify to one channel.

The identity of the target is not given as a label: it must be inferred from
the source HRRP, exactly as requested by the task ("generate the HRRP of the
original target at the target azimuth").
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DiT model configurations from Table 1 of the paper (layers, width, heads).
DIT_CONFIGS: dict[str, Tuple[int, int, int]] = {
    "DiT-S": (12, 384, 6),
    "DiT-B": (12, 768, 12),
    "DiT-L": (24, 1024, 16),
    "DiT-XL": (28, 1152, 16),
}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class SourceEmbedder(nn.Module):
    """Global embedding of the source HRRP, used like an angle embedding.

    The full 800-cell profile is projected to the DiT hidden width and added to
    the adaLN conditioning vector ``c``.  This is the "angles-style" injection:
    no extra input channel is used; the source only enters through ``c``.
    """

    def __init__(self, input_length: int, hidden_size: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_length, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (ADM/DiT convention), float32 output."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_size))


class AnglePairEmbedder(nn.Module):
    """Embedding of (original azimuth, target azimuth) as one vector.

    Fourier features avoid a piecewise-linear scalar mapping and let the model
    interpolate smoothly between angles.  The pair is concatenated (not summed)
    so that source and target roles cannot be confused.
    """

    def __init__(
        self,
        hidden_size: int,
        angle_max: float = 180.0,
        fourier_bands: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        hidden: int = 256,
    ):
        super().__init__()
        self.angle_max = float(angle_max)
        self.bands = tuple(float(b) for b in fourier_bands)
        per_angle = 1 + 2 * len(self.bands)
        self.mlp = nn.Sequential(
            nn.Linear(2 * per_angle, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden_size),
        )

    def _features(self, angle: torch.Tensor) -> torch.Tensor:
        a = angle.float() / self.angle_max
        if a.dim() == 1:
            a = a[:, None]
        feats = [a]
        for b in self.bands:
            feats.append(torch.sin(math.pi * b * a))
            feats.append(torch.cos(math.pi * b * a))
        return torch.cat(feats, dim=1)

    def forward(self, a_src: torch.Tensor, a_tgt: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self._features(a_src), self._features(a_tgt)], dim=1))


class SingleAngleEmbedder(nn.Module):
    """Fourier embedding of one azimuth (class-conditional v4)."""

    def __init__(
        self,
        hidden_size: int,
        angle_max: float = 180.0,
        fourier_bands: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        hidden: int = 256,
    ):
        super().__init__()
        self.angle_max = float(angle_max)
        self.bands = tuple(float(b) for b in fourier_bands)
        per_angle = 1 + 2 * len(self.bands)
        self.mlp = nn.Sequential(
            nn.Linear(per_angle, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden_size),
        )

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        a = angle.float() / self.angle_max
        if a.dim() == 1:
            a = a[:, None]
        feats = [a]
        for b in self.bands:
            feats.append(torch.sin(math.pi * b * a))
            feats.append(torch.cos(math.pi * b * a))
        return self.mlp(torch.cat(feats, dim=1))


class AngleTokenEmbedder(nn.Module):
    """K learned angle tokens for cross-attention (class-adaLN + angle-cross).

    The scalar angle is encoded by the same Fourier MLP, then combined with K
    learned slot embeddings so the target tokens can attend to several angle
    "views" instead of a single vector.
    """

    def __init__(
        self,
        hidden_size: int,
        num_tokens: int = 4,
        angle_max: float = 180.0,
        fourier_bands: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        hidden: int = 256,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.angle_emb = SingleAngleEmbedder(
            hidden_size,
            angle_max=angle_max,
            fourier_bands=fourier_bands,
            hidden=hidden,
        )
        self.slots = nn.Parameter(torch.zeros(self.num_tokens, hidden_size))
        nn.init.normal_(self.slots, std=0.02)

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        base = self.angle_emb(angle)  # (B, D)
        return base.unsqueeze(1) + self.slots.unsqueeze(0)  # (B, K, D)


class ClassEmbedder(nn.Module):
    """Class embedding with a learned null class for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.null_index = int(num_classes)
        self.embed = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, y: torch.Tensor, drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = y.long()
        if drop is not None:
            y = torch.where(
                drop.bool(),
                torch.full_like(y, self.null_index),
                y,
            )
        return self.embed(y)


def get_1d_sincos_pos_embed(embed_dim: int, length: int, max_period: float = 10000.0) -> np.ndarray:
    """Fixed 1-D sine-cosine positional embedding, shape (length, embed_dim)."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 1-D sine-cosine embeddings")
    pos = np.arange(length, dtype=np.float64)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (max_period ** omega)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Patch embedding / attention / blocks
# ---------------------------------------------------------------------------
class PatchEmbed1d(nn.Module):
    """Non-overlapping patchify of a 2-channel 1-D signal."""

    def __init__(self, input_length: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if input_length % patch_size != 0:
            raise ValueError(
                f"input_length={input_length} must be divisible by patch_size={patch_size}"
            )
        self.input_length = int(input_length)
        self.patch_size = int(patch_size)
        self.num_patches = input_length // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)  # (B, T, D)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head self-attention over the token sequence."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: target tokens query source tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        m = context.shape[1]
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(b, m, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, M, Dh)
        k, v = kv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning (Eq. in Sec. 3.2).

    With ``use_cross_attn=True`` the block additionally attends to a source
    token sequence after self-attention (DiT's cross-attention variant).  A
    zero-initialised scalar gate keeps the block an identity function at init.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.use_cross_attn = bool(use_cross_attn)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads, dropout=dropout)
            self.cross_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        if self.use_cross_attn:
            if context is None:
                raise ValueError("cross-attention block requires a source token sequence")
            x = x + self.cross_gate * self.cross_attn(self.norm_cross(x), context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer1d(nn.Module):
    """Adaptive final norm + linear unpatchify to ``out_channels``."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class DiT1D(nn.Module):
    def __init__(
        self,
        input_length: int = 800,
        patch_size: int = 8,
        in_channels: int = 1,
        cond_channels: int = 1,
        cond_mode: str = "concat",
        source_hidden: int = 512,
        cross_gate_init: float = 0.0,
        cross_proj_zero: bool = False,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
        class_angle_mode: str = "additive",
        class_angle_hidden: int = 512,
        film_hidden: int = 512,
        angle_tokens: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        angle_max: float = 180.0,
        angle_fourier_bands: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        angle_hidden: int = 256,
        out_channels: int = 1,
    ):
        super().__init__()
        self.input_length = int(input_length)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.cond_mode = str(cond_mode).lower()
        if self.cond_mode not in ("concat", "adaln", "cross", "class", "class_cross"):
            raise ValueError(f"unknown cond_mode {cond_mode!r}")
        if self.cond_mode in ("adaln", "cross", "class", "class_cross"):
            cond_channels = 0  # source does not use an input channel
        self.cond_channels = int(cond_channels)
        self.out_channels = int(out_channels)
        self.hidden_size = int(hidden_size)
        self.cross_gate_init = float(cross_gate_init)
        self.cross_proj_zero = bool(cross_proj_zero)
        self.num_classes = int(num_classes)
        self.class_dropout_prob = float(class_dropout_prob)
        self.class_angle_mode = str(class_angle_mode).lower()
        self.angle_tokens = int(angle_tokens)
        if self.class_angle_mode not in ("additive", "joint_mlp", "block_film"):
            raise ValueError(f"unknown class_angle_mode {class_angle_mode!r}")

        self.x_embedder = PatchEmbed1d(
            input_length, patch_size, in_channels + self.cond_channels, hidden_size
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.cond_mode in ("class", "class_cross"):
            self.class_embedder = ClassEmbedder(num_classes, hidden_size)
            self.null_cond = nn.Parameter(torch.zeros(hidden_size))  # unused, kept for symmetry
            if self.cond_mode == "class":
                self.angle_embedder = SingleAngleEmbedder(
                    hidden_size,
                    angle_max=angle_max,
                    fourier_bands=angle_fourier_bands,
                    hidden=angle_hidden,
                )
                if self.class_angle_mode == "joint_mlp":
                    self.class_angle_mlp = nn.Sequential(
                        nn.Linear(2 * hidden_size, int(class_angle_hidden)),
                        nn.SiLU(),
                        nn.Linear(int(class_angle_hidden), hidden_size),
                    )
                elif self.class_angle_mode == "block_film":
                    self.block_films = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(hidden_size, int(film_hidden)),
                                nn.SiLU(),
                                nn.Linear(int(film_hidden), 2 * hidden_size),
                            )
                            for _ in range(int(depth))
                        ]
                    )
            else:
                self.angle_tokenizer = AngleTokenEmbedder(
                    hidden_size,
                    num_tokens=self.angle_tokens,
                    angle_max=angle_max,
                    fourier_bands=angle_fourier_bands,
                    hidden=angle_hidden,
                )
        else:
            self.angle_embedder = AnglePairEmbedder(
                hidden_size,
                angle_max=angle_max,
                fourier_bands=angle_fourier_bands,
                hidden=angle_hidden,
            )
            self.null_cond = nn.Parameter(torch.zeros(hidden_size))
        if self.cond_mode == "adaln":
            self.source_embedder = SourceEmbedder(input_length, hidden_size, hidden=source_hidden)
            self.null_source = nn.Parameter(torch.zeros(hidden_size))
        if self.cond_mode == "cross":
            self.source_patch_embed = PatchEmbed1d(
                input_length, patch_size, in_channels=1, embed_dim=hidden_size
            )
            self.null_source_token = nn.Parameter(torch.zeros(hidden_size))

        num_patches = self.x_embedder.num_patches
        pos = get_1d_sincos_pos_embed(hidden_size, num_patches)
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_cross_attn=(self.cond_mode in ("cross", "class_cross")),
                )
                for _ in range(int(depth))
            ]
        )
        self.final_layer = FinalLayer1d(hidden_size, patch_size, out_channels)
        self.initialize_weights()

    # ------------------------------------------------------------------
    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].bias, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].bias, std=0.02)

        # adaLN-Zero: every block starts as the identity function.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.null_cond, 0)
        if self.cond_mode == "adaln":
            nn.init.constant_(self.null_source, 0)
        if self.cond_mode == "class" and self.class_angle_mode == "block_film":
            for film in self.block_films:
                nn.init.constant_(film[-1].weight, 0)
                nn.init.constant_(film[-1].bias, 0)
        if self.cond_mode == "cross":
            nn.init.constant_(self.null_source_token, 0)
        if self.cond_mode in ("cross", "class_cross"):
            for block in self.blocks:
                nn.init.constant_(block.cross_gate, self.cross_gate_init)
                if self.cross_proj_zero:
                    nn.init.constant_(block.cross_attn.proj.weight, 0)
                    nn.init.constant_(block.cross_attn.proj.bias, 0)

    # ------------------------------------------------------------------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.reshape(b, n, self.out_channels, self.patch_size)
        x = x.permute(0, 2, 3, 1).reshape(b, self.out_channels, n * self.patch_size)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        cond_drop: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise (or v) for ``x`` at diffusion step ``t``.

        Parameters
        ----------
        x : (B, 1, L) noised target HRRP in scaled units
        t : (B,) integer diffusion steps
        x_cond : (B, 1, L) source HRRP in scaled units (ignored in class mode)
        a_src, a_tgt : (B,) azimuths; in class mode ``a_src`` carries the
            integer class label (cast to long) when ``y`` is not given
        cond_drop : optional (B,) bool; in class mode True drops the class
        y : optional (B,) class labels; class mode only
        """
        drop = None
        if cond_drop is not None:
            drop = cond_drop.bool()

        conditions = None
        if self.cond_mode in ("class", "class_cross"):
            if y is None:
                y = a_src.long()
            h = self.x_embedder(x)
            h = h + self.pos_embed.to(h.dtype)
            c_cls = self.class_embedder(y, drop)
            if self.cond_mode == "class":
                c_ang = self.angle_embedder(a_tgt)
                if self.class_angle_mode == "joint_mlp":
                    c = self.t_embedder(t) + self.class_angle_mlp(torch.cat([c_cls, c_ang], dim=1))
                elif self.class_angle_mode == "block_film":
                    c_base = self.t_embedder(t) + c_cls
                    conditions = []
                    for film in self.block_films:
                        gamma, beta = film(c_cls).chunk(2, dim=1)
                        conditions.append(c_base + (1.0 + gamma) * c_ang + beta)
                    c = conditions[-1]
                else:
                    c = self.t_embedder(t) + c_ang + c_cls
            else:
                # class goes through adaLN; angle enters as cross-attn tokens.
                c = self.t_embedder(t) + c_cls
            c = c.to(h.dtype)
        else:
            if self.cond_mode == "concat":
                if drop is not None:
                    keep = (~drop).to(x_cond.dtype).view(-1, 1, 1)
                    x_cond = x_cond * keep
                h = self.x_embedder(torch.cat([x, x_cond], dim=1))  # (B, T, D)
            else:
                # adaln: source enters c; cross: source enters as token context.
                h = self.x_embedder(x)

            h = h + self.pos_embed.to(h.dtype)

            c = self.t_embedder(t)
            c_angle = self.angle_embedder(a_src, a_tgt)
            if drop is not None:
                c_angle = torch.where(
                    drop.unsqueeze(1),
                    self.null_cond.to(c_angle.dtype).expand_as(c_angle),
                    c_angle,
                )
            c = c + c_angle
            if self.cond_mode == "adaln":
                c_src = self.source_embedder(x_cond)
                if drop is not None:
                    c_src = torch.where(
                        drop.unsqueeze(1),
                        self.null_source.to(c_src.dtype).expand_as(c_src),
                        c_src,
                    )
                c = c + c_src
            c = c.to(h.dtype)

        context = None
        if self.cond_mode == "class_cross":
            context = self.angle_tokenizer(a_tgt)
        elif self.cond_mode == "cross":
            context = self.source_patch_embed(x_cond)
            context = context + self.pos_embed.to(context.dtype)
            if drop is not None:
                null = self.null_source_token.to(context.dtype).view(1, 1, -1)
                context = torch.where(drop.view(-1, 1, 1), null.expand_as(context), context)

        if conditions is None:
            for block in self.blocks:
                h = block(h, c, context)
            c_final = c
        else:
            for block, ci in zip(self.blocks, conditions):
                h = block(h, ci, context)
            c_final = conditions[-1]
        h = self.final_layer(h, c_final)
        return self.unpatchify(h)

    @property
    def num_patches(self) -> int:
        return self.x_embedder.num_patches


def build_dit(cfg: dict) -> DiT1D:
    """Build the model from a config dictionary (``cfg['model']``)."""
    m = cfg["model"]
    variant = str(m.get("variant", "DiT-B"))
    if variant in DIT_CONFIGS:
        depth, hidden, heads = DIT_CONFIGS[variant]
    else:
        depth, hidden, heads = 12, 768, 12
    depth = int(m.get("depth", depth))
    hidden = int(m.get("hidden_size", hidden))
    heads = int(m.get("num_heads", heads))
    cond_mode = str(m.get("cond_mode", "concat")).lower()
    default_cond_channels = 1 if cond_mode == "concat" else 0
    return DiT1D(
        input_length=int(m.get("input_length", cfg.get("data", {}).get("length", 800))),
        patch_size=int(m.get("patch_size", 8)),
        in_channels=int(m.get("in_channels", 1)),
        cond_channels=int(m.get("cond_channels", default_cond_channels)),
        cond_mode=cond_mode,
        source_hidden=int(m.get("source_hidden", 512)),
        cross_gate_init=float(m.get("cross_gate_init", 0.0)),
        cross_proj_zero=bool(m.get("cross_proj_zero", False)),
        num_classes=int(m.get("num_classes", cfg.get("data", {}).get("num_classes", 10))),
        class_dropout_prob=float(m.get("class_dropout_prob", 0.1)),
        class_angle_mode=str(m.get("class_angle_mode", "additive")),
        class_angle_hidden=int(m.get("class_angle_hidden", 512)),
        film_hidden=int(m.get("film_hidden", 512)),
        angle_tokens=int(m.get("angle_tokens", 4)),
        hidden_size=hidden,
        depth=depth,
        num_heads=heads,
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        dropout=float(m.get("dropout", 0.0)),
        angle_max=float(m.get("angle_max", cfg.get("data", {}).get("angle_max", 180.0))),
        angle_fourier_bands=tuple(m.get("angle_fourier_bands", (1.0, 2.0, 4.0, 8.0))),
        angle_hidden=int(m.get("angle_hidden", 256)),
        out_channels=int(m.get("out_channels", 1)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
