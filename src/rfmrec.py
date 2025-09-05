import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, vmap


class Sphere:
    def __init__(self):
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}

    def _eps(self, dtype):
        return self.eps.get(dtype, 1e-6)

    def projx(self, x):
        eps = self._eps(x.dtype)
        norm = x.norm(dim=-1, keepdim=True)
        safe_norm = norm.clamp_min(eps)
        return x / safe_norm

    def proju(self, x, u):
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def _safe_sinc(self, t):
        small = t.abs() < 1e-3
        approx = 1 - t**2 / 6 + t**4 / 120
        return torch.where(small, approx, torch.sin(t) / t)

    def expmap(self, x, u):
        eps = self._eps(u.dtype)
        norm_u = u.norm(dim=-1, keepdim=True)
        safe_norm = norm_u.clamp_min(eps)
        sinc = self._safe_sinc(safe_norm)
        exp = x * torch.cos(safe_norm) + u * sinc
        return torch.where(norm_u > eps, exp, x + u)

    def inner(self, u, v, keepdim=False):
        return (u * v).sum(-1, keepdim=keepdim)

    def dist(self, x, y, keepdim=False):
        eps = self._eps(x.dtype)
        inner = self.inner(x, y, keepdim=keepdim).clamp(-1 + eps, 1 - eps)
        return torch.acos(inner)

    def logmap(self, x, y):
        eps = self._eps(x.dtype)
        dist = self.dist(x, y, keepdim=True)
        u = self.proju(x, y - x)
        u_norm = u.norm(dim=-1, keepdim=True)
        safe_u_norm = u_norm.clamp_min(eps)
        scaled = u * dist / safe_u_norm
        return torch.where(dist > eps, scaled, u)


class DenoisingNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.denoise_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

    def compute_sinusoidal_embedding(self, timesteps, dim, max_period=10000):
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, noisy_latent, timesteps):
        time_emb = self.time_mlp(
            self.compute_sinusoidal_embedding(timesteps, self.hidden_size)
        )
        model_input = torch.cat([noisy_latent, time_emb], dim=-1)
        return self.denoise_mlp(model_input)


class RiemannFlowMatchingRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.sample_steps = args.sample_steps
        self.flow_denoiser = DenoisingNetwork(args.hidden_size)
        self.seq_encoder = TransformerEncoder(
            args.num_blocks,
            args.hidden_size,
            args.max_len,
            args.num_heads,
            args.dropout,
            args.is_causal,
            args.pre_norm,
        )
        self.sphere = Sphere()

    def scale_t(self, t):
        return t * 1000

    def sample_t(self, shape, device):
        return torch.rand(shape, device=device)

    def geodesic(self, z_0, z_1):
        shooting_tangent_vec = self.sphere.logmap(z_0, z_1)

        def path(t):
            tangent_vecs = t.unsqueeze(-1) * shooting_tangent_vec
            points_at_time_t = self.sphere.expmap(z_0, tangent_vecs)
            return points_at_time_t

        return path

    def forward(self, input_embs, target_embs, input_mask, target_mask):
        seq_embs = self.seq_encoder(input_embs, input_mask)

        z_0 = target_embs
        z_0 = F.normalize(z_0, p=2, dim=-1)
        z_1 = seq_embs
        z_1 = F.normalize(z_1, p=2, dim=-1)

        t = self.sample_t([z_0.shape[0], z_0.shape[1]], z_0.device)

        def cond_v(z_0, z_1, t):
            path = self.geodesic(z_0, z_1)
            z_t, v_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return z_t, v_t

        z_t, target_vec = vmap(cond_v)(z_0, z_1, t)

        pred_vec = self.flow_denoiser(z_t, self.scale_t(t))

        diff = pred_vec - target_vec.detach()
        fm_loss = self.sphere.inner(diff, diff, keepdim=True) / pred_vec.size(-1)

        denom = target_mask.sum(1, keepdim=True)
        weight = target_mask / denom

        fm_loss = fm_loss * weight.unsqueeze(-1)
        fm_loss = fm_loss.sum(1).mean()

        return fm_loss, seq_embs

    def inference(self, input_embs, input_mask):
        batch_size = input_embs.size(0)
        device = input_embs.device

        seq_embs = self.seq_encoder(input_embs, input_mask)
        seq_emb = seq_embs[:, -1]
        z_t = F.normalize(seq_emb, p=2, dim=-1)

        for step in reversed(range(1, 1 + self.sample_steps)):
            t = torch.full((batch_size,), step / self.sample_steps, device=device)
            pred_vec = self.flow_denoiser(z_t, self.scale_t(t))
            z_t = z_t - pred_vec * 1 / self.sample_steps
            z_t = F.normalize(z_t, p=2, dim=-1)

        return z_t, seq_emb


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, dropout, pre_norm):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, sublayer):
        if self.pre_norm:
            return x + self.dropout(sublayer(self.layer_norm(x)))
        else:
            return self.layer_norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, max_len, dropout, rotary_base=10000):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        self._setup_rotary_embeddings(max_len, rotary_base)

    def _setup_rotary_embeddings(self, max_len, base):
        inv_frequencies = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        position_ids = torch.arange(max_len).float()
        frequencies = torch.einsum("i,j->ij", position_ids, inv_frequencies)

        self.register_buffer(
            "rotary_cos", torch.cos(frequencies)[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "rotary_sin", torch.sin(frequencies)[None, None, :, :], persistent=False
        )

    def _apply_rotary_encoding(self, tensor, cos_vals, sin_vals):
        even_dims = tensor[..., 0::2]
        odd_dims = tensor[..., 1::2]
        rotated = torch.cat(
            [
                even_dims * cos_vals - odd_dims * sin_vals,
                even_dims * sin_vals + odd_dims * cos_vals,
            ],
            dim=-1,
        )
        return rotated

    def forward(self, query, key, value, padding_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos_emb = self.rotary_cos[:, :, :seq_len, :]
        sin_emb = self.rotary_sin[:, :, :seq_len, :]
        q = self._apply_rotary_encoding(q, cos_emb, sin_emb)
        k = self._apply_rotary_encoding(k, cos_emb, sin_emb)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        if is_causal:
            seq_len = attn_scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask.bool(), -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        hidden = torch.matmul(attn_probs, v)
        hidden = (
            hidden.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        hidden = self.out_linear(hidden)
        return hidden


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, max_len, num_heads, dropout, is_causal, pre_norm):
        super().__init__()
        self.self_attn = MultiHeadedAttention(num_heads, hidden_size, max_len, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.sublayer1 = ResidualConnection(hidden_size, dropout, pre_norm)
        self.sublayer2 = ResidualConnection(hidden_size, dropout, pre_norm)

        self.is_causal = is_causal

    def forward(self, x, mask):
        x = self.sublayer1(
            x,
            lambda _x: self.self_attn(
                _x, _x, _x, padding_mask=mask, is_causal=self.is_causal
            ),
        )
        x = self.sublayer2(x, self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_blocks, hidden_size, max_len, num_heads, dropout, is_causal, pre_norm
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size, max_len, num_heads, dropout, is_causal, pre_norm
                )
                for _ in range(num_blocks)
            ]
        )
        if pre_norm:
            self.last_norm = nn.LayerNorm(hidden_size)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        x = self.last_norm(x)
        return x
