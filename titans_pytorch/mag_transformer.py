from __future__ import annotations

"""MAG (Memory-as-Gating) draft WITHOUT persistent memory tokens.

This file is a *starting point* for your initial MAG ablation:
- remove persistent memory prefix (Np) entirely
- keep short-term: sliding-window attention (SWA)
- keep long-term: NeuralMemory (Titans)
- combine with elementwise gate: y = g * y_mem + (1-g) * y_attn

About "flash attention": standard PyTorch SDPA/FlashAttention kernels handle
causal attention well, but *sliding-window* is still a masked variant.
This draft keeps the SWA mask very small (per-block seg x 2*seg) to make it
cheap, and relies on x-transformers Attend (which uses SDPA when possible).
"""

from typing import Callable
from math import ceil
from copy import deepcopy
from collections import namedtuple

import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# flash-attn (optional). If available on CUDA, we use local attention via window_size=(W, 0).
try:
    from flash_attn import flash_attn_func  # flash-attn 2.x / 3.x
except Exception:
    flash_attn_func = None


from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

from x_transformers.attend import Attend
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions

from titans_pytorch.neural_memory import NeuralMemory

LinearNoBias = lambda in_f, out_f: nn.Linear(in_f, out_f, bias=False)
AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def pad_to_multiple(x: Tensor, mult: int):
    b, n = x.shape[:2]
    n2 = round_up_multiple(n, mult)
    pad = n2 - n
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))
    return x, pad

# -----------------------------
# sampling utils
# -----------------------------

def _log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -_log(-_log(noise))


def gumbel_sample(t, temperature=1.0):
    if temperature > 0.0:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim=-1, keepdim=True)


def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)


# -----------------------------
# MLP
# -----------------------------

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def FeedForward(dim: int, mult: int = 4) -> nn.Module:
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim),
    )

# -----------------------------
# Sliding-window attention (no persistent prefix)
# -----------------------------

class SlidingWindowAttention(nn.Module):
    """Causal sliding-window attention (SWA), optimized with FlashAttention when available.

    Short-term memory = attend to at most `segment_len` tokens to the left (no future).
    This matches Titans MAG's blue band when you set window size = segment_len.

    If flash-attn is installed and the input is CUDA fp16/bf16, we call:

        flash_attn_func(q, k, v, causal=True, window_size=(segment_len, 0))

    Otherwise we fall back to x-transformers Attend:
    - training: uses a tiny (seg x 2*seg) boolean mask per block
    - inference: uses the cached last `segment_len` keys/values with causal=True (no mask needed)
    """

    def __init__(
        self,
        dim: int,
        *,
        segment_len: int,
        dim_head: int = 64,
        heads: int = 8,
        accept_value_residual: bool = False,
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        self.segment_len = segment_len
        self.dim_head = dim_head
        self.heads = heads

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = Attend(causal=True, **attend_kwargs)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_learned_v_mix = (
            nn.Sequential(
                nn.Linear(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid(),
            )
            if accept_value_residual
            else None
        )

    def _mix_v(self, v, value_residual, x_normed):
        if not exists(self.to_learned_v_mix):
            return v
        mix = self.to_learned_v_mix(x_normed)
        return v.lerp(value_residual, mix)

    @staticmethod
    def _can_use_flash(x: Tensor) -> bool:
        if flash_attn_func is None:
            return False
        if not x.is_cuda:
            return False
        return x.dtype in (torch.float16, torch.bfloat16)

    def _flash(self, q_bhnd: Tensor, k_bhnd: Tensor, v_bhnd: Tensor) -> Tensor:
        """Run flash-attn local causal attention.

        Inputs are (b, h, n, d). flash-attn expects (b, n, h, d).
        Returns (b, h, n, d).
        """
        q = rearrange(q_bhnd, 'b h n d -> b n h d')
        k = rearrange(k_bhnd, 'b h n d -> b n h d')
        v = rearrange(v_bhnd, 'b h n d -> b n h d')

        out = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            window_size=(self.segment_len, 0),
        )

        return rearrange(out, 'b n h d -> b h n d')

    def forward_inference(self, token: Tensor, cache, value_residual=None):
        """token: (b, 1, d); cache: (k, v) each (b, h, t, dh) with t <= segment_len"""
        token = self.norm(token)

        q, k, v = self.to_qkv(token).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v

        v = self._mix_v(v, value_residual, token)

        ck, cv = cache
        k = torch.cat((ck, k), dim=-2)
        v = torch.cat((cv, v), dim=-2)

        # keep last window (unrotated for cache)
        k = k[..., -self.segment_len :, :]
        v = v[..., -self.segment_len :, :]

        next_cache = (k, v)

        # rotary for compute
        q_rot, k_rot = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        if self._can_use_flash(token):
            out_bhnd = self._flash(q_rot, k_rot, v)
        else:
            out_bhnd, _ = self.attend(q_rot, k_rot, v)

        out = self.merge_heads(out_bhnd)
        out = self.to_out(out)
        return out, AttnIntermediates(orig_v, next_cache)

    def forward_train(self, seq: Tensor, value_residual=None):
        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v

        v = self._mix_v(v, value_residual, seq)

        # keep unrotated k,v for cache format
        k_cache, v_cache = k, v

        # rotary for compute
        q_rot, k_rot = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        if self._can_use_flash(seq):
            out_bhnd = self._flash(q_rot, k_rot, v)
            out = self.merge_heads(out_bhnd)
            out = self.to_out(out)
            return out, AttnIntermediates(orig_v, (k_cache, v_cache))

        # -------- fallback: block into prev+cur and use small boolean mask --------
        b, n = seq.shape[:2]
        seg = self.segment_len

        seq2, pad = pad_to_multiple(seq, seg)
        n2 = seq2.shape[1]

        q, k, v = self.to_qkv(seq2).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        v = self._mix_v(v, value_residual, seq2)

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold into blocks
        q, k, v = [rearrange(t, 'b h (w n) d -> (b w) h n d', n=seg) for t in (q, k, v)]
        w = n2 // seg

        # build (prev + cur) kv
        k2 = rearrange(k, '(b w) h n d -> b w h n d', b=b, w=w)
        v2 = rearrange(v, '(b w) h n d -> b w h n d', b=b, w=w)

        k2 = pad_at_dim(k2, (1, 0), dim=1, value=0.0)
        v2 = pad_at_dim(v2, (1, 0), dim=1, value=0.0)

        k_cat = torch.cat((k2[:, :-1], k2[:, 1:]), dim=-2)
        v_cat = torch.cat((v2[:, :-1], v2[:, 1:]), dim=-2)

        k_cat = rearrange(k_cat, 'b w h n d -> (b w) h n d')
        v_cat = rearrange(v_cat, 'b w h n d -> (b w) h n d')

        # mask: causal + within last seg keys
        idx = torch.arange(n2, device=seq.device)
        q_idx = rearrange(idx, '(w n) -> w n', n=seg)

        k_idx = pad_at_dim(q_idx, (1, 0), dim=0, value=-1e4)
        k_idx = torch.cat((k_idx[:-1], k_idx[1:]), dim=-1)

        q_idx = rearrange(q_idx, 'w i -> w i 1')
        k_idx = rearrange(k_idx, 'w j -> w 1 j')

        mask = (q_idx >= k_idx) & ((q_idx - k_idx) <= seg)
        mask = repeat(mask, 'w i j -> (b w) 1 i j', b=b)

        out_bhnd, _ = self.attend(q, k_cat, v_cat, mask=mask)
        out = self.merge_heads(out_bhnd)
        out = self.to_out(out)
        out = rearrange(out, '(b w) n d -> b (w n) d', b=b)

        if pad > 0:
            out = out[:, :-pad]

        # cache tensors
        k_full = rearrange(k, '(b w) h n d -> b h (w n) d', b=b, w=w)
        v_full = rearrange(v, '(b w) h n d -> b h (w n) d', b=b, w=w)
        if pad > 0:
            k_full = k_full[..., :-pad, :]
            v_full = v_full[..., :-pad, :]

        return out, AttnIntermediates(orig_v, (k_full, v_full))

    def forward(self, seq: Tensor, value_residual=None, cache=None):
        if exists(cache):
            assert seq.shape[-2] == 1
            return self.forward_inference(seq, cache, value_residual)
        return self.forward_train(seq, value_residual=value_residual)


# -----------------------------
# MAG model (no persistent prefix)
# -----------------------------

class MemoryAsGateTransformerNoPersist(nn.Module):
    """MAG without persistent memory.

    Cache format matches your MAC code:
      (inference_seq_index, kv_caches, neural_mem_caches)

    kv_caches is stacked as: (layers, kv=2, b, h, t, dh), with t <= segment_len.
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        dim: int,
        depth: int,
        segment_len: int,
        neural_memory_segment_len: int | None = None,
        dim_head: int = 64,
        heads: int = 8,
        attn_heads: int | None = None,
        mem_heads: int | None = None,
        ff_mult: int = 4,
        num_residual_streams: int = 4,
        neural_memory_model: nn.Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        neural_memory_batch_size: int | None = None,
        neural_memory_qkv_receives_diff_views: bool = False,
        neural_mem_weight_residual: bool = False,
        token_emb: nn.Module | None = None,
    ):
        super().__init__()

        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)
        self.token_emb = token_emb

        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)

        self.segment_len = segment_len
        self.neural_memory_segment_len = default(neural_memory_segment_len, segment_len)

        # head split between attention and memory branches
        if attn_heads is None and mem_heads is None:
            mem_heads = heads // 2
            attn_heads = heads - mem_heads
        elif attn_heads is None:
            attn_heads = heads - int(mem_heads)
        elif mem_heads is None:
            mem_heads = heads - int(attn_heads)

        assert attn_heads > 0 and mem_heads > 0
        self.attn_heads = int(attn_heads)
        self.mem_heads = int(mem_heads)

        init_hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(
            num_residual_streams,
            dim=dim,
            add_stream_embed=True,
            disable=num_residual_streams == 1,
        )

        self.layers = nn.ModuleList([])

        layers = tuple(range(1, depth + 1))
        neural_memory_layers = default(neural_memory_layers, layers)

        self.neural_mem_weight_residual = neural_mem_weight_residual
        is_first_neural_mem = True

        for layer in layers:
            is_first = layer == 1

            attn = SlidingWindowAttention(
                dim,
                segment_len=segment_len,
                dim_head=dim_head,
                heads=self.attn_heads,
                accept_value_residual=not is_first,
            )

            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual=False)

                if (not is_first) and neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1
                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim),
                        nn.Linear(dim, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views=3),
                        nn.Softmax(dim=-1),
                    )

                mem = NeuralMemory(
                    dim=dim,
                    # dim_head=dim_head,
                    # heads=self.mem_heads,
                    chunk_size=self.neural_memory_segment_len,
                    batch_size=neural_memory_batch_size,
                    model=deepcopy(neural_memory_model),
                    qkv_receives_diff_views=True,
                    accept_weight_residual=neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs,
                )

                is_first_neural_mem = False

            ff = FeedForward(dim, mult=ff_mult)

            gate = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )

            self.layers.append(
                nn.ModuleList([
                    mem_hyper_conn,
                    init_hyper_conn(),
                    init_hyper_conn(),
                    mem_qkv_layer_selector,
                    mem,
                    attn,
                    ff,
                    gate,
                ])
            )

        self.norm = nn.RMSNorm(dim)
        self.to_logits = LinearNoBias(dim, num_tokens)

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature=1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(min_p=0.1),
        show_progress=True,
        use_cache=False,
    ):
        was_training = self.training
        self.eval()

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        cache = None
        factorized_pos_emb = None

        if use_cache:
            axial_dims = self.axial_pos_emb.maybe_derive_outer_dim(seq_len, (self.neural_memory_segment_len,))
            factorized_pos_emb = self.axial_pos_emb(axial_dims, return_factorized=True)

        with tqdm.tqdm(total=sample_num_times, disable=not show_progress) as pbar:
            while out.shape[-1] < seq_len:
                logits, next_cache = self.forward(
                    out,
                    cache=cache,
                    return_cache=True,
                    factorized_pos_emb=factorized_pos_emb,
                )

                if use_cache:
                    cache = next_cache

                logits = logits[:, -1]
                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature=temperature)
                out = torch.cat((out, sample), dim=-1)
                pbar.update(1)

        self.train(was_training)
        return out[..., prompt_seq_len:]

    def forward(
        self,
        x: Tensor,
        *,
        return_loss: bool = False,
        cache=None,
        return_cache: bool = False,
        factorized_pos_emb=None,
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        b, n = x.shape[:2]

        # embed + pos
        x = self.token_emb(x)
        pos_emb = self.axial_pos_emb.forward_with_seq_len(n, (self.neural_memory_segment_len,), factorized=factorized_pos_emb)
        x = x + pos_emb

        # cache protocol (same shape semantics as MAC)
        is_inferencing = exists(cache)
        if not exists(cache):
            cache = (n - 1, None, None)

        inference_seq_index, kv_caches, neural_mem_caches = cache
        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        value_residual = None
        mem_weight_residual = None
        mem_input_layers = []

        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]

        x = self.expand_streams(x)

        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff, gate in self.layers:
            mem_out = None
            next_neural_mem_cache = None

            # memory branch
            if exists(mem):
                mem_input, _ = mem_hyper_conn(x)

                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = torch.stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = torch.stack((mem_input, *mem_input_layers))
                    selected = mem_qkv_layer_selector(mem_input)
                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                mem_out, next_neural_mem_cache = mem.forward(
                    qkv_mem_input,
                    state=next(neural_mem_caches, None),
                    prev_weights=mem_weight_residual,
                )

                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

            # attention branch
            attn_in, add_residual = attn_hyper_conn(x)
            mem_input_layers.append(attn_in)

            kv_cache = next(kv_caches, None)
            if is_inferencing and not exists(kv_cache):
                # empty cache (b,h,0,dh)
                device = attn_in.device
                k0 = torch.zeros((b, attn.heads, 0, attn.dim_head), device=device, dtype=attn_in.dtype)
                v0 = torch.zeros_like(k0)
                kv_cache = (k0, v0)

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual=value_residual,
                cache=kv_cache if is_inferencing else None,
            )

            mem_input_layers.append(attn_out)
            value_residual = default(value_residual, values)

            # MAG gating
            if exists(mem_out):
                g = gate(mem_out)
                mixed = g * mem_out + (1.0 - g) * attn_out
            else:
                mixed = attn_out

            x = add_residual(mixed)

            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # FF
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            mem_input_layers.append(ff_out)
            x = add_ff_residual(ff_out)

        if return_cache:
            next_kv_caches = torch.stack([torch.stack(kv_cache) for kv_cache in next_kv_caches])
            next_kv_caches = next_kv_caches[..., -self.segment_len:, :]
            next_cache = (inference_seq_index + 1, next_kv_caches, next_neural_mem_caches)

        x = self.reduce_streams(x)
        x = self.norm(x)
        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits
            return logits, next_cache

        loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
        if not return_cache:
            return loss
        return loss, next_cache
