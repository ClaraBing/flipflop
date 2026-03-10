from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from math import prod, sqrt
from typing import Literal

import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array

from src.ops import get_embedding, rmsnorm, rotary_embedding, sdpa

DtypeName = Literal["float32", "float16", "bfloat16"]

Param = namedtuple("Param", "shape std lr")
is_param = lambda x: isinstance(x, Param)


@dataclass
class Transformer:
    D: int = 768  # d_model
    L: int = 12  # n_layers
    M: int = 3072  # d_mlp
    H: int = 12  # n_heads
    K: int = 64  # d_head
    V: int = 50257  # vocab_size
    fsdp: bool = False
    scan_unroll: int | bool = True
    dtype: DtypeName = "bfloat16"
    norm_dtype: DtypeName | None = None
    norm_eps: float = 1e-6
    rope_dtype: DtypeName | None = None
    rope_freq: int = 10_000
    flash_attention: bool = True
    grad_checkpoint: bool = False

    @property
    def spec(self):
        D, L, M, V = self.D, self.L, self.M, self.V
        return {
            "embd": Param("VD", 1, D),
            "blocks": {
                "attn": {
                    "q": Param("LDHK", 1 / sqrt(D), L),
                    "k": Param("LDHK", 1 / sqrt(D), L),
                    "v": Param("LDHK", 1 / sqrt(D), L),
                    "head": Param("LHKD", 0, L),
                },
                "mlp": {
                    "up": Param("LDM", 1 / sqrt(D), L * M / D),
                    "head": Param("LMD", 0, L * D / M),
                },
            },
            "head": Param("DV", 0, V / D),
        }

    def init(self, key: Array) -> dict:
        keys = iter(jr.split(key, len(jax.tree.leaves(self.spec, is_leaf=is_param))))

        def _init(s):
            shape = [getattr(self, d) for d in s.shape]
            shard = ["data" if d == "D" else None for d in s.shape]
            shard = P(*shard) if self.fsdp else None
            return jr.normal(next(keys), shape, self.dtype, out_sharding=shard) * s.std

        return jax.tree.map(_init, self.spec, is_leaf=is_param)

    def apply(self, params: dict, x: Array) -> Array:
        collect = lambda p: reshard(p, P()) if self.fsdp else p
        norm = partial(rmsnorm, eps=self.norm_eps, dtype=self.norm_dtype)
        rope = partial(rotary_embedding, freq=self.rope_freq, dtype=self.rope_dtype)

        def attn(params, x):
            q, k, v = (jnp.einsum("...D,DHK->...HK", x, params[s]) for s in "qkv")
            q, k = rope(norm(q)), rope(norm(k))
            impl = "flash" if self.flash_attention else "xla"
            a = sdpa(q, k, v, is_causal=True, scale=8 / self.K, implementation=impl)
            return jnp.einsum("...HK,HKD->...D", a, params["head"])

        def mlp(params, x):
            h = jnp.einsum("...D,DM->...M", x, params["up"])
            h = jax.nn.gelu(h)
            return jnp.einsum("...M,MD->...D", h, params["head"])

        def block(x, params):
            x += attn(collect(params["attn"]), norm(x)) / self.L
            x += mlp(collect(params["mlp"]), norm(x)) / self.L
            return x, None

        if self.grad_checkpoint:
            block = jax.checkpoint(block)

        x = get_embedding(collect(params["embd"]), x)
        x = lax.scan(block, x, params["blocks"], unroll=self.scan_unroll)[0]
        x = jnp.einsum("...D,DV->...V", norm(x), collect(params["head"]))
        return x

    @property
    def lrs(self):
        return jax.tree.map(lambda s: s.lr, self.spec, is_leaf=is_param)

    @property
    def n_params(self) -> int:
        _size = lambda s: prod(getattr(self, d) for d in s.shape)
        return sum(_size(s) for s in jax.tree.leaves(self.spec, is_leaf=is_param))

    @property
    def embd_params(self) -> int:
        return self.V * self.D

    def flops_per_token(self, seq_len: int) -> int:
        return 6 * self.n_params + 12 * self.L * self.D * seq_len
