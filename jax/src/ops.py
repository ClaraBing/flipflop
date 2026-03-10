import warnings
from functools import partial

import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array


@partial(jax.jit, static_argnames=("dtype",))
def rotary_embedding(x: Array, freq: float, dtype: jnp.dtype | None = None) -> Array:
    dtype = dtype or x.dtype
    out = x.astype(dtype)
    T, _, K = out.shape[-3:]
    if K % 2:
        raise ValueError(f"rotary_embedding expects even K, got {K}")
    half_K = K // 2
    with jax.ensure_compile_time_eval():
        pos = jnp.arange(T, dtype=dtype)[:, None, None]
        idx = jnp.arange(half_K, dtype=dtype)[None, None, :]
        freq_arr = jnp.asarray(freq, dtype=dtype)
        theta = pos * freq_arr ** (-idx / half_K)
        theta_cis = jnp.exp(1j * theta)
    out = out[..., :half_K] + 1j * out[..., half_K:]
    out *= theta_cis
    return jnp.concatenate([out.real, out.imag], axis=-1, dtype=x.dtype)


@partial(jax.jit, static_argnames=("dtype",))
def rmsnorm(x: Array, eps: float, dtype: jnp.dtype | None = None) -> Array:
    out = x.astype(dtype)
    out *= lax.rsqrt(jnp.mean(out**2, axis=-1, keepdims=True) + eps)
    return out.astype(x.dtype)


@jax.jit
def get_embedding(w, x):
    out_spec = jax.typeof(x).sharding
    w_dtype = w.dtype
    if w_dtype in [jnp.bfloat16, jnp.float16]:
        w = w.astype(jnp.float32)
    out = w.at[x].get(out_sharding=out_spec)
    return out.astype(w_dtype)


@partial(jax.jit, static_argnames=("is_causal", "scale"))
def flash_attention(q, k, v, *, is_causal=False, scale=None):
    def attn(q, k, v):
        return jax.nn.dot_product_attention(
            q, k, v, implementation="cudnn", is_causal=is_causal, scale=scale
        )

    sharding = jax.typeof(q).sharding
    if sharding.spec[0] is None:
        return attn(q, k, v)

    spec = P(sharding.spec[0])
    return jax.shard_map(
        attn,
        mesh=sharding.mesh,
        in_specs=(spec,) * 3,
        out_specs=spec,
        check_vma=False,
    )(q, k, v)


_flash_attn_warned = False


def sdpa(q, k, v, *, is_causal=False, scale=None, implementation="flash"):
    assert implementation in ("xla", "flash")
    if implementation == "xla":
        return jax.nn.dot_product_attention(
            q, k, v, implementation="xla", is_causal=is_causal, scale=scale
        )
    try:
        return flash_attention(q, k, v, is_causal=is_causal, scale=scale)
    except Exception as e:
        global _flash_attn_warned
        if not _flash_attn_warned:
            warnings.warn(f"Flash Attention failed ({e}), falling back to XLA")
            _flash_attn_warned = True
        return jax.nn.dot_product_attention(
            q, k, v, implementation="xla", is_causal=is_causal, scale=scale
        )
