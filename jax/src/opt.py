from dataclasses import dataclass
from math import sqrt

import jax
import optax
from jax import numpy as jnp


def tree_transform(fn, *rest):
    return optax.stateless(lambda g, _: jax.tree.map(fn, g, *rest))


@dataclass
class Adam:
    lr: float = 1.0
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8

    def build(self, lrs) -> optax.GradientTransformation:
        return optax.chain(
            tree_transform(lambda g, lr: g * sqrt(lr * g.size), lrs),
            optax.adam(
                self.lr, b1=self.b1, b2=self.b2, eps=self.eps, mu_dtype=jnp.float32
            ),
            tree_transform(lambda g, lr: g * sqrt(lr / g.size), lrs),
        )


@dataclass
class SGD:
    lr: float = 1.0
    momentum: float = 0.9

    def build(self, lrs) -> optax.GradientTransformation:
        return optax.chain(
            tree_transform(lambda g, lr: g * lr, lrs),
            optax.sgd(self.lr, momentum=self.momentum),
        )
