import jax
import numpy as np
import pytest

# Must run before any array is created: parity with the float64 reference
# implementations is only meaningful at double precision.
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def make_pair():
    """Return a factory for a correlated ``(x, y)`` pair of a given shape."""

    def _make_pair(shape, *, noise=0.2, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=shape)
        y = x + noise * rng.normal(size=shape)
        return x, y

    return _make_pair
