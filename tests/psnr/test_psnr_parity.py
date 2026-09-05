"""Numerical parity of :func:`psnr` against established implementations."""

import jax.numpy as jnp
import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from picax._jax.psnr import psnr

TOL = {"rtol": 1e-12, "atol": 1e-12}

SHAPES = [(57,), (31, 35), (15, 17, 19)]


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_skimage(make_pair, shape):
    x, y = make_pair(shape)

    expected = peak_signal_noise_ratio(x, y, data_range=8.0)
    actual = psnr(jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0)

    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("dtype", [jnp.uint8, jnp.int16])
def test_matches_skimage_for_integer_dtypes(dtype):
    rng = np.random.default_rng(0)
    info = jnp.iinfo(dtype)
    x = rng.integers(info.min, info.max, size=(31, 35), endpoint=True)
    y = rng.integers(info.min, info.max, size=(31, 35), endpoint=True)
    x, y = x.astype(dtype), y.astype(dtype)

    expected = peak_signal_noise_ratio(x, y)
    actual = psnr(jnp.asarray(x), jnp.asarray(y), dynamic_range="dtype")

    np.testing.assert_allclose(actual, expected, **TOL)


def test_matches_skimage_for_inferred_range(make_pair):
    x, y = make_pair((31, 35))
    data_range = max(x.max(), y.max()) - min(x.min(), y.min())

    expected = peak_signal_noise_ratio(x, y, data_range=data_range)
    actual = psnr(jnp.asarray(x), jnp.asarray(y), dynamic_range="range")

    np.testing.assert_allclose(actual, expected, **TOL)


def test_matches_definition(make_pair):
    x, y = make_pair((31, 35))
    dynamic_range = 8.0

    mean_squared_error = np.mean((x - y) ** 2)
    expected = 10 * np.log10(dynamic_range**2 / mean_squared_error)
    actual = psnr(jnp.asarray(x), jnp.asarray(y), dynamic_range=dynamic_range)

    np.testing.assert_allclose(actual, expected, **TOL)
