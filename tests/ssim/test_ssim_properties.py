"""Mathematical and JAX-transformation properties of :func:`ssim_local`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from picax._jax.ssim import ssim_local

PRESETS = ["skimage-default", "wang-2004-implementation", "wang-2004-paper"]
SHAPES = [(57,), (31, 35), (15, 17, 19)]


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("shape", SHAPES)
def test_identical_inputs_give_unity(make_pair, preset, shape):
    x, _ = make_pair(shape)
    x = jnp.asarray(x)

    actual = ssim_local(x, x, dynamic_range=8.0, preset=preset)

    np.testing.assert_allclose(actual, np.ones(shape), rtol=0, atol=1e-12)


@pytest.mark.parametrize("preset", PRESETS)
def test_symmetric_in_its_arguments(make_pair, preset):
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)

    forward = ssim_local(x, y, dynamic_range=8.0, preset=preset)
    reverse = ssim_local(y, x, dynamic_range=8.0, preset=preset)

    np.testing.assert_allclose(forward, reverse, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("noise", [0.0, 0.2, 5.0])
def test_bounded_by_unity(make_pair, preset, noise):
    x, y = make_pair((31, 35), noise=noise)

    actual = ssim_local(
        jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, preset=preset
    )

    assert np.all(np.abs(np.asarray(actual)) <= 1.0 + 1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_preserves_shape(make_pair, shape):
    x, y = make_pair(shape)

    actual = ssim_local(
        jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, preset="skimage-default"
    )

    assert actual.shape == shape


def test_degrades_monotonically_with_noise(make_pair):
    means = []
    for noise in (0.1, 0.5, 2.0):
        x, y = make_pair((31, 35), noise=noise)
        actual = ssim_local(
            jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, preset="skimage-default"
        )
        means.append(float(jnp.mean(actual)))

    assert means[0] > means[1] > means[2]


def test_constant_inputs_reduce_to_the_luminance_term():
    """With zero local variance both contrast and structure are exactly one."""
    x = jnp.full((21, 21), 3.0)
    y = jnp.full((21, 21), 5.0)
    dynamic_range = 8.0
    C1 = (0.01 * dynamic_range) ** 2

    actual = ssim_local(x, y, dynamic_range=dynamic_range, preset="skimage-default")

    expected = (2 * 3.0 * 5.0 + C1) / (3.0**2 + 5.0**2 + C1)
    np.testing.assert_allclose(actual, np.full((21, 21), expected), rtol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_promotes_to_double_precision(make_pair, dtype):
    """Accumulation happens in float64 regardless of the input precision."""
    x, y = make_pair((31, 35))
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)

    actual = ssim_local(x, y, dynamic_range=8.0, preset="skimage-default")

    assert actual.dtype == jnp.float64


class TestDynamicRange:
    def test_range_matches_the_equivalent_literal(self, make_pair):
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)
        expected_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))

        inferred = ssim_local(x, y, dynamic_range="range", preset="skimage-default")
        literal = ssim_local(
            x, y, dynamic_range=expected_range, preset="skimage-default"
        )

        np.testing.assert_array_equal(inferred, literal)

    def test_dtype_matches_the_equivalent_literal(self):
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.integers(-2000, 2000, size=(31, 35)), dtype=jnp.int16)
        y = x + jnp.asarray(rng.integers(-50, 50, size=(31, 35)), dtype=jnp.int16)

        inferred = ssim_local(x, y, dynamic_range="dtype", preset="skimage-default")
        literal = ssim_local(x, y, dynamic_range=65535.0, preset="skimage-default")

        np.testing.assert_array_equal(inferred, literal)

    def test_scales_the_stabilizing_constants(self, make_pair):
        """Doubling both the data and the range leaves the map unchanged."""
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        base = ssim_local(x, y, dynamic_range=8.0, preset="skimage-default")
        scaled = ssim_local(2 * x, 2 * y, dynamic_range=16.0, preset="skimage-default")

        np.testing.assert_allclose(base, scaled, rtol=1e-12, atol=1e-12)


class TestJaxTransformations:
    def test_jit(self, make_pair):
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        def compute(a, b):
            return ssim_local(a, b, dynamic_range=8.0, preset="skimage-default")

        # XLA is free to fuse differently, so this is not bit-exact.
        np.testing.assert_allclose(
            jax.jit(compute)(x, y), compute(x, y), rtol=1e-12, atol=1e-12
        )

    def test_grad_is_finite(self, make_pair):
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        def mean_ssim(a):
            return ssim_local(a, y, dynamic_range=8.0, preset="skimage-default").mean()

        gradient = jax.grad(mean_ssim)(x)

        assert gradient.shape == x.shape
        assert np.all(np.isfinite(np.asarray(gradient)))

    def test_vmap_over_a_batch_axis(self, make_pair):
        """All axes are spatial, so batching is expressed with ``vmap``."""
        x, y = make_pair((4, 31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        def compute(a, b):
            return ssim_local(a, b, dynamic_range=8.0, preset="skimage-default")

        batched = jax.vmap(compute)(x, y)
        expected = jnp.stack([compute(a, b) for a, b in zip(x, y)])

        np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-12)
