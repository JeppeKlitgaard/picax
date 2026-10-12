"""Mathematical and JAX-transformation properties of :func:`ms_ssim`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from picax._jax.ms_ssim import _downsample, ms_ssim

PRESET = "wang-2003"
SHAPE = (171, 187)

# Three scales keep volumetric inputs small.
THREE_SCALES = {
    "scale_weights": (0.2, 0.5, 0.3),
    "window_size": 11,
    "weights": "gaussian",
    "covariance_normalization": "population",
    "sigma": 1.5,
    "sigma_truncate": None,
    "pooling_region": "valid",
}
THREE_SCALE_SHAPES = [(57,), (45, 51), (45, 47, 49)]


@pytest.mark.parametrize("shape", THREE_SCALE_SHAPES)
def test_identical_inputs_give_unity(make_pair, shape):
    x, _ = make_pair(shape)
    x = jnp.asarray(x)

    actual = ms_ssim(x, x, dynamic_range=8.0, **THREE_SCALES)

    np.testing.assert_allclose(actual, 1.0, rtol=0, atol=1e-12)


def test_identical_inputs_give_unity_for_preset(make_pair):
    x, _ = make_pair(SHAPE)
    x = jnp.asarray(x)

    actual = ms_ssim(x, x, dynamic_range=8.0, preset=PRESET)

    np.testing.assert_allclose(actual, 1.0, rtol=0, atol=1e-12)


def test_symmetric_in_its_arguments(make_pair):
    x, y = make_pair(SHAPE)
    x, y = jnp.asarray(x), jnp.asarray(y)

    forward = ms_ssim(x, y, dynamic_range=8.0, preset=PRESET)
    reverse = ms_ssim(y, x, dynamic_range=8.0, preset=PRESET)

    np.testing.assert_allclose(forward, reverse, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("noise", [0.0, 0.2, 5.0])
def test_bounded_by_unity(make_pair, noise):
    x, y = make_pair(SHAPE, noise=noise)

    actual = ms_ssim(jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, preset=PRESET)

    assert 0.0 < float(actual) <= 1.0 + 1e-12


@pytest.mark.parametrize("shape", THREE_SCALE_SHAPES)
def test_returns_a_scalar(make_pair, shape):
    x, y = make_pair(shape)

    actual = ms_ssim(jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, **THREE_SCALES)

    assert actual.shape == ()


def test_degrades_monotonically_with_noise(make_pair):
    values = []
    for noise in (0.1, 0.5, 2.0):
        x, y = make_pair(SHAPE, noise=noise)
        actual = ms_ssim(
            jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, preset=PRESET
        )
        values.append(float(actual))

    assert values[0] > values[1] > values[2]


def test_scale_weights_act_as_exponents(make_pair):
    """Scaling every weight by ``k`` raises the index to the ``k``-th power."""
    x, y = make_pair((45, 51))
    x, y = jnp.asarray(x), jnp.asarray(y)
    doubled = THREE_SCALES | {
        "scale_weights": tuple(2 * w for w in THREE_SCALES["scale_weights"])
    }

    base = ms_ssim(x, y, dynamic_range=8.0, **THREE_SCALES)
    squared = ms_ssim(x, y, dynamic_range=8.0, **doubled)

    np.testing.assert_allclose(squared, base**2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_promotes_to_double_precision(make_pair, dtype):
    """Accumulation happens in float64 regardless of the input precision."""
    x, y = make_pair(SHAPE)
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)

    actual = ms_ssim(x, y, dynamic_range=8.0, preset=PRESET)

    assert actual.dtype == jnp.float64


class TestDownsample:
    @pytest.mark.parametrize("shape", [(8, 10), (7, 9), (5, 6, 7)])
    def test_rounds_every_extent_up(self, shape):
        actual = _downsample(jnp.zeros(shape))

        assert actual.shape == tuple(-(-extent // 2) for extent in shape)

    def test_averages_adjacent_pairs_and_keeps_an_odd_trailing_sample(self):
        x = jnp.asarray([1.0, 3.0, 5.0, 9.0, 4.0])

        np.testing.assert_array_equal(_downsample(x), [2.0, 7.0, 4.0])

    def test_averages_two_by_two_blocks(self):
        x = jnp.arange(16.0).reshape(4, 4)

        expected = [[2.5, 4.5], [10.5, 12.5]]
        np.testing.assert_array_equal(_downsample(x), expected)


class TestDynamicRange:
    def test_range_matches_the_full_resolution_literal(self, make_pair):
        """The range is inferred once, from the inputs before downsampling."""
        x, y = make_pair(SHAPE)
        x, y = jnp.asarray(x), jnp.asarray(y)
        expected_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))

        inferred = ms_ssim(x, y, dynamic_range="range", preset=PRESET)
        literal = ms_ssim(x, y, dynamic_range=expected_range, preset=PRESET)

        np.testing.assert_array_equal(inferred, literal)

    def test_dtype_matches_the_equivalent_literal(self):
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.integers(-2000, 2000, size=SHAPE), dtype=jnp.int16)
        y = x + jnp.asarray(rng.integers(-50, 50, size=SHAPE), dtype=jnp.int16)

        inferred = ms_ssim(x, y, dynamic_range="dtype", preset=PRESET)
        literal = ms_ssim(x, y, dynamic_range=65535.0, preset=PRESET)

        np.testing.assert_array_equal(inferred, literal)

    def test_scales_the_stabilizing_constants(self, make_pair):
        """Doubling both the data and the range leaves the index unchanged."""
        x, y = make_pair(SHAPE)
        x, y = jnp.asarray(x), jnp.asarray(y)

        base = ms_ssim(x, y, dynamic_range=8.0, preset=PRESET)
        scaled = ms_ssim(2 * x, 2 * y, dynamic_range=16.0, preset=PRESET)

        np.testing.assert_allclose(base, scaled, rtol=1e-12, atol=1e-12)


class TestJaxTransformations:
    def test_jit(self, make_pair):
        x, y = make_pair(SHAPE)
        x, y = jnp.asarray(x), jnp.asarray(y)

        def compute(a, b):
            return ms_ssim(a, b, dynamic_range=8.0, preset=PRESET)

        # XLA is free to fuse differently, so this is not bit-exact.
        np.testing.assert_allclose(
            jax.jit(compute)(x, y), compute(x, y), rtol=1e-12, atol=1e-12
        )

    def test_grad_is_finite(self, make_pair):
        x, y = make_pair((45, 51))
        x, y = jnp.asarray(x), jnp.asarray(y)

        def compute(a):
            return ms_ssim(a, y, dynamic_range=8.0, **THREE_SCALES)

        gradient = jax.grad(compute)(x)

        assert gradient.shape == x.shape
        assert np.all(np.isfinite(np.asarray(gradient)))

    def test_vmap_over_a_batch_axis(self, make_pair):
        """All axes are spatial, so batching is expressed with ``vmap``."""
        x, y = make_pair((4, 45, 51))
        x, y = jnp.asarray(x), jnp.asarray(y)

        def compute(a, b):
            return ms_ssim(a, b, dynamic_range=8.0, **THREE_SCALES)

        batched = jax.vmap(compute)(x, y)
        expected = jnp.stack([compute(a, b) for a, b in zip(x, y)])

        np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-12)
