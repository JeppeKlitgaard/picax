"""Mathematical and JAX-transformation properties of :func:`psnr`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from picax._jax.psnr import psnr

SHAPES = [(57,), (31, 35), (15, 17, 19)]


@pytest.mark.parametrize("shape", SHAPES)
def test_identical_inputs_give_infinity(make_pair, shape):
    x, _ = make_pair(shape)
    x = jnp.asarray(x)

    actual = psnr(x, x, dynamic_range=8.0)

    assert jnp.isinf(actual)
    assert actual > 0


def test_symmetric_in_its_arguments(make_pair):
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)

    forward = psnr(x, y, dynamic_range=8.0)
    reverse = psnr(y, x, dynamic_range=8.0)

    np.testing.assert_allclose(forward, reverse, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("noise", [0.05, 0.2, 1.0, 5.0])
def test_decreases_with_noise(make_pair, noise):
    x, quiet = make_pair((31, 35), noise=0.01)
    _, loud = make_pair((31, 35), noise=noise)
    x = jnp.asarray(x)

    assert psnr(x, jnp.asarray(quiet), dynamic_range=8.0) > psnr(
        x, jnp.asarray(loud), dynamic_range=8.0
    )


def test_invariant_under_common_rescaling(make_pair):
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)

    base = psnr(x, y, dynamic_range=8.0)
    scaled = psnr(2 * x, 2 * y, dynamic_range=16.0)

    np.testing.assert_allclose(scaled, base, rtol=1e-13, atol=1e-13)


def test_doubling_the_range_adds_six_decibels(make_pair):
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)

    base = psnr(x, y, dynamic_range=8.0)
    doubled = psnr(x, y, dynamic_range=16.0)

    np.testing.assert_allclose(doubled - base, 20 * np.log10(2), rtol=1e-12, atol=0)


class TestAxis:
    def test_none_reduces_every_axis(self, make_pair):
        x, y = make_pair((15, 17, 19))
        x, y = jnp.asarray(x), jnp.asarray(y)

        explicit = psnr(x, y, dynamic_range=8.0, axis=(0, 1, 2))

        np.testing.assert_array_equal(psnr(x, y, dynamic_range=8.0), explicit)

    def test_scalar_axis_matches_single_element_tuple(self, make_pair):
        x, y = make_pair((15, 17))
        x, y = jnp.asarray(x), jnp.asarray(y)

        np.testing.assert_array_equal(
            psnr(x, y, dynamic_range=8.0, axis=1),
            psnr(x, y, dynamic_range=8.0, axis=(1,)),
        )

    def test_negative_axis_counts_from_the_end(self, make_pair):
        x, y = make_pair((4, 15, 17))
        x, y = jnp.asarray(x), jnp.asarray(y)

        np.testing.assert_array_equal(
            psnr(x, y, dynamic_range=8.0, axis=(-2, -1)),
            psnr(x, y, dynamic_range=8.0, axis=(1, 2)),
        )

    def test_batch_reduction_matches_per_element_calls(self, make_pair):
        x, y = make_pair((4, 15, 17))
        x, y = jnp.asarray(x), jnp.asarray(y)

        batched = psnr(x, y, dynamic_range=8.0, axis=(1, 2))
        individual = [psnr(x[i], y[i], dynamic_range=8.0) for i in range(x.shape[0])]

        assert batched.shape == (4,)
        np.testing.assert_allclose(batched, jnp.stack(individual), rtol=0, atol=0)


class TestTransformations:
    def test_is_jittable(self, make_pair):
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        jitted = jax.jit(lambda a, b: psnr(a, b, dynamic_range=8.0))

        np.testing.assert_allclose(
            jitted(x, y), psnr(x, y, dynamic_range=8.0), rtol=1e-14, atol=1e-14
        )

    def test_is_differentiable(self, make_pair):
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        gradient = jax.grad(lambda a: psnr(a, y, dynamic_range=8.0))(x)

        assert gradient.shape == x.shape
        assert jnp.all(jnp.isfinite(gradient))

    def test_is_vmappable(self, make_pair):
        x, y = make_pair((4, 15, 17))
        x, y = jnp.asarray(x), jnp.asarray(y)

        mapped = jax.vmap(lambda a, b: psnr(a, b, dynamic_range=8.0))(x, y)

        np.testing.assert_allclose(
            mapped, psnr(x, y, dynamic_range=8.0, axis=(1, 2)), rtol=1e-14, atol=1e-14
        )

    def test_accepts_a_traced_dynamic_range(self, make_pair):
        x, y = make_pair((31, 35))
        x, y = jnp.asarray(x), jnp.asarray(y)

        jitted = jax.jit(lambda a, b, r: psnr(a, b, dynamic_range=r))

        np.testing.assert_allclose(
            jitted(x, y, jnp.asarray(8.0)),
            psnr(x, y, dynamic_range=8.0),
            rtol=1e-14,
            atol=1e-14,
        )
