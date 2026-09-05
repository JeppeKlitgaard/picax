"""Argument validation performed by :func:`psnr`."""

import jax.numpy as jnp
import pytest

from picax._jax.psnr import psnr


@pytest.fixture
def pair():
    x = jnp.zeros((31, 35))
    return x, x


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="x and y must have same shape"):
        psnr(jnp.zeros((31, 35)), jnp.zeros((31, 36)), dynamic_range=8.0)


def test_scalar_inputs_raise():
    scalar = jnp.asarray(1.0)
    with pytest.raises(ValueError, match="at least one dimension"):
        psnr(scalar, scalar, dynamic_range=8.0)


class TestDynamicRangeValidation:
    def test_unknown_string_raises(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="dynamic_range must be a float"):
            psnr(x, y, dynamic_range="nonexistent")

    def test_non_numeric_raises(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="dynamic_range must be a float"):
            psnr(x, y, dynamic_range=None)

    def test_non_scalar_raises(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="dynamic_range must be a scalar"):
            psnr(x, y, dynamic_range=jnp.asarray([8.0, 16.0]))

    def test_dtype_mode_rejects_floating_point(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="must have an integer"):
            psnr(x, y, dynamic_range="dtype")

    def test_dtype_mode_rejects_mismatched_dtypes(self):
        x = jnp.zeros((31, 35), dtype=jnp.int16)
        y = jnp.zeros((31, 35), dtype=jnp.int32)
        with pytest.raises(ValueError, match="must have the same"):
            psnr(x, y, dynamic_range="dtype")


class TestAxisValidation:
    @pytest.mark.parametrize("axis", [2, -3, (0, 2)])
    def test_out_of_bounds_raises(self, pair, axis):
        x, y = pair
        with pytest.raises(ValueError, match="axis is out of bounds"):
            psnr(x, y, dynamic_range=8.0, axis=axis)

    @pytest.mark.parametrize("axis", [(0, 0), (0, -2)])
    def test_duplicate_entries_raise(self, pair, axis):
        x, y = pair
        with pytest.raises(ValueError, match="must not contain duplicate entries"):
            psnr(x, y, dynamic_range=8.0, axis=axis)

    def test_wrong_type_raises(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="axis must be an int"):
            psnr(x, y, dynamic_range=8.0, axis="0")
