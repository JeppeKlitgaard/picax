"""Argument validation performed by :func:`ssim_local`."""

import jax.numpy as jnp
import pytest

from picax._jax.ssim import ssim_local

UNIFORM_7 = {
    "window_size": 7,
    "weights": "uniform",
    "covariance_normalization": "sample",
    "sigma": None,
    "sigma_truncate": None,
}
GAUSSIAN_11 = {
    "window_size": 11,
    "weights": "gaussian",
    "covariance_normalization": "population",
    "sigma": 1.5,
    "sigma_truncate": None,
}


@pytest.fixture
def pair():
    x = jnp.zeros((31, 35))
    return x, x


INVALID_CONFIGS = [
    pytest.param(
        {"preset": "skimage-default", "window_size": 9},
        "all other parameters must be left as default",
        id="preset-with-override",
    ),
    pytest.param({"preset": "nonexistent"}, "Unknown preset", id="unknown-preset"),
    pytest.param({}, "window_size must be specified", id="nothing-specified"),
    pytest.param(
        {"window_size": 7},
        "weights must be specified",
        id="missing-weights",
    ),
    pytest.param(
        {"window_size": 7, "weights": "uniform"},
        "covariance_normalization must be specified",
        id="missing-covariance-normalization",
    ),
    pytest.param(
        {"window_size": 7, "weights": "uniform", "covariance_normalization": "sample"},
        "sigma must be specified",
        id="missing-sigma",
    ),
    pytest.param(
        {
            "window_size": 7,
            "weights": "uniform",
            "covariance_normalization": "sample",
            "sigma": None,
        },
        "sigma_truncate must be specified",
        id="missing-sigma-truncate",
    ),
    pytest.param(
        UNIFORM_7 | {"weights": "nonexistent"}, "Unknown weights", id="unknown-weights"
    ),
    pytest.param(
        UNIFORM_7 | {"covariance_normalization": "nonexistent"},
        "covariance_normalization must be 'sample' or 'population'",
        id="unknown-covariance-normalization",
    ),
    pytest.param(
        UNIFORM_7 | {"sigma": 1.5},
        "sigma must be None when weights is 'uniform'",
        id="uniform-with-sigma",
    ),
    pytest.param(
        UNIFORM_7 | {"sigma_truncate": 3.5},
        "sigma_truncate must be None when weights is 'uniform'",
        id="uniform-with-sigma-truncate",
    ),
    pytest.param(
        UNIFORM_7 | {"window_size": None},
        "window_size must be specified when weights is 'uniform'",
        id="uniform-without-window-size",
    ),
    pytest.param(
        UNIFORM_7 | {"window_size": 0}, "window_size must be positive", id="zero-window"
    ),
    pytest.param(
        GAUSSIAN_11 | {"sigma": None},
        "sigma must be specified when weights is 'gaussian'",
        id="gaussian-without-sigma",
    ),
    pytest.param(
        GAUSSIAN_11 | {"sigma": -1.0}, "sigma must be positive", id="negative-sigma"
    ),
    pytest.param(
        GAUSSIAN_11 | {"window_size": 10},
        "window_size must be odd",
        id="even-gaussian-window",
    ),
    pytest.param(
        GAUSSIAN_11 | {"sigma_truncate": 3.5},
        "sigma_truncate must be None when window_size is specified",
        id="gaussian-window-size-and-truncate",
    ),
    pytest.param(
        GAUSSIAN_11 | {"window_size": None, "sigma_truncate": None},
        "sigma_truncate must be specified when window_size is None",
        id="gaussian-neither-window-nor-truncate",
    ),
    pytest.param(
        GAUSSIAN_11 | {"window_size": None, "sigma_truncate": -1.0},
        "sigma_truncate must be positive",
        id="negative-sigma-truncate",
    ),
    pytest.param(
        UNIFORM_7 | {"window_size": (7,)},
        "window_size must contain one size per input axis",
        id="window-size-wrong-length",
    ),
    pytest.param(
        UNIFORM_7 | {"window_size": (7, 99)},
        "window size 99 exceeds extent 35 along axis 1",
        id="window-larger-than-input",
    ),
    pytest.param(
        UNIFORM_7 | {"alpha": 0}, "alpha must be positive", id="non-positive-alpha"
    ),
    pytest.param(
        UNIFORM_7 | {"beta": -1}, "beta must be positive", id="non-positive-beta"
    ),
    pytest.param(
        UNIFORM_7 | {"gamma": 0}, "gamma must be positive", id="non-positive-gamma"
    ),
]


@pytest.mark.parametrize(("kwargs", "message"), INVALID_CONFIGS)
def test_invalid_configuration_raises(pair, kwargs, message):
    x, y = pair
    with pytest.raises(ValueError, match=message):
        ssim_local(x, y, dynamic_range=8.0, **kwargs)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="x and y must have same shape"):
        ssim_local(
            jnp.zeros((31, 35)),
            jnp.zeros((31, 36)),
            dynamic_range=8.0,
            preset="skimage-default",
        )


def test_scalar_inputs_raise():
    scalar = jnp.asarray(1.0)
    with pytest.raises(ValueError, match="at least one dimension"):
        ssim_local(scalar, scalar, dynamic_range=8.0, preset="skimage-default")


class TestDynamicRangeValidation:
    def test_unknown_string_raises(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="dynamic_range must be a float"):
            ssim_local(x, y, dynamic_range="nonexistent", preset="skimage-default")

    def test_dtype_mode_rejects_floating_point(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="must have an integer"):
            ssim_local(x, y, dynamic_range="dtype", preset="skimage-default")

    def test_dtype_mode_rejects_mismatched_dtypes(self):
        x = jnp.zeros((31, 35), dtype=jnp.int16)
        y = jnp.zeros((31, 35), dtype=jnp.int32)
        with pytest.raises(ValueError, match="must have the same"):
            ssim_local(x, y, dynamic_range="dtype", preset="skimage-default")
