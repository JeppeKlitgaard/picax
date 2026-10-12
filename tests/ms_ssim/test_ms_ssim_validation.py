"""Argument validation performed by :func:`ms_ssim`."""

import jax.numpy as jnp
import pytest

from picax._jax.ms_ssim import ms_ssim

PRESET = "wang-2003"

GAUSSIAN_11 = {
    "scale_weights": (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    "window_size": 11,
    "weights": "gaussian",
    "covariance_normalization": "population",
    "sigma": 1.5,
    "sigma_truncate": None,
    "pooling_region": "valid",
}


@pytest.fixture
def pair():
    x = jnp.zeros((171, 187))
    return x, x


INVALID_CONFIGS = [
    pytest.param(
        {"preset": "wang-2003", "scale_weights": (1.0,)},
        "all other parameters must be left as default",
        id="preset-with-scale-weights",
    ),
    pytest.param(
        {"preset": "wang-2003", "window_size": 7},
        "all other parameters must be left as default",
        id="preset-with-window-size",
    ),
    pytest.param({"preset": "nonexistent"}, "Unknown preset", id="unknown-preset"),
    pytest.param({}, "scale_weights must be specified", id="nothing-specified"),
    pytest.param(
        {"scale_weights": (1.0,)},
        "window_size must be specified",
        id="missing-window-size",
    ),
    pytest.param(
        {"scale_weights": (1.0,), "window_size": 7},
        "weights must be specified",
        id="missing-weights",
    ),
    pytest.param(
        {"scale_weights": (1.0,), "window_size": 7, "weights": "uniform"},
        "covariance_normalization must be specified",
        id="missing-covariance-normalization",
    ),
    pytest.param(
        {
            "scale_weights": (1.0,),
            "window_size": 7,
            "weights": "uniform",
            "covariance_normalization": "sample",
        },
        "sigma must be specified",
        id="missing-sigma",
    ),
    pytest.param(
        {
            "scale_weights": (1.0,),
            "window_size": 7,
            "weights": "uniform",
            "covariance_normalization": "sample",
            "sigma": None,
        },
        "sigma_truncate must be specified",
        id="missing-sigma-truncate",
    ),
    pytest.param(
        {
            "scale_weights": (1.0,),
            "window_size": 7,
            "weights": "uniform",
            "covariance_normalization": "sample",
            "sigma": None,
            "sigma_truncate": None,
        },
        "pooling_region must be specified",
        id="missing-pooling-region",
    ),
    pytest.param(
        {"preset": "wang-2003", "pooling_region": "full"},
        "all other parameters must be left as default",
        id="preset-with-pooling-region",
    ),
    pytest.param(
        GAUSSIAN_11 | {"pooling_region": "nonexistent"},
        "pooling_region must be 'full' or 'valid'",
        id="unknown-pooling-region",
    ),
    pytest.param(
        GAUSSIAN_11 | {"scale_weights": ()},
        "scale_weights must contain at least one weight",
        id="empty-scale-weights",
    ),
    pytest.param(
        GAUSSIAN_11 | {"scale_weights": (0.5, 0.0)},
        "scale_weights must be positive",
        id="zero-scale-weight",
    ),
    pytest.param(
        GAUSSIAN_11 | {"scale_weights": (0.5, -0.5)},
        "scale_weights must be positive",
        id="negative-scale-weight",
    ),
    pytest.param(
        GAUSSIAN_11 | {"weights": "nonexistent"},
        "Unknown weights",
        id="unknown-weights",
    ),
    pytest.param(
        GAUSSIAN_11 | {"window_size": 10},
        "window_size must be odd",
        id="even-gaussian-window",
    ),
    pytest.param(
        GAUSSIAN_11 | {"scale_weights": (1.0,) * 6},
        "window size 11 exceeds extent 6 along axis 0 at the coarsest of 6 scales",
        id="too-many-scales",
    ),
]


@pytest.mark.parametrize(("kwargs", "message"), INVALID_CONFIGS)
def test_invalid_configuration_raises(pair, kwargs, message):
    x, y = pair
    with pytest.raises(ValueError, match=message):
        ms_ssim(x, y, dynamic_range=8.0, **kwargs)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        pytest.param(
            (160, 187),
            "window size 11 exceeds extent 10 along axis 0",
            id="first-axis",
        ),
        pytest.param(
            (171, 160),
            "window size 11 exceeds extent 10 along axis 1",
            id="second-axis",
        ),
    ],
)
def test_input_too_small_for_the_preset_raises(shape, message):
    x = jnp.zeros(shape)
    with pytest.raises(ValueError, match=message):
        ms_ssim(x, x, dynamic_range=8.0, preset=PRESET)


def test_smallest_input_for_the_preset_is_accepted():
    """``ceil(161 / 16) == 11`` is exactly the window size."""
    x = jnp.zeros((161, 161))

    ms_ssim(x, x, dynamic_range=8.0, preset=PRESET)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="x and y must have same shape"):
        ms_ssim(
            jnp.zeros((171, 187)),
            jnp.zeros((171, 188)),
            dynamic_range=8.0,
            preset=PRESET,
        )


def test_scalar_inputs_raise():
    scalar = jnp.asarray(1.0)
    with pytest.raises(ValueError, match="at least one dimension"):
        ms_ssim(scalar, scalar, dynamic_range=8.0, preset=PRESET)


class TestDynamicRangeValidation:
    def test_unknown_string_raises(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="dynamic_range must be a float"):
            ms_ssim(x, y, dynamic_range="nonexistent", preset=PRESET)

    def test_dtype_mode_rejects_floating_point(self, pair):
        x, y = pair
        with pytest.raises(ValueError, match="must have an integer"):
            ms_ssim(x, y, dynamic_range="dtype", preset=PRESET)

    def test_dtype_mode_rejects_mismatched_dtypes(self):
        x = jnp.zeros((171, 187), dtype=jnp.int16)
        y = jnp.zeros((171, 187), dtype=jnp.int32)
        with pytest.raises(ValueError, match="must have the same"):
            ms_ssim(x, y, dynamic_range="dtype", preset=PRESET)
