"""Numerical parity of :func:`ms_ssim` against reference implementations."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat

from picax._jax.ms_ssim import ms_ssim
from picax._jax.ssim import ssim_mean

TOL = {"rtol": 1e-12, "atol": 1e-12}

# Output of Wang et al.'s msssim.m, written by fixtures/generate_msssim_matlab.m.
MATLAB_CASES = loadmat(
    Path(__file__).parent / "fixtures" / "msssim_matlab.mat", simplify_cells=True
)["cases"]

WANG_2003_SCALE_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

# Five scales with an 11-sample window need at least 161 samples along each axis.
PRESET_SHAPES = [(203,), (171, 187)]

# Fewer scales keep volumetric inputs small.
THREE_SCALE_SHAPES = [(57,), (45, 51), (45, 47, 49)]

# The explicit parameters each preset is meant to be shorthand for.
PRESET_PARAMETERS = {
    "wang-2003": {
        "scale_weights": WANG_2003_SCALE_WEIGHTS,
        "window_size": 11,
        "weights": "gaussian",
        "covariance_normalization": "population",
        "sigma": 1.5,
        "sigma_truncate": None,
        "pooling_region": "valid",
    },
}

# The explicit parameters of each SSIM preset, for comparison with ``ssim_mean``.
SSIM_PRESET_PARAMETERS = {
    "skimage-default": {
        "window_size": 7,
        "weights": "uniform",
        "covariance_normalization": "sample",
        "sigma": None,
        "sigma_truncate": None,
        "pooling_region": "valid",
    },
    "wang-2004-implementation": {
        "window_size": 11,
        "weights": "gaussian",
        "covariance_normalization": "population",
        "sigma": 1.5,
        "sigma_truncate": None,
        "pooling_region": "valid",
    },
    "wang-2004-paper": {
        "window_size": 11,
        "weights": "gaussian",
        "covariance_normalization": "sample",
        "sigma": 1.5,
        "sigma_truncate": None,
        "pooling_region": "valid",
    },
}


def _reference_downsample(a):
    """Port of ``imfilter(a, ones(2) / 4, 'symmetric')`` then ``a(1:2:end, ...)``."""
    a = np.pad(a, [(0, n % 2) for n in a.shape], mode="edge")
    for axis in range(a.ndim):
        a = 0.5 * (
            a.take(range(0, a.shape[axis], 2), axis=axis)
            + a.take(range(1, a.shape[axis], 2), axis=axis)
        )
    return a


def _reference_filter(a, *, size=11, sigma=1.5):
    """Port of ``filter2(fspecial('gaussian', size, sigma), a, 'valid')``.

    The Gaussian window is separable, so it is applied along one axis at a time.
    Only windows lying entirely inside ``a`` are evaluated, so no boundary mode is
    involved.
    """
    offsets = np.arange(size) - size // 2
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    for axis in range(a.ndim):
        windows = sliding_window_view(np.moveaxis(a, axis, -1), size, axis=-1)
        a = np.moveaxis(windows @ kernel, -1, axis)
    return a


def _reference_ms_ssim(x, y, *, data_range, scale_weights, K=(0.01, 0.03)):
    """Port of Wang et al.'s ``msssim.m`` and ``ssim_index_new.m`` to NumPy."""
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    result = 1.0
    for scale, weight in enumerate(scale_weights):
        if scale > 0:
            x, y = _reference_downsample(x), _reference_downsample(y)

        mu_x, mu_y = _reference_filter(x), _reference_filter(y)
        sigma_xx = _reference_filter(x * x) - mu_x**2
        sigma_yy = _reference_filter(y * y) - mu_y**2
        sigma_xy = _reference_filter(x * y) - mu_x * mu_y

        cs_map = (2 * sigma_xy + C2) / (sigma_xx + sigma_yy + C2)
        if scale == len(scale_weights) - 1:
            luminance = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
            value = np.mean(luminance * cs_map)
        else:
            value = np.mean(cs_map)

        result *= value**weight

    return result


@pytest.mark.parametrize("shape", PRESET_SHAPES)
def test_preset_matches_reference(make_pair, shape):
    x, y = make_pair(shape)
    dynamic_range = 8.0

    expected = _reference_ms_ssim(
        x, y, data_range=dynamic_range, scale_weights=WANG_2003_SCALE_WEIGHTS
    )
    actual = ms_ssim(
        jnp.asarray(x),
        jnp.asarray(y),
        dynamic_range=dynamic_range,
        preset="wang-2003",
    )

    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("case", MATLAB_CASES, ids=lambda case: case["name"])
def test_matches_matlab(case):
    """Agree with ``msssim.m`` itself, which hardcodes a dynamic range of 255."""
    K1, K2 = case["K"]
    parameters = PRESET_PARAMETERS["wang-2003"] | {
        "scale_weights": tuple(np.atleast_1d(case["weight"]))
    }

    actual = ms_ssim(
        jnp.asarray(case["x"]),
        jnp.asarray(case["y"]),
        dynamic_range="dtype",
        K1=K1,
        K2=K2,
        **parameters,
    )

    assert case["x"].dtype == np.uint8
    np.testing.assert_allclose(actual, case["expected"], **TOL)


@pytest.mark.parametrize("case", MATLAB_CASES, ids=lambda case: case["name"])
def test_reference_port_matches_matlab(case):
    """The NumPy port used for inputs MATLAB was not run on is itself faithful."""
    expected = _reference_ms_ssim(
        case["x"].astype(np.float64),
        case["y"].astype(np.float64),
        data_range=255.0,
        scale_weights=np.atleast_1d(case["weight"]),
        K=case["K"],
    )

    np.testing.assert_allclose(expected, case["expected"], **TOL)


@pytest.mark.parametrize("shape", THREE_SCALE_SHAPES)
def test_fewer_scales_match_reference(make_pair, shape):
    x, y = make_pair(shape)
    scale_weights = (0.2, 0.5, 0.3)
    parameters = PRESET_PARAMETERS["wang-2003"] | {"scale_weights": scale_weights}

    expected = _reference_ms_ssim(x, y, data_range=8.0, scale_weights=scale_weights)
    actual = ms_ssim(jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, **parameters)

    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("shape", THREE_SCALE_SHAPES)
@pytest.mark.parametrize("pooling_region", ["full", "valid"])
@pytest.mark.parametrize("ssim_preset", list(SSIM_PRESET_PARAMETERS))
def test_single_scale_matches_ssim_mean(make_pair, ssim_preset, pooling_region, shape):
    """With one unit-weight scale, MS-SSIM reduces to the mean SSIM."""
    x, y = make_pair(shape)
    x, y = jnp.asarray(x), jnp.asarray(y)

    parameters = SSIM_PRESET_PARAMETERS[ssim_preset] | {
        "pooling_region": pooling_region
    }

    expected = ssim_mean(x, y, dynamic_range=8.0, **parameters)
    actual = ms_ssim(x, y, dynamic_range=8.0, scale_weights=(1.0,), **parameters)

    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("preset", list(PRESET_PARAMETERS))
def test_preset_matches_its_explicit_parameters(make_pair, preset):
    x, y = make_pair((171, 187))
    x, y = jnp.asarray(x), jnp.asarray(y)

    from_preset = ms_ssim(x, y, dynamic_range=8.0, preset=preset)
    from_parameters = ms_ssim(x, y, dynamic_range=8.0, **PRESET_PARAMETERS[preset])

    np.testing.assert_array_equal(from_preset, from_parameters)
