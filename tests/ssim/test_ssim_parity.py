"""Numerical parity of :func:`ssim_local` against established implementations."""

import jax.numpy as jnp
import numpy as np
import pytest
from skimage.metrics import structural_similarity

from picax._jax.ssim import ssim_local

TOL = {"rtol": 1e-12, "atol": 1e-12}

SHAPES = [(57,), (31, 35), (15, 17, 19)]

# Each preset alongside the skimage arguments that reproduce it.
PRESETS = [
    pytest.param("skimage-default", {}, id="skimage-default"),
    pytest.param(
        "wang-2004-implementation",
        {"gaussian_weights": True, "sigma": 1.5, "use_sample_covariance": False},
        id="wang-2004-implementation",
    ),
]
# "wang-2004-paper" has no skimage counterpart: for Gaussian weights skimage
# applies the unweighted Bessel correction NP / (NP - 1), whereas the paper's
# estimator uses the weighted form 1 / (1 - sum_i w_i**2). Until there is an
# implementation to compare against, it is only checked for consistency with its
# own explicit parameters below.

# The explicit parameters each preset is meant to be shorthand for.
PRESET_PARAMETERS = {
    "skimage-default": {
        "window_size": 7,
        "weights": "uniform",
        "covariance_normalization": "sample",
        "sigma": None,
        "sigma_truncate": None,
    },
    "wang-2004-implementation": {
        "window_size": 11,
        "weights": "gaussian",
        "covariance_normalization": "population",
        "sigma": 1.5,
        "sigma_truncate": None,
    },
    "wang-2004-paper": {
        "window_size": 11,
        "weights": "gaussian",
        "covariance_normalization": "sample",
        "sigma": 1.5,
        "sigma_truncate": None,
    },
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(("preset", "skimage_options"), PRESETS)
def test_preset_matches_skimage(make_pair, preset, skimage_options, shape):
    x, y = make_pair(shape)
    dynamic_range = 8.0

    _, expected = structural_similarity(
        x, y, data_range=dynamic_range, full=True, **skimage_options
    )
    actual = ssim_local(
        jnp.asarray(x),
        jnp.asarray(y),
        dynamic_range=dynamic_range,
        preset=preset,
    )

    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("preset", list(PRESET_PARAMETERS))
def test_preset_matches_its_explicit_parameters(make_pair, preset, shape):
    x, y = make_pair(shape)
    x, y = jnp.asarray(x), jnp.asarray(y)

    from_preset = ssim_local(x, y, dynamic_range=8.0, preset=preset)
    from_parameters = ssim_local(x, y, dynamic_range=8.0, **PRESET_PARAMETERS[preset])

    np.testing.assert_array_equal(from_preset, from_parameters)


def test_window_size_inferred_from_sigma(make_pair):
    """``sigma=1.5`` truncated at 3.5 standard deviations gives an 11-tap window."""
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)

    inferred = ssim_local(
        x,
        y,
        dynamic_range=8.0,
        window_size=None,
        weights="gaussian",
        covariance_normalization="population",
        sigma=1.5,
        sigma_truncate=3.5,
    )
    explicit = ssim_local(x, y, dynamic_range=8.0, preset="wang-2004-implementation")

    np.testing.assert_allclose(inferred, explicit, **TOL)


@pytest.mark.parametrize(
    ("scalar", "per_axis"),
    [
        pytest.param({"sigma": 2}, {"sigma": (2.0, 2.0)}, id="int-sigma"),
        pytest.param({"sigma": 2.0}, {"sigma": (2.0, 2.0)}, id="float-sigma"),
        pytest.param(
            {"sigma": 1.5, "sigma_truncate": 4},
            {"sigma": (1.5, 1.5), "sigma_truncate": (4.0, 4.0)},
            id="int-sigma-truncate",
        ),
        pytest.param(
            {"sigma": 1.5, "sigma_truncate": 4.0},
            {"sigma": (1.5, 1.5), "sigma_truncate": (4.0, 4.0)},
            id="float-sigma-truncate",
        ),
    ],
)
def test_scalars_broadcast_across_axes(make_pair, scalar, per_axis):
    """A scalar ``sigma``/``sigma_truncate`` repeats over every spatial axis."""
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)
    # An explicit window size and an inferred one take different code paths.
    window = {"window_size": None} if "sigma_truncate" in scalar else {}
    common = {
        "dynamic_range": 8.0,
        "weights": "gaussian",
        "covariance_normalization": "population",
        "window_size": 11,
        "sigma_truncate": None,
    } | window

    broadcast = ssim_local(x, y, **common | scalar)
    explicit = ssim_local(x, y, **common | per_axis)

    np.testing.assert_array_equal(broadcast, explicit)
