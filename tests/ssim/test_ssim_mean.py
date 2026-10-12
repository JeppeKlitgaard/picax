import jax.numpy as jnp
import numpy as np
import pytest
from skimage.metrics import structural_similarity

from picax._jax.ssim import ssim_local, ssim_mean

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

# The explicit parameters each preset is meant to be shorthand for.
PRESET_PARAMETERS = {
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

UNIFORM_7 = PRESET_PARAMETERS["skimage-default"]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(("preset", "skimage_options"), PRESETS)
def test_preset_matches_skimage(make_pair, preset, skimage_options, shape):
    x, y = make_pair(shape)
    dynamic_range = 8.0

    expected = structural_similarity(x, y, data_range=dynamic_range, **skimage_options)
    actual = ssim_mean(
        jnp.asarray(x),
        jnp.asarray(y),
        dynamic_range=dynamic_range,
        preset=preset,
    )

    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("preset", list(PRESET_PARAMETERS))
def test_preset_matches_its_explicit_parameters(make_pair, preset):
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)

    from_preset = ssim_mean(x, y, dynamic_range=8.0, preset=preset)
    from_parameters = ssim_mean(x, y, dynamic_range=8.0, **PRESET_PARAMETERS[preset])

    np.testing.assert_array_equal(from_preset, from_parameters)


@pytest.mark.parametrize("shape", SHAPES)
def test_full_region_averages_the_whole_map(make_pair, shape):
    x, y = make_pair(shape)
    x, y = jnp.asarray(x), jnp.asarray(y)
    parameters = UNIFORM_7 | {"pooling_region": "full"}

    expected = jnp.mean(ssim_local(x, y, dynamic_range=8.0, preset="skimage-default"))
    actual = ssim_mean(x, y, dynamic_range=8.0, **parameters)

    np.testing.assert_allclose(actual, expected, **TOL)


def test_valid_region_drops_half_a_window_from_each_edge(make_pair):
    """Each axis loses ``(size - 1) // 2`` locations at either end."""
    x, y = make_pair((31, 35))
    x, y = jnp.asarray(x), jnp.asarray(y)
    parameters = UNIFORM_7 | {"window_size": (5, 9)}
    local_parameters = {k: v for k, v in parameters.items() if k != "pooling_region"}

    ssim_map = ssim_local(x, y, dynamic_range=8.0, **local_parameters)
    expected = jnp.mean(ssim_map[2:-2, 4:-4])
    actual = ssim_mean(x, y, dynamic_range=8.0, **parameters)

    np.testing.assert_allclose(actual, expected, **TOL)


def test_valid_region_of_a_window_sized_input_is_its_centre(make_pair):
    """An input exactly one window long has a single valid location."""
    x, y = make_pair((7, 7))
    x, y = jnp.asarray(x), jnp.asarray(y)

    ssim_map = ssim_local(x, y, dynamic_range=8.0, preset="skimage-default")
    actual = ssim_mean(x, y, dynamic_range=8.0, preset="skimage-default")

    np.testing.assert_allclose(actual, ssim_map[3, 3], **TOL)


def test_returns_a_scalar(make_pair):
    x, y = make_pair((31, 35))

    actual = ssim_mean(
        jnp.asarray(x), jnp.asarray(y), dynamic_range=8.0, preset="skimage-default"
    )

    assert actual.shape == ()


INVALID_CONFIGS = [
    pytest.param(
        {"preset": "skimage-default", "pooling_region": "full"},
        "all other parameters must be left as default",
        id="preset-with-pooling-region",
    ),
    pytest.param(
        {k: v for k, v in UNIFORM_7.items() if k != "pooling_region"},
        "pooling_region must be specified",
        id="missing-pooling-region",
    ),
    pytest.param(
        UNIFORM_7 | {"pooling_region": "nonexistent"},
        "pooling_region must be 'full' or 'valid'",
        id="unknown-pooling-region",
    ),
    pytest.param({"preset": "nonexistent"}, "Unknown preset", id="unknown-preset"),
]


@pytest.mark.parametrize(("kwargs", "message"), INVALID_CONFIGS)
def test_invalid_configuration_raises(kwargs, message):
    x = jnp.zeros((31, 35))
    with pytest.raises(ValueError, match=message):
        ssim_mean(x, x, dynamic_range=8.0, **kwargs)
