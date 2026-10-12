from types import EllipsisType
from typing import Literal

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Real

from picax._jax.ssim import (
    _assert_specified,
    _correlate1d,
    _pool,
    _resolve_config,
    _ssim_components,
)
from picax._types import FloatScalar

# Per-scale exponents, finest scale first, fitted to subjective image quality
# ratings in Wang et al. (2003).
WANG_2003_SCALE_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _downsample(x: Real[Array, "*shape"]) -> Real[Array, "*coarse_shape"]:
    """Halve the resolution of an array along every axis.

    Each axis is low-pass filtered with a two-sample box filter, averaging every
    sample with its successor, and then every other sample is kept. As in Wang et
    al.'s reference implementation, the trailing edge is mirrored, so an axis of
    odd length keeps its last sample unchanged and has length
    ``ceil(length / 2)`` afterwards.
    """
    kernel = jnp.full((2,), 0.5, dtype=x.dtype)

    result = x
    for axis in range(x.ndim):
        result = _correlate1d(result, kernel, axis=axis)
        result = lax.slice_in_dim(result, 0, None, stride=2, axis=axis)

    return result


def ms_ssim(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    preset: Literal["wang-2003"] | None = None,
    window_size: int | tuple[int, ...] | None | EllipsisType = ...,
    weights: Literal["gaussian", "uniform"] | EllipsisType = ...,
    covariance_normalization: Literal["sample", "population"] | EllipsisType = ...,
    sigma: float | tuple[float, ...] | None | EllipsisType = ...,
    sigma_truncate: float | tuple[float, ...] | None | EllipsisType = ...,
    pooling_region: Literal["full", "valid"] | EllipsisType = ...,
    K1: FloatScalar = 0.01,
    K2: FloatScalar = 0.03,
    C3: FloatScalar | None = None,
    scale_weights: tuple[float, ...] | EllipsisType = ...,
) -> Real[Array, ""]:
    r"""Compute the multi-scale structural similarity index between two arrays.

    The inputs are compared at :math:`M` successively halved resolutions,
    the finest being the inputs themselves. Contrast and structure are compared
    at every scale, while luminance is compared at the coarsest scale only,

    .. math::

        \mathrm{MS\text{-}SSIM} = \overline{l_M c_M s_M}^{\,w_M}
        \prod_{j=1}^{M-1} \overline{c_j s_j}^{\,w_j},

    where :math:`l_j`, :math:`c_j` and :math:`s_j` are the local luminance,
    contrast and structure maps of :func:`ssim_local` at scale :math:`j`, the bar
    denotes the mean over the ``pooling_region`` of the map, and :math:`w_j` are
    the scale weights.
    Following Wang et al.'s reference implementation, the coarsest scale reduces
    the full SSIM map rather than multiplying separate means.

    Higher values indicate greater similarity,
    with a maximum of 1.0 for identical inputs. A scale whose mean
    contrast-structure term is negative, which requires the inputs to be
    anticorrelated at that scale, yields ``nan``.

    Parameters
    ----------
    x, y
        Input arrays. They must have identical shapes, and every axis must still
        be at least as long as the window after ``len(scale_weights) - 1``
        halvings.
    dynamic_range
        Difference between the maximum and minimum possible image values. This is
        used to scale ``K1`` and ``K2``, which are defined as fractions of the
        dynamic range. ``"dtype"`` infers the range from the dtype of ``x``;
        floating-point dtypes are not supported in this mode. ``"range"`` infers
        the range from the extrema across ``x`` and ``y`` at full resolution, and
        the same range is used at every scale.
    preset
        Preset configuration matching a common MS-SSIM definition or
        implementation. When a preset is specified, all other configuration
        parameters must be left at their defaults.

        ``"wang-2003"`` uses five scales with the weights
        ``(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)`` and, at every scale, an
        11-sample Gaussian window with a standard deviation of 1.5, population
        covariance normalization and the ``"valid"`` pooling region, matching
        Wang et al.'s reference implementation.
    window_size
        Window size along each spatial axis. A scalar is repeated for every axis.
        For Gaussian weights, ``None`` infers the size from ``sigma`` and
        ``sigma_truncate``. The same window is used at every scale.
    weights
        Window weighting function.
    covariance_normalization
        ``"sample"`` applies the weighted Bessel correction
        ``1 / (1 - sum_i(w_i**2))``, treating the inputs as samples from a larger
        population.

        ``"population"`` applies no degrees-of-freedom correction.
    sigma
        Gaussian standard deviation along each spatial axis. A scalar is repeated
        for every axis. Required for Gaussian weights and required to be ``None``
        for uniform weights.
    sigma_truncate
        Number of standard deviations included on either side of each Gaussian
        kernel when ``window_size`` is inferred.
        A scalar is repeated for every axis.
    pooling_region
        Locations of each scale's local maps that are averaged.

        ``"valid"`` averages only the locations whose window lies entirely inside
        that scale's inputs, as Wang et al.'s reference implementation does.

        ``"full"`` averages every location, including those whose window extends
        into the reflected padding beyond the edges of the inputs.
    K1
        Stabilizing constant for regions where the sum of first moments is close to
        zero.
        The default is 0.01, as in Wang et al. (2003).
    K2
        Stabilizing constant for regions where the sum of second moments is close to
        zero.
        The default is 0.03, as in Wang et al. (2003).
    C3
        Stabilizing constant for regions where cross-correlation is close to zero.
        The default is :math:`C_3 = C_2 / 2`, where
        :math:`C_2 = (K_2 L)^2` and :math:`L` is the dynamic range.
    scale_weights
        Exponent applied to the similarity at each scale, finest scale first. Its
        length sets the number of scales. Every weight must be positive.

    Returns
    -------
    ms_ssim
        Multi-scale structural similarity as a scalar.

    Raises
    ------
    ValueError
        If the configuration is invalid, the input shapes do not match, or the
        inputs are too small for the requested number of scales.
    """
    # Validate preset
    # Ellipsis '...' is used to indicate that a parameter is left as default
    if preset is not None and not all(
        val is ...
        for val in (
            scale_weights,
            window_size,
            weights,
            covariance_normalization,
            sigma,
            sigma_truncate,
            pooling_region,
        )
    ):
        raise ValueError(
            "When a preset is specified, all other parameters must be left as default"
        )

    # Apply preset
    match preset:
        case "wang-2003":
            scale_weights = WANG_2003_SCALE_WEIGHTS
            window_size = 11
            weights = "gaussian"
            covariance_normalization = "population"
            sigma = 1.5
            sigma_truncate = None
            pooling_region = "valid"
        case None:
            pass
        case _:
            raise ValueError(f"Unknown preset: {preset}")

    # Check that all parameters have been set
    scale_weights = _assert_specified("scale_weights", scale_weights)

    # Validate scale weights
    scale_weights = tuple(scale_weights)
    if len(scale_weights) == 0:
        raise ValueError("scale_weights must contain at least one weight")
    if not all(w > 0 for w in scale_weights):
        raise ValueError(f"scale_weights must be positive, got {scale_weights=}")
    n_scales = len(scale_weights)

    config = _resolve_config(
        x,
        y,
        dynamic_range=dynamic_range,
        window_size=window_size,
        weights=weights,
        covariance_normalization=covariance_normalization,
        sigma=sigma,
        sigma_truncate=sigma_truncate,
        K1=K1,
        K2=K2,
        C3=C3,
    )
    pooling_region = _assert_specified("pooling_region", pooling_region)

    # Every halving rounds up, so the coarsest extent is ceil(extent / 2**(M-1)).
    coarsest_shape = x.shape
    for _ in range(n_scales - 1):
        coarsest_shape = tuple(-(-extent // 2) for extent in coarsest_shape)

    for axis, (extent, size) in enumerate(
        zip(coarsest_shape, config.window_sizes, strict=True)
    ):
        if extent < size:
            raise ValueError(
                f"window size {size} exceeds extent {extent} along axis {axis} at "
                f"the coarsest of {n_scales} scales, got input extent "
                f"{x.shape[axis]}"
            )

    x = x.astype(config.dtype)
    y = y.astype(config.dtype)

    result = jnp.asarray(1.0, dtype=config.dtype)
    for scale, scale_weight in enumerate(scale_weights):
        if scale > 0:
            x = _downsample(x)
            y = _downsample(y)

        luminance, contrast, structure = _ssim_components(x, y, config)

        similarity = contrast * structure
        if scale == n_scales - 1:
            similarity = luminance * similarity

        pooled = _pool(
            similarity, window_sizes=config.window_sizes, pooling_region=pooling_region
        )
        result = result * pooled**scale_weight

    return result
