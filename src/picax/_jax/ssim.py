from types import EllipsisType
from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Real

from picax._resolve import resolve_dynamic_range
from picax._types import FloatScalar


def _assert_window_sizes(window_size: tuple[int, ...], ndim: int) -> None:
    if any(size <= 0 for size in window_size):
        raise ValueError(f"window_size must be positive, got {window_size=}")

    if any(size % 2 == 0 for size in window_size):
        raise ValueError(f"window_size must be odd, got {window_size=}")

    if not len(window_size) == ndim:
        raise ValueError(
            "window_size must contain one size per input axis, got "
            f"{window_size=}, {ndim=}"
        )


def _assert_sigma_truncate(sigma_truncate: tuple[float, ...], ndim: int) -> None:
    if any(st <= 0 for st in sigma_truncate):
        raise ValueError(f"sigma_truncate must be positive, got {sigma_truncate=}")

    if not len(sigma_truncate) == ndim:
        raise ValueError(
            f"sigma_truncate must have same length as "
            f"number of spatial dimensions {ndim}, got {sigma_truncate=}"
        )


def _assert_sigma(sigma: tuple[float, ...], ndim: int) -> None:
    if any(s <= 0 for s in sigma):
        raise ValueError(f"sigma must be positive, got {sigma=}")

    if not len(sigma) == ndim:
        raise ValueError(
            f"sigma must have same length as "
            f"number of spatial dimensions {ndim}, got {sigma=}"
        )


def _assert_specified[T](name: str, value: T | EllipsisType) -> T:
    """Check that a parameter was not left as Ellipsis after applying any preset.

    Returns the value with ``EllipsisType`` removed from its type.
    """
    if value is ...:
        raise ValueError(f"{name} must be specified, either directly or via a preset")
    return value


def _resolve_per_axis_parameters(
    *,
    window_size: int | tuple[int, ...] | None,
    sigma: float | tuple[float, ...] | None,
    sigma_truncate: float | tuple[float, ...] | None,
    ndim: int,
) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None, tuple[float, ...] | None]:
    """Repeat scalar window parameters for every spatial axis.

    Tuples and ``None`` are returned unchanged.
    """
    if isinstance(window_size, int):
        window_size = (window_size,) * ndim

    if isinstance(sigma, (int, float)):
        sigma = (float(sigma),) * ndim

    if isinstance(sigma_truncate, (int, float)):
        sigma_truncate = (float(sigma_truncate),) * ndim

    return window_size, sigma, sigma_truncate


def _resolve_windows_and_kernels(
    *,
    window_size: tuple[int, ...] | None,
    weights: Literal["gaussian", "uniform"],
    sigma: tuple[float, ...] | None,
    sigma_truncate: tuple[float, ...] | None,
    ndim: int,
    dtype: jnp.dtype,
) -> tuple[tuple[int, ...], tuple[Real[Array, "*window_shape"], ...]]:
    """Resolve the window shape and construct normalized, separable kernels.

    Benchmarking shows that XLA cannot exploit the separability of a full N-dimensional
    kernel, so we manually construct a one-dimensional kernel for each spatial axis,
    which can then be applied successively along each axis.

    Parameters
    ----------
    window_size
        Size of the kernel along each axis.
        Gaussian window sizes must be positive and odd.
        If ``None`` with Gaussian weights,
        each size is inferred from ``sigma`` and ``sigma_truncate`` as
        ``2 * int(sigma * sigma_truncate + 0.5) + 1``.
        A window size is required for uniform weights.
    weights
        Window weighting function.

        ``"uniform"`` constructs box kernels.

        ``"gaussian"`` constructs sampled Gaussian kernels.
    sigma
        Gaussian standard deviation along each spatial axis.
        Required for Gaussian weights and required to be ``None`` for uniform weights.
    sigma_truncate
        Number of standard deviations included on either side of each Gaussian
        kernel when ``window_size`` is inferred.
        It must have the same length as ``sigma`` and contain only positive values.
    ndim
        Number of spatial dimensions.
    dtype
        Floating-point dtype of the returned kernels.

    Returns
    -------
    resolved_window_size
        Resolved size of the kernel along each spatial axis.
    kernels
        Normalized one-dimensional kernels in spatial-axis order. Every kernel sums
        to one.

    Raises
    ------
    ValueError
        Bad combination of parameters.
    """
    match weights:
        case "uniform":
            if sigma is not None:
                raise ValueError(
                    "sigma must be None when weights is 'uniform', got "
                    f"sigma={sigma}, weights={weights}"
                )

            if sigma_truncate is not None:
                raise ValueError(
                    "sigma_truncate must be None when weights is 'uniform', got "
                    f"sigma_truncate={sigma_truncate}, weights={weights}"
                )

            if window_size is None:
                raise ValueError(
                    "window_size must be specified when weights is 'uniform', got "
                    f"window_size={window_size}, weights={weights}"
                )

            _assert_window_sizes(window_size, ndim=ndim)

            kernels = tuple(
                jnp.full((size,), 1.0 / size, dtype=dtype) for size in window_size
            )
            return window_size, kernels

        case "gaussian":
            if sigma is None:
                raise ValueError(
                    "sigma must be specified when weights is 'gaussian', got "
                    f"sigma={sigma}, weights={weights}"
                )

            _assert_sigma(sigma, ndim=ndim)

            if window_size is None:
                if sigma_truncate is None:
                    raise ValueError(
                        "sigma_truncate must be specified when window_size is None and "
                        f"weights is 'gaussian', got sigma_truncate={sigma_truncate}, "
                        f"window_size={window_size}, weights={weights}"
                    )

                _assert_sigma_truncate(sigma_truncate, ndim=ndim)

                window_size = tuple(
                    2 * int(st * s + 0.5) + 1
                    for s, st in zip(sigma, sigma_truncate, strict=True)
                )
            elif sigma_truncate is not None:
                raise ValueError(
                    "sigma_truncate must be None when window_size is specified, "
                    f"got sigma_truncate={sigma_truncate}, window_size={window_size}"
                )

            _assert_window_sizes(window_size, ndim=ndim)

            kernels = []
            for size, axis_sigma in zip(window_size, sigma, strict=True):
                coordinates = jnp.arange(size, dtype=dtype) - size // 2
                kernel = jnp.exp(-0.5 * (coordinates / axis_sigma) ** 2)
                kernels.append(kernel / kernel.sum())

            return window_size, tuple(kernels)

        case _:
            raise ValueError(
                f'Unknown weights: {weights}, must be one of {{"gaussian", "uniform"}}'
            )


def _resolve_covariance_normalization(
    covariance_normalization: Literal["sample", "population"],
    kernels: tuple[Real[Array, "*window_shape"], ...],
    dtype: jnp.dtype,
) -> Real[Array, ""]:
    sum_squared_weights = jnp.asarray(1.0, dtype=dtype)
    match covariance_normalization:
        case "population":
            return sum_squared_weights
        case "sample":
            for kernel in kernels:
                sum_squared_weights = sum_squared_weights * jnp.sum(kernel**2)
            covariance_scale = 1.0 / (1.0 - sum_squared_weights)
            return covariance_scale
        case _:
            raise ValueError(
                "covariance_normalization must be 'sample' or 'population', got "
                f"covariance_normalization={covariance_normalization}"
            )


def _resolve_stabilizing_constants(
    *,
    dynamic_range: Real[Array, ""],
    K1: FloatScalar,
    K2: FloatScalar,
    C3: FloatScalar | None,
    dtype: jnp.dtype,
) -> tuple[Real[Array, ""], Real[Array, ""], Real[Array, ""]]:
    C1 = (K1 * dynamic_range) ** 2
    C2 = (K2 * dynamic_range) ** 2
    if C3 is None:
        C3 = C2 / 2
    else:
        C3 = jnp.asarray(C3, dtype=dtype)

    return C1, C2, C3


class _SSIMConfig(NamedTuple):
    """Resolved configuration of the local SSIM comparisons."""

    dtype: jnp.dtype
    window_sizes: tuple[int, ...]
    kernels: tuple[Real[Array, " window"], ...]
    covariance_scale: Real[Array, ""]
    C1: Real[Array, ""]
    C2: Real[Array, ""]
    C3: Real[Array, ""]


def _resolve_config(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    window_size: int | tuple[int, ...] | None | EllipsisType,
    weights: Literal["gaussian", "uniform"] | EllipsisType,
    covariance_normalization: Literal["sample", "population"] | EllipsisType,
    sigma: float | tuple[float, ...] | None | EllipsisType,
    sigma_truncate: float | tuple[float, ...] | None | EllipsisType,
    K1: FloatScalar,
    K2: FloatScalar,
    C3: FloatScalar | None,
) -> _SSIMConfig:
    """Validate the inputs and resolve the configuration of the local comparisons.

    Any preset must already have been applied. Window extents are not checked
    against the inputs, since the extent that matters depends on the caller.
    """
    # Check that all parameters have been set
    window_size = _assert_specified("window_size", window_size)
    weights = _assert_specified("weights", weights)
    covariance_normalization = _assert_specified(
        "covariance_normalization", covariance_normalization
    )
    sigma = _assert_specified("sigma", sigma)
    sigma_truncate = _assert_specified("sigma_truncate", sigma_truncate)

    # Check shapes
    if not x.shape == y.shape:
        raise ValueError(
            f"x and y must have same shape. Their shapes are: {x.shape=}, {y.shape=}"
        )
    if x.ndim == 0:
        raise ValueError("x and y must have at least one dimension")

    # Repeat scalar window parameters for every axis
    window_size, sigma, sigma_truncate = _resolve_per_axis_parameters(
        window_size=window_size,
        sigma=sigma,
        sigma_truncate=sigma_truncate,
        ndim=x.ndim,
    )

    # Type promotion for precision
    dtype = jnp.result_type(x, y, jnp.float64)

    # Resolve dynamic range
    dynamic_range = resolve_dynamic_range(dynamic_range, x, y, dtype=dtype)

    # Resolve constants
    C1, C2, C3 = _resolve_stabilizing_constants(
        dynamic_range=dynamic_range, K1=K1, K2=K2, C3=C3, dtype=dtype
    )

    window_sizes, kernels = _resolve_windows_and_kernels(
        window_size=window_size,
        weights=weights,
        sigma=sigma,
        sigma_truncate=sigma_truncate,
        ndim=x.ndim,
        dtype=dtype,
    )

    covariance_scale = _resolve_covariance_normalization(
        covariance_normalization=covariance_normalization,
        kernels=kernels,
        dtype=dtype,
    )

    return _SSIMConfig(
        dtype=dtype,
        window_sizes=window_sizes,
        kernels=kernels,
        covariance_scale=covariance_scale,
        C1=C1,
        C2=C2,
        C3=C3,
    )


def _correlate1d(
    x: Real[Array, "*shape"],
    kernel: Real[Array, " window"],
    *,
    axis: int,
) -> Real[Array, "*shape"]:
    """Correlate an array along one axis using SciPy's reflect boundary mode."""
    (size,) = kernel.shape
    pad_left = (size - 1) // 2
    pad_right = size // 2

    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad_left, pad_right)
    padded = jnp.pad(x, pad_width, mode="symmetric")

    axis_length = x.shape[axis]
    result = kernel[0] * lax.slice_in_dim(padded, 0, axis_length, axis=axis)
    for offset in range(1, size):
        result = result + kernel[offset] * lax.slice_in_dim(
            padded, offset, offset + axis_length, axis=axis
        )

    return result


def _correlate_separable(
    x: Real[Array, "*shape"],
    kernels: tuple[Real[Array, " window"], ...],
) -> Real[Array, "*shape"]:
    """Apply one correlation kernel along each corresponding array axis."""
    result = x
    for axis, kernel in enumerate(kernels):
        result = _correlate1d(result, kernel, axis=axis)

    return result


def _ssim_components(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    config: _SSIMConfig,
) -> tuple[Real[Array, "*shape"], Real[Array, "*shape"], Real[Array, "*shape"]]:
    """Compute the local luminance, contrast and structure comparison maps.

    Parameters
    ----------
    x, y
        Input arrays of identical shape, already promoted to ``config.dtype``.
    config
        Resolved window kernels, covariance normalization and stabilizing
        constants.

    Returns
    -------
    luminance, contrast, structure
        Local comparison maps with the same shape as ``x`` and ``y``.
    """
    covariance_scale = config.covariance_scale
    C1, C2, C3 = config.C1, config.C2, config.C3

    def local_mean(values: Real[Array, "*shape"]) -> Real[Array, "*shape"]:
        return _correlate_separable(values, config.kernels)

    mean_x = local_mean(x)
    mean_y = local_mean(y)
    mean_xx = local_mean(x * x)
    mean_yy = local_mean(y * y)
    mean_xy = local_mean(x * y)

    variance_x = covariance_scale * (mean_xx - mean_x**2)
    variance_y = covariance_scale * (mean_yy - mean_y**2)
    covariance_xy = covariance_scale * (mean_xy - mean_x * mean_y)

    # Roundoff can make a theoretically nonnegative variance slightly negative.
    stddev_product = jnp.sqrt(jnp.maximum(variance_x, 0) * jnp.maximum(variance_y, 0))

    luminance = (2 * mean_x * mean_y + C1) / (mean_x**2 + mean_y**2 + C1)
    contrast = (2 * stddev_product + C2) / (variance_x + variance_y + C2)
    structure = (covariance_xy + C3) / (stddev_product + C3)

    return luminance, contrast, structure


def _pool(
    values: Real[Array, "*shape"],
    *,
    window_sizes: tuple[int, ...],
    pooling_region: Literal["full", "valid"],
) -> Real[Array, ""]:
    """Average a local map over the requested region.

    ``"valid"`` keeps only the locations whose window lies entirely inside the
    inputs, which is the output region of a ``"valid"`` correlation. Window sizes
    are odd, so ``(size - 1) // 2`` locations are dropped from either end of each
    axis.
    """
    match pooling_region:
        case "full":
            return jnp.mean(values)
        case "valid":
            for axis, size in enumerate(window_sizes):
                border = (size - 1) // 2
                values = lax.slice_in_dim(
                    values, border, values.shape[axis] - border, axis=axis
                )
            return jnp.mean(values)
        case _:
            raise ValueError(
                "pooling_region must be 'full' or 'valid', got "
                f"pooling_region={pooling_region}"
            )


def _resolve_ssim_config(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    preset: Literal["wang-2004-paper", "wang-2004-implementation", "skimage-default"]
    | None,
    window_size: int | tuple[int, ...] | None | EllipsisType,
    weights: Literal["gaussian", "uniform"] | EllipsisType,
    covariance_normalization: Literal["sample", "population"] | EllipsisType,
    sigma: float | tuple[float, ...] | None | EllipsisType,
    sigma_truncate: float | tuple[float, ...] | None | EllipsisType,
    K1: FloatScalar,
    K2: FloatScalar,
    C3: FloatScalar | None,
) -> _SSIMConfig:
    """Apply an SSIM preset and resolve the configuration of the local comparisons.

    See :func:`ssim_local` for the parameters.
    """
    # Validate preset
    # Ellipsis '...' is used to indicate that a parameter is left as default
    if preset is not None and not all(
        val is ...
        for val in (
            window_size,
            weights,
            covariance_normalization,
            sigma,
            sigma_truncate,
        )
    ):
        raise ValueError(
            "When a preset is specified, all other parameters must be left as default"
        )

    # Apply preset
    match preset:
        case "wang-2004-implementation":
            window_size = 11
            weights = "gaussian"
            covariance_normalization = "population"
            sigma = 1.5
            sigma_truncate = None
        case "wang-2004-paper":
            window_size = 11
            weights = "gaussian"
            covariance_normalization = "sample"
            sigma = 1.5
            sigma_truncate = None
        case "skimage-default":
            window_size = 7
            weights = "uniform"
            covariance_normalization = "sample"
            sigma = None
            sigma_truncate = None
        case None:
            pass
        case _:
            raise ValueError(f"Unknown preset: {preset}")

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

    for axis, (extent, size) in enumerate(
        zip(x.shape, config.window_sizes, strict=True)
    ):
        if extent < size:
            raise ValueError(
                f"window size {size} exceeds extent {extent} along axis {axis}"
            )

    return config


def _ssim_map(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    config: _SSIMConfig,
    alpha: FloatScalar,
    beta: FloatScalar,
    gamma: FloatScalar,
) -> Real[Array, "*shape"]:
    """Combine the local comparison maps into the local SSIM map."""
    # Validate constants
    if not alpha > 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not beta > 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not gamma > 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    luminance, contrast, structure = _ssim_components(
        x.astype(config.dtype), y.astype(config.dtype), config
    )

    return luminance**alpha * contrast**beta * structure**gamma


def ssim_local(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    preset: Literal["wang-2004-paper", "wang-2004-implementation", "skimage-default"]
    | None = None,
    window_size: int | tuple[int, ...] | None | EllipsisType = ...,
    weights: Literal["gaussian", "uniform"] | EllipsisType = ...,
    covariance_normalization: Literal["sample", "population"] | EllipsisType = ...,
    sigma: float | tuple[float, ...] | None | EllipsisType = ...,
    sigma_truncate: float | tuple[float, ...] | None | EllipsisType = ...,
    K1: FloatScalar = 0.01,
    K2: FloatScalar = 0.03,
    C3: FloatScalar | None = None,
    alpha: FloatScalar = 1,
    beta: FloatScalar = 1,
    gamma: FloatScalar = 1,
) -> Real[Array, "*shape"]:
    """Compute the local structural similarity index between two arrays.

    Higher values indicate greater similarity,
    with a maximum of 1.0 for identical inputs.

    Parameters
    ----------
    x, y
        Input arrays. They must have identical shapes.
    dynamic_range
        Difference between the maximum and minimum possible image values. This is
        used to scale ``K1`` and ``K2``, which are defined as fractions of the
        dynamic range. ``"dtype"`` infers the range from the dtype of ``x``;
        floating-point dtypes are not supported in this mode. ``"range"`` infers
        the range from the extrema across ``x`` and ``y``.
    preset
        Preset configuration matching a common SSIM definition or implementation.
        When a preset is specified, all other configuration parameters must be left
        at their defaults.

        ``"wang-2004-implementation"`` uses an 11-sample Gaussian window with a
        standard deviation of 1.5 and population covariance normalization,
        matching Wang et al.'s reference implementation.

        ``"wang-2004-paper"`` uses the same Gaussian window with sample covariance
        normalization, matching the estimator described in the paper.

        ``"skimage-default"`` matches the default configuration used by
        :func:`skimage.metrics.structural_similarity`.
    window_size
        Window size along each spatial axis. A scalar is repeated for every axis.
        For Gaussian weights, ``None`` infers the size from ``sigma`` and
        ``sigma_truncate``.
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
    K1
        Stabilizing constant for regions where the sum of first moments is close to
        zero.
        The default is 0.01, as in Wang et al. (2004).
    K2
        Stabilizing constant for regions where the sum of second moments is close to
        zero.
        The default is 0.03, as in Wang et al. (2004).
    C3
        Stabilizing constant for regions where cross-correlation is close to zero.
        The default is :math:`C_3 = C_2 / 2`, where
        :math:`C_2 = (K_2 L)^2` and :math:`L` is the dynamic range.
    alpha
        Exponent controlling the influence of luminance.
        The default is 1, as in Wang et al. (2004).
    beta
        Exponent controlling the influence of contrast.
        The default is 1, as in Wang et al. (2004).
    gamma
        Exponent controlling the influence of structure.
        The default is 1, as in Wang et al. (2004).

    Returns
    -------
    ssim_map
        Local SSIM value at every input location. The output has the same shape as
        ``x`` and ``y``.

    Raises
    ------
    ValueError
        If the configuration is invalid or the input shapes do not match.
    """
    config = _resolve_ssim_config(
        x,
        y,
        dynamic_range=dynamic_range,
        preset=preset,
        window_size=window_size,
        weights=weights,
        covariance_normalization=covariance_normalization,
        sigma=sigma,
        sigma_truncate=sigma_truncate,
        K1=K1,
        K2=K2,
        C3=C3,
    )
    return _ssim_map(x, y, config=config, alpha=alpha, beta=beta, gamma=gamma)


def ssim_mean(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    preset: Literal["wang-2004-paper", "wang-2004-implementation", "skimage-default"]
    | None = None,
    window_size: int | tuple[int, ...] | None | EllipsisType = ...,
    weights: Literal["gaussian", "uniform"] | EllipsisType = ...,
    covariance_normalization: Literal["sample", "population"] | EllipsisType = ...,
    sigma: float | tuple[float, ...] | None | EllipsisType = ...,
    sigma_truncate: float | tuple[float, ...] | None | EllipsisType = ...,
    pooling_region: Literal["full", "valid"] | EllipsisType = ...,
    K1: FloatScalar = 0.01,
    K2: FloatScalar = 0.03,
    C3: FloatScalar | None = None,
    alpha: FloatScalar = 1,
    beta: FloatScalar = 1,
    gamma: FloatScalar = 1,
) -> Real[Array, ""]:
    """Compute the mean structural similarity index between two arrays.

    Higher values indicate greater similarity,
    with a maximum of 1.0 for identical inputs.

    Parameters
    ----------
    x, y
        Input arrays. They must have identical shapes.
    dynamic_range
        Difference between the maximum and minimum possible image values. This is
        used to scale ``K1`` and ``K2``, which are defined as fractions of the
        dynamic range. ``"dtype"`` infers the range from the dtype of ``x``;
        floating-point dtypes are not supported in this mode. ``"range"`` infers
        the range from the extrema across ``x`` and ``y``.
    preset
        Preset configuration matching a common SSIM definition or implementation.
        When a preset is specified, all other configuration parameters must be left
        at their defaults.

        ``"wang-2004-implementation"`` uses an 11-sample Gaussian window with a
        standard deviation of 1.5 and population covariance normalization,
        matching Wang et al.'s reference implementation.

        ``"wang-2004-paper"`` uses the same Gaussian window with sample covariance
        normalization, matching the estimator described in the paper.

        ``"skimage-default"`` matches the default configuration used by
        :func:`skimage.metrics.structural_similarity`.

        Every preset averages over the ``"valid"`` region.
    window_size
        Window size along each spatial axis. A scalar is repeated for every axis.
        For Gaussian weights, ``None`` infers the size from ``sigma`` and
        ``sigma_truncate``.
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
        Locations of the local SSIM map that are averaged.

        ``"valid"`` averages only the locations whose window lies entirely inside
        the inputs, as Wang et al.'s reference implementation and
        :func:`skimage.metrics.structural_similarity` do.

        ``"full"`` averages every location, including those whose window extends
        into the reflected padding beyond the edges of the inputs.
    K1
        Stabilizing constant for regions where the sum of first moments is close to
        zero.
        The default is 0.01, as in Wang et al. (2004).
    K2
        Stabilizing constant for regions where the sum of second moments is close to
        zero.
        The default is 0.03, as in Wang et al. (2004).
    C3
        Stabilizing constant for regions where cross-correlation is close to zero.
        The default is :math:`C_3 = C_2 / 2`, where
        :math:`C_2 = (K_2 L)^2` and :math:`L` is the dynamic range.
    alpha
        Exponent controlling the influence of luminance.
        The default is 1, as in Wang et al. (2004).
    beta
        Exponent controlling the influence of contrast.
        The default is 1, as in Wang et al. (2004).
    gamma
        Exponent controlling the influence of structure.
        The default is 1, as in Wang et al. (2004).

    Returns
    -------
    ssim_mean
        Mean structural similarity as a scalar.

    Raises
    ------
    ValueError
        If the configuration is invalid or the input shapes do not match.
    """
    # Validate preset
    # Ellipsis '...' is used to indicate that a parameter is left as default
    if preset is not None and pooling_region is not ...:
        raise ValueError(
            "When a preset is specified, all other parameters must be left as default"
        )

    # Apply preset. Unknown presets are rejected by _resolve_ssim_config.
    match preset:
        case "wang-2004-implementation" | "wang-2004-paper" | "skimage-default":
            pooling_region = "valid"
        case None:
            pass
        case _:
            raise ValueError(f"Unknown preset: {preset}")

    pooling_region = _assert_specified("pooling_region", pooling_region)

    config = _resolve_ssim_config(
        x,
        y,
        dynamic_range=dynamic_range,
        preset=preset,
        window_size=window_size,
        weights=weights,
        covariance_normalization=covariance_normalization,
        sigma=sigma,
        sigma_truncate=sigma_truncate,
        K1=K1,
        K2=K2,
        C3=C3,
    )
    ssim_map = _ssim_map(x, y, config=config, alpha=alpha, beta=beta, gamma=gamma)
    return _pool(
        ssim_map, window_sizes=config.window_sizes, pooling_region=pooling_region
    )
