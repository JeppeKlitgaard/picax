from types import EllipsisType
from typing import Literal

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
    # Validate constants
    if not alpha > 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not beta > 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not gamma > 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

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

    # Check that all parameters have been set
    if window_size is ...:
        raise ValueError(
            "window_size must be specified, either directly or via a preset"
        )
    if weights is ...:
        raise ValueError("weights must be specified, either directly or via a preset")
    if covariance_normalization is ...:
        raise ValueError(
            "covariance_normalization must be specified, either directly or via "
            "a preset"
        )
    if sigma is ...:
        raise ValueError("sigma must be specified, either directly or via a preset")
    if sigma_truncate is ...:
        raise ValueError(
            "sigma_truncate must be specified, either directly or via a preset"
        )

    # Check shapes
    if not x.shape == y.shape:
        raise ValueError(
            f"x and y must have same shape. Their shapes are: {x.shape=}, {y.shape=}"
        )
    if x.ndim == 0:
        raise ValueError("x and y must have at least one dimension")

    # Resolve window_size
    if isinstance(window_size, int):
        window_size = tuple(window_size for i in range(x.ndim))

    # Resolve sigma
    if isinstance(sigma, (int, float)):
        sigma = tuple(float(sigma) for i in range(x.ndim))

    if isinstance(sigma_truncate, (int, float)):
        sigma_truncate = tuple(float(sigma_truncate) for i in range(x.ndim))

    # Type promotion for precision
    dtype = jnp.result_type(x, y, jnp.float64)
    ndim = x.ndim

    # Resolve dynamic range
    dynamic_range = resolve_dynamic_range(dynamic_range, x, y, dtype=dtype)

    # Resolve constants
    C1 = (K1 * dynamic_range) ** 2
    C2 = (K2 * dynamic_range) ** 2
    if C3 is None:
        C3 = C2 / 2
    else:
        C3 = jnp.asarray(C3, dtype=dtype)

    window_sizes, kernels = _resolve_windows_and_kernels(
        window_size=window_size,
        weights=weights,
        sigma=sigma,
        sigma_truncate=sigma_truncate,
        ndim=ndim,
        dtype=dtype,
    )

    covariance_scale = _resolve_covariance_normalization(
        covariance_normalization=covariance_normalization,
        kernels=kernels,
        dtype=dtype,
    )

    for axis, (extent, size) in enumerate(zip(x.shape, window_sizes, strict=True)):
        if extent < size:
            raise ValueError(
                f"window size {size} exceeds extent {extent} along axis {axis}"
            )

    x = x.astype(dtype)
    y = y.astype(dtype)

    def local_mean(values: Real[Array, "*shape"]) -> Real[Array, "*shape"]:
        return _correlate_separable(values, kernels)

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

    return luminance**alpha * contrast**beta * structure**gamma


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
    K1: FloatScalar = 0.01,
    K2: FloatScalar = 0.03,
    C3: FloatScalar | None = None,
    alpha: FloatScalar = 1,
    beta: FloatScalar = 1,
    gamma: FloatScalar = 1,
) -> Real[Array, "*shape"]:
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
    ssim_map = ssim_local(
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
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return jnp.mean(ssim_map)
