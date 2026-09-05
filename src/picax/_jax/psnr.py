from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Real

from picax._resolve import resolve_dynamic_range
from picax._types import FloatScalar


def _resolve_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    """Normalize ``axis`` into a tuple of non-negative, distinct axis indices."""
    match axis:
        case None:
            return tuple(range(ndim))
        case int():
            candidates = (axis,)
        case tuple():
            candidates = axis
        case _:
            raise ValueError(
                f"axis must be an int, a tuple of ints, or None, got {axis} of type "
                f"{type(axis)}"
            )

    axes = tuple(a + ndim if a < 0 else a for a in candidates)

    if any(not 0 <= a < ndim for a in axes):
        raise ValueError(
            f"axis is out of bounds for an array with {ndim} dimensions, got {axis=}"
        )

    if len(set(axes)) != len(axes):
        raise ValueError(f"axis must not contain duplicate entries, got {axis=}")

    return axes


def psnr(
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    axis: int | tuple[int, ...] | None = None,
) -> Real[Array, "..."]:
    r"""Compute the peak signal-to-noise ratio between two arrays.

    The peak signal-to-noise ratio is the ratio between the largest possible
    signal power and the power of the distortion separating the two inputs,
    expressed in decibels,

    .. math::

        \mathrm{PSNR} = 10 \log_{10}\left(\frac{L^2}{\mathrm{MSE}}\right),

    where :math:`L` is the dynamic range and :math:`\mathrm{MSE}` is the mean
    squared error between the inputs. Higher values indicate greater similarity,
    with ``inf`` for identical inputs.

    Parameters
    ----------
    x, y
        Input arrays. They must have identical shapes.
    dynamic_range
        Difference between the maximum and minimum possible image values. This is
        the peak signal amplitude the error is measured against.
        ``"dtype"`` infers the range from the dtype of ``x``;
        floating-point dtypes are not supported in this mode. ``"range"`` infers
        the range from the extrema across ``x`` and ``y``, over the whole of both
        arrays rather than per reduction slice.
    axis
        Axis or axes the mean squared error is averaged over. A scalar is treated
        as a single axis and negative values count from the last axis.
        ``None`` averages over every axis, yielding a scalar. Remaining axes are
        kept, so batched inputs can be scored individually by reducing over the
        image axes alone.

    Returns
    -------
    psnr
        Peak signal-to-noise ratio in decibels. The output has the shape of ``x``
        with the reduced axes removed.

    Raises
    ------
    ValueError
        If the configuration is invalid or the input shapes do not match.
    """
    # Check shapes
    if not x.shape == y.shape:
        raise ValueError(
            f"x and y must have same shape. Their shapes are: {x.shape=}, {y.shape=}"
        )
    if x.ndim == 0:
        raise ValueError("x and y must have at least one dimension")

    axes = _resolve_axes(axis, x.ndim)

    # Type promotion for precision
    dtype = jnp.result_type(x, y, jnp.float64)

    # Resolve dynamic range
    dynamic_range = resolve_dynamic_range(dynamic_range, x, y, dtype=dtype)

    x = x.astype(dtype)
    y = y.astype(dtype)

    mean_squared_error = jnp.mean((x - y) ** 2, axis=axes)

    # Identical inputs give a zero error, whose division yields the expected
    # infinite ratio rather than an error.
    return 10.0 * jnp.log10(dynamic_range**2 / mean_squared_error)
