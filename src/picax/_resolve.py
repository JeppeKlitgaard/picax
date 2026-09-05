from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Real

from picax._types import FloatScalar


def resolve_dynamic_range(
    dynamic_range: FloatScalar | Literal["dtype", "range"],
    x: Real[Array, "*shape"],
    y: Real[Array, "*shape"],
    *,
    dtype: jnp.dtype,
) -> Real[Array, ""]:
    """Resolve the difference between the largest and smallest possible values.

    Parameters
    ----------
    dynamic_range
        Difference between the maximum and minimum possible image values.
        ``"dtype"`` infers the range from the dtype of ``x`` and ``y``;
        floating-point dtypes are not supported in this mode. ``"range"`` infers
        the range from the extrema across ``x`` and ``y``.
    x, y
        Input arrays the range is inferred from.
    dtype
        Floating-point dtype of the returned range.

    Returns
    -------
    resolved_dynamic_range
        Dynamic range as a scalar array.

    Raises
    ------
    ValueError
        Bad combination of parameters.
    """
    match dynamic_range:
        case "dtype":
            if not x.dtype == y.dtype:
                raise ValueError(
                    "When dynamic_range is 'dtype', x and y must have the same "
                    f"dtype. Dtypes are: {x.dtype=}, {y.dtype=}"
                )

            if jnp.issubdtype(x.dtype, jnp.floating):
                raise ValueError(
                    "When dynamic_range is 'dtype', x and y must have an integer "
                    f"dtype. Got x.dtype={x.dtype}, y.dtype={y.dtype}"
                )

            info = jnp.iinfo(x.dtype)
            return jnp.asarray(info.max - info.min, dtype=dtype)

        case "range":
            min_val = jnp.minimum(x.min(), y.min())
            max_val = jnp.maximum(x.max(), y.max())
            return (max_val - min_val).astype(dtype)

        case str():
            raise ValueError(
                "dynamic_range must be a float, int, 'dtype', or 'range'. "
                f"Got {dynamic_range} of type {type(dynamic_range)}"
            )

        case _:
            try:
                resolved = jnp.asarray(dynamic_range, dtype=dtype)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "dynamic_range must be a float, int, 'dtype', or 'range'. "
                    f"Got {dynamic_range} of type {type(dynamic_range)}"
                ) from error

            if resolved.ndim != 0:
                raise ValueError(
                    f"dynamic_range must be a scalar, got shape {resolved.shape}"
                )

            return resolved
