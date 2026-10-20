"""Seaborn/matplotlib plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy import ndarray
    from numpy.typing import ArrayLike
    from pandas import DataFrame, Index, Series


def barplot(
    data: DataFrame | None = None,
    x: str | Series | Index | ndarray | None = None,
    y: str | Series | Index | ndarray | None = None,
    hue: str | None = None,
    ax: Axes | None = None,
    figsize: ArrayLike | None = (12, 5),
    rotation: int | None = None,
) -> Axes:
    """Draw a bar plot, creating a new figure if no axes is given.

    Args:
        data: Dataframe to plot from.
        x: Column name, or array-like of values, for the x-axis.
        y: Column name, or array-like of values, for the y-axis.
        hue: Column name to color bars by.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.
        rotation: Degrees to rotate x-axis tick labels by.

    Returns:
        The axes the plot was drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
    )
    if rotation is not None:
        ax.tick_params(axis="x", rotation=rotation)
    return ax


def countplot(
    data: DataFrame | None = None,
    x: str | Series | Index | ndarray | None = None,
    hue: str | None = None,
    ax: Axes | None = None,
    figsize: ArrayLike | None = (5, 4),
) -> Axes:
    """Draw a count plot with comma-formatted count labels on the y-axis.

    Args:
        data: Dataframe to plot from.
        x: Column name, or array-like of values, for the x-axis.
        hue: Column name to color bars by.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.

    Returns:
        The axes the plot was drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.countplot(
        data=data,
        x=x,
        hue=hue,
        ax=ax,
    )
    ax.yaxis.set_label_text("Count")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return ax


def heatmap(
    data: DataFrame | ndarray,
    ax: Axes | None = None,
    figsize: ArrayLike | None = None,
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".1f",
    linewidths: float = 0.5,
    square: bool = True,
    mask: ndarray | None = None,
) -> Axes:
    """Draw a heatmap, sizing the figure to the data's shape by default.

    Args:
        data: 2D data to plot.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.
            Defaults to a size scaled to `data`'s shape.
        cmap: Colormap name.
        annot: Whether to annotate each cell with its value.
        fmt: Format string for cell annotations and, if it contains a
            comma, the colorbar tick labels.
        linewidths: Width of the lines separating cells.
        square: Whether to force square cells.
        mask: Boolean array of cells to hide.

    Returns:
        The axes the plot was drawn on.

    """
    if figsize is None:
        h = max(data.shape[0] * 0.75, 5)
        w = max(data.shape[1] * 0.75, 5)
        figsize = (w, h)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data=data,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=linewidths,
        square=square,
        mask=mask,
    )
    ax.grid(visible=False)
    cbar = ax.collections[0].colorbar
    if cbar is not None and "," in fmt:
        fmt_fn = ticker.FuncFormatter(lambda x, _: f"{x:{fmt}}")
        cbar.ax.yaxis.set_major_formatter(fmt_fn)
    return ax


def histplot(
    data: DataFrame | None = None,
    x: str | Series | Index | ndarray | None = None,
    hue: str | None = None,
    ax: Axes | None = None,
    figsize: ArrayLike | None = (5, 4),
    discrete: bool = False,
) -> Axes:
    """Draw a histogram, overlaying one per `hue` group if given.

    Args:
        data: Dataframe to plot from.
        x: Column name, or array-like of values, to bin.
        hue: Column name to split into overlaid, labeled histograms by.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.
        discrete: Whether to treat each unique value as its own bin.

    Returns:
        The axes the plot was drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if hue is not None and data is not None and isinstance(x, str):
        for group in data[hue].unique():
            sns.histplot(
                data=data.loc[data[hue] == group],
                x=x,
                label=group,
                discrete=discrete,
                ax=ax,
            )
        ax.legend(title=hue)
    else:
        sns.histplot(
            data=data,
            x=x,
            discrete=discrete,
            ax=ax,
        )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return ax


def kdeplot(
    data: DataFrame | None = None,
    x: str | Series | Index | ndarray | None = None,
    hue: str | None = None,
    ax: Axes | None = None,
    figsize: ArrayLike | None = (5, 4),
    fill: bool = True,
) -> Axes:
    """Draw a kernel density estimate plot.

    Args:
        data: Dataframe to plot from.
        x: Column name, or array-like of values, to estimate density for.
        hue: Column name to color and group densities by.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.
        fill: Whether to fill the area under each density curve.

    Returns:
        The axes the plot was drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(
        data=data,
        x=x,
        hue=hue,
        fill=fill,
        ax=ax,
    )
    return ax


def scatterplot(
    data: DataFrame | None = None,
    x: str | Series | Index | ndarray | None = None,
    y: str | Series | Index | ndarray | None = None,
    hue: str | None = None,
    ax: Axes | None = None,
    figsize: ArrayLike | None = (5, 4),
    alpha: float | None = None,
    s: float | None = None,
    edgecolor: str | None = None,
) -> Axes:
    """Draw a scatter plot.

    Args:
        data: Dataframe to plot from.
        x: Column name, or array-like of values, for the x-axis.
        y: Column name, or array-like of values, for the y-axis.
        hue: Column name to color points by.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.
        alpha: Point opacity.
        s: Point size.
        edgecolor: Point edge color.

    Returns:
        The axes the plot was drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    extra = {
        k: v
        for k, v in {"alpha": alpha, "s": s, "edgecolor": edgecolor}.items()
        if v is not None
    }
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        **extra,
    )
    return ax


def lineplot(
    data: DataFrame | None = None,
    x: str | Series | Index | ndarray | None = None,
    y: str | Series | Index | ndarray | None = None,
    hue: str | None = None,
    ax: Axes | None = None,
    figsize: ArrayLike | None = (5, 4),
    rotation: int | None = None,
) -> Axes:
    """Draw a line plot.

    Args:
        data: Dataframe to plot from.
        x: Column name, or array-like of values, for the x-axis.
        y: Column name, or array-like of values, for the y-axis.
        hue: Column name to color lines by.
        ax: Axes to draw on. A new figure is created if omitted.
        figsize: Size of the new figure, used only when `ax` is omitted.
        rotation: Degrees to rotate x-axis tick labels by.

    Returns:
        The axes the plot was drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
    )
    if rotation is not None:
        ax.tick_params(axis="x", rotation=rotation)
    return ax


def subplots(
    groups: Sequence[tuple[list[str], Callable[..., Axes]]],
    ncols: int = 2,
    figsize: ArrayLike | None = None,
    **kwargs: DataFrame | str | None,
) -> tuple[Figure, ndarray]:
    """Lay out one plot per feature across a grid of subplots.

    Each group's features share that group's plotting function, called
    once per feature as `plot_fn(x=feature, ax=ax, **kwargs)`.

    Args:
        groups: Pairs of (feature names, plotting function) to lay out.
        ncols: Number of columns in the grid.
        figsize: Size of the figure. Defaults to a size scaled to the
            number of rows and columns.
        **kwargs: Additional keyword arguments passed to every `plot_fn`
            call (e.g. `data`, `hue`).

    Returns:
        The figure and its grid of axes, with unused axes hidden.

    """
    pairs = [(feature, plot_fn) for features, plot_fn in groups for feature in features]
    nrows = int(np.ceil(len(pairs) / ncols))
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, (feature, plot_fn) in enumerate(pairs):
        ax = axes[i // ncols, i % ncols]
        plot_fn(x=feature, ax=ax, **kwargs)

    for i in range(len(pairs), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    plt.tight_layout()
    return fig, axes
