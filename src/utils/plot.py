from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Callable

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
    ax.grid(False)
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
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    extra = {k: v for k, v in {"alpha": alpha, "s": s, "edgecolor": edgecolor}.items() if v is not None}
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
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
    )
    return ax


def subplots(
    groups: list[tuple[list[str], Callable]],
    ncols: int = 2,
    figsize: ArrayLike | None = None,
    **kwargs,
) -> tuple[Figure, ndarray]:
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
