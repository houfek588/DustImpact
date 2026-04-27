#!/usr/bin/env python3
"""
plot_four_csv_fixed.py — Load 4 CSV files and plot them into one graph.
All parameters are defined as variables inside main().
"""

import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pick_xy_columns(df: pd.DataFrame,
                    xcol: Optional[str],
                    ycol: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Choose x and y columns from DataFrame.
    Priority:
      1) User-specified xcol/ycol (by name or 0-based index string)
      2) First two numeric columns (coercing if needed)
    """
    def col_from_spec(spec):
        if spec is None:
            return None
        # integer index as string?
        try:
            idx = int(spec)
            return df.columns[idx]
        except (ValueError, IndexError):
            if spec in df.columns:
                return spec
            raise KeyError(f"Column '{spec}' not found. Available: {list(df.columns)}")

    xname = col_from_spec(xcol)
    yname = col_from_spec(ycol)

    if xname is not None and yname is not None:
        return df[xname], df[yname]
    else:
        # Auto-pick first two numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            coerced = df.apply(pd.to_numeric, errors="coerce")
            numeric_cols = [c for c in coerced.columns if pd.api.types.is_numeric_dtype(coerced[c])]
            if len(numeric_cols) < 2:
                raise ValueError("Unable to find two numeric columns to plot.")
            df = coerced
        xser, yser = df[numeric_cols[0]], df[numeric_cols[1]]

        # --- trim trailing (x==0 and y==0) rows only ---
    tol = 1e-12
    xv = np.asarray(xser.values, dtype=float)
    yv = np.asarray(yser.values, dtype=float)

    nonzero = (np.abs(xv) > tol) | (np.abs(yv) > tol)
    if nonzero.any():
        last_idx = np.where(nonzero)[0].max()
        xser = xser.iloc[:last_idx + 1]
        yser = yser.iloc[:last_idx + 1]
    else:
        # all zeros -> return empty series with same dtype/index type
        xser = xser.iloc[:0]
        yser = yser.iloc[:0]

    return xser[1:], yser[1:]


def load_csv(path: str, delimiter: Optional[str], decimal: str) -> pd.DataFrame:
    # pandas will auto-detect delimiter if None
    return pd.read_csv(path, sep=delimiter, decimal=decimal)


def main(path, show):
    # -----------------------------
    # "Parameters from bash call"
    # -----------------------------
    # path = "../MyProj/test6/DefaultProject_orb20ev.spis5/"
    # path = "../MyProj/test6/Orbiter_mat_highT_lowRho.spis5/"
    csv_paths = [
        path + "Node_1_average_surface_potential_V_versus_time_s.nc.csv",
        path + "Node_2_average_surface_potential_V_versus_time_s.nc.csv",
        path + "Node_3_average_surface_potential_V_versus_time_s.nc.csv",
        path + "Spacecraft_average_surface_potential_V_versus_time_s_(the_ones_on_top_of_node_0).nc.csv",
    ]

    max_y = 12
    title  = "Average spacecraft surface potentials"
    xlabel = "Time [s]"
    ylabel = "Voltage [V]"
    output = path + "avg_surface_potentials.png"    # set to None to show instead of saving
    output2 = path + "avg_surface_potentials_diff.png"

    # Optional extras (tweak as needed)
    labels     = ["Antenne 1", "Antenne 2", "Antenne 3", "Spacecraft body"]  # e.g., ["Run A", "Run B", "Run C", "Run D"] or None to use filenames
    delimiter  = ","  # e.g., ';' for semicolon; None = auto-detect
    decimal    = "."   # use ',' for EU decimal comma
    xcol       = None  # name or "0" for index
    ycol       = None  # name or "1" for index
    marker     = ""    # e.g., 'o', '.', '+'
    linestyle  = "-"   # e.g., '-', '--', ':', ''
    alpha      = 1.0

    # -----------------------------
    # Plotting logic
    # -----------------------------
    if labels is None:
        labels = [os.path.basename(p) for p in csv_paths]
    # if len(csv_paths) != 4 or len(labels) != 4:
    #     raise ValueError("Provide exactly 4 CSV paths and 4 labels.")

    plt.figure(figsize=(5, 3))

    data = []
    for path, label in zip(csv_paths, labels):
        df = load_csv(path, delimiter, decimal)
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        x, y = pick_xy_columns(df, xcol, ycol)

        if label == "Spacecraft body":
            const = 13
        elif label == "Antenne 1":
            const = 1.8
        elif label == "Antenne 2":
            const = 2.2
        elif label == "Antenne 3":
            const = 2

        data.append((x.values, y.values))
        plt.plot(
            np.asarray(x.values),
            # np.asarray(y.values),
            np.multiply(np.asarray(y.values), const),
            label=label,
            linestyle=linestyle,
            marker=marker,
            alpha=alpha,
        )


    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.ylim([0, max_y])
    plt.savefig(output, dpi=150)
    # if output:
    #     plt.savefig(output, dpi=150)
    #     print(f"Saved figure to {output}")
    # else:
    #     plt.show()



    # last = data[-1]
    # plt.figure(figsize=(5, 3))
    # for d in range(0, len(data) - 1):
    #
    #     line = data[d]
    #
    #
    #     plt.plot(
    #         np.asarray(line[0]),
    #         np.asarray(np.subtract(line[1], last[1])),
    #         # np.asarray(l[1] - last[0]),
    #         label=labels[d],
    #         linestyle=linestyle,
    #         marker=marker,
    #         alpha=alpha,
    #     )
    # plt.title(title + " difference")
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    # plt.legend()
    # plt.tight_layout()
    # plt.ylim([0, max_y])
    #
    #
    # plt.savefig(output2, dpi=150)
    print(f"Saved figure to {output} and {output2}")

    if show:
        plt.show()

if __name__ == "__main__":

    together = False

    if together:
        paths = ["../MyProj/test6/DefaultProject_orb.spis5/",
             "../MyProj/test6/DefaultProject_orb5ev.spis5/",
             "../MyProj/test6/DefaultProject_orb20ev.spis5/",
             "../MyProj/test6/DefaultProject_orb10ev_photo1ev.spis5/",
             "../MyProj/test6/DefaultProject_orb10ev_photo3ev.spis5/",
             "../MyProj/test6/Orbiter_mat_highT_lowRho.spis5/"
             ]
        for p in paths:
            main(p, False)
    else:
        main("../../MyProj/test6/DefaultProject_orb10ev_photo1ev_thr.spis5/", True)


