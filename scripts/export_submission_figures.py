#!/usr/bin/env python3
"""Export the eight main-manuscript figures as journal-ready PDF files.

The export reuses the study's existing data, prediction arrays, spatial
boundaries, aggregation rules, and confidence-interval calculations.  It
changes presentation only: every file is 107 mm wide, contains no figure
title or caption, uses lowercase panel labels, and omits the top and right
axis spines so graphs do not have a box outline.  Layout, colors, legends,
labels, and relative proportions follow the previously approved figures as
closely as the journal artwork rules permit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mld01d-submission-figures-matplotlib")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import reviewer1_comment4_regenerate_maps as maps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
MM = 1 / 25.4
FIGURE_WIDTH_MM = 107

CAPTIONS = {
    1: "Spatial Distribution of Respondents in Each Wave",
    2: "Regional Average Probability of Household Disease Incidence in Each Wave",
    3: "Regional Average Type Counts of Natural Disasters in Each Wave",
    4: "Regional Average Percentage of Population with Climate Knowledge in Each Wave",
    5: "Global Relationship between Natural Disaster Count and Disease Increase Probability",
    6: "Global Relationship between Natural Disaster Count and Disease Increase Probability by Climate Change Knowledge Status",
    7: "Spatial Heterogeneity in Climate Change Knowledge Prediction Differences",
    8: "Heterogeneity in Climate Change Knowledge Prediction Differences among Different Groups",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def open_axes(ax: plt.Axes) -> None:
    """Retain ordinary x/y axes without a four-sided box outline."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=2.5)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    parentheses: bool = True,
    bold: bool = False,
    fontsize: float | None = None,
) -> None:
    ax.text(
        0.002 if parentheses else 0.02,
        1.006 if parentheses else 0.95,
        f"({label})" if parentheses else label,
        transform=ax.transAxes,
        ha="left",
        va="bottom" if parentheses else "top",
        fontsize=fontsize if fontsize is not None else (9 if parentheses else 10),
        fontweight="bold" if bold else "normal",
        zorder=10,
        clip_on=False,
    )


def save_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", dpi=300, metadata={"Title": "", "Subject": ""})
    plt.close(fig)


def add_map_labels(ax: plt.Axes, frame) -> None:
    for _, row in frame.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        point = geometry.centroid
        text = ax.text(
            point.x,
            point.y,
            f"{row['Province']}\n{row['EcoBelt']}",
            fontsize=4.0,
            ha="center",
            va="center",
            linespacing=0.9,
        )
        text.set_path_effects(
            [
                maps.path_effects.Stroke(linewidth=1.1, foreground="white"),
                maps.path_effects.Normal(),
            ]
        )


def export_wave_map(
    units,
    values: pd.DataFrame,
    output: Path,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    cmap,
) -> None:
    frame = units.merge(values, on=["Province", "EcoBelt"], how="left")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_MM * MM, 135.3 * MM),
        gridspec_kw={"hspace": 0.16},
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.975, bottom=0.20)

    for ax, year, label in zip(axes, ["2016", "2022"], ["a", "b"]):
        frame.plot(
            column=year,
            cmap=cmap,
            edgecolor="black",
            linewidth=0.4,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            legend=False,
            ax=ax,
            missing_kwds={"color": "0.7", "edgecolor": "black"},
        )
        add_map_labels(ax, frame)
        panel_label(ax, label)
        ax.set_xlabel("Longitude", labelpad=1)
        ax.set_ylabel("Latitude", labelpad=1)
        ax.grid(True, linestyle="--", alpha=0.4)
        open_axes(ax)

    colorbar_ax = fig.add_axes([0.25, 0.075, 0.50, 0.015])
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(colorbar_label, fontsize=9, labelpad=2)
    colorbar.ax.tick_params(labelsize=8, width=0.6, length=2)
    colorbar.outline.set_visible(False)
    save_pdf(fig, output)


def export_pdp(output: Path, with_knowledge: bool) -> None:
    x = np.arange(11)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_MM * MM, 80.4 * MM))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.97, bottom=0.18)

    if with_knowledge:
        arrays = [
            (
                np.load(RESULTS / "pdp_array_without_knowledge.npy"),
                "Mean Prediction Without Knowledge",
                "#1F77B4",
            ),
            (
                np.load(RESULTS / "pdp_array_with_knowledge.npy"),
                "Mean Prediction With Knowledge",
                "red",
            ),
        ]
        line_handles = []
        band_handle = None
        for index, (values, label, color) in enumerate(arrays):
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            line = ax.plot(x, mean, linewidth=2, color=color, label=label)[0]
            band = ax.fill_between(
                x,
                mean - 1.96 * std,
                mean + 1.96 * std,
                color="0.55",
                alpha=0.3,
                label=r"$\pm 1.96\sigma$" if index == 0 else None,
            )
            line_handles.append(line)
            if index == 0:
                band_handle = band
        ax.legend(
            [line_handles[0], line_handles[1], band_handle],
            [arrays[0][1], arrays[1][1], r"$\pm 1.96\sigma$"],
            loc="lower right",
            frameon=True,
            handlelength=2.2,
            fontsize=7,
        )
    else:
        values = np.load(RESULTS / "pdp_array_DisasterExpInd.npy")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(x, mean, linewidth=2, color="#1F77B4", label="Mean Prediction")
        ax.fill_between(
            x,
            mean - 1.96 * std,
            mean + 1.96 * std,
            color="0.55",
            alpha=0.3,
            label=r"$\pm 1.96\sigma$",
        )
        ax.legend(loc="upper left", frameon=True)

    ax.set_xlabel("Natural Disaster Count")
    ax.set_ylabel("Predicted Disease Increase Probability")
    ax.grid(True)
    open_axes(ax)
    save_pdf(fig, output)


def export_difference_map(frame, output: Path, cmap) -> None:
    vmin, vmax = -0.07, -0.01
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_MM * MM, 57.8 * MM))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.98, bottom=0.17)
    frame.plot(
        column="difference",
        cmap=cmap,
        edgecolor="black",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        legend=False,
        ax=ax,
        missing_kwds={"color": "0.7", "edgecolor": "black"},
    )
    add_map_labels(ax, frame)
    ax.set_xlabel("Longitude", labelpad=1)
    ax.set_ylabel("Latitude", labelpad=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    open_axes(ax)

    colorbar_ax = fig.add_axes([0.84, 0.23, 0.030, 0.58])
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="vertical")
    colorbar.set_label("Mean Prediction Difference", fontsize=7, labelpad=4)
    colorbar.ax.tick_params(labelsize=7, width=0.6, length=2)
    save_pdf(fig, output)


def subgroup_summary(features: pd.DataFrame, differences: np.ndarray):
    ratio_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    ratio_labels = [
        "0-10%",
        "10%-20%",
        "20%-30%",
        "30%-40%",
        "40%-50%",
        "50%-60%",
        "60%-100%",
    ]
    specs = [
        ("Literal_Ratio", "Literate Member Ratio", ratio_bins, ratio_labels),
        (
            "Edu12_Ratio",
            "Members with 12-Year Education or above Ratio",
            ratio_bins,
            ratio_labels,
        ),
        ("Female_Ratio", "Female Ratio in Household", ratio_bins, ratio_labels),
        ("A65_Ratio", "Seniors Ratio", ratio_bins, ratio_labels),
        (
            "DisasterExpInd",
            "Natural Disaster Experience Indicator",
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 15.5],
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ">=10"],
        ),
    ]

    summaries = []
    for variable, label, bins, labels in specs:
        values = pd.DataFrame({"x": features[variable].to_numpy(), "y": differences})
        values["group"] = pd.cut(
            values["x"], bins=bins, labels=labels, include_lowest=True
        )
        grouped = values.groupby("group", observed=False)["y"].agg(["mean", "std", "count"])
        grouped = grouped.dropna(subset=["mean"]).reset_index()
        grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
        summaries.append((variable, label, grouped))
    return summaries


def export_subgroups(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    inputs = maps.SettingForFeatures.return_input_variables()
    outcome = maps.SettingForFeatures.return_output_variables()[0]
    features, _ = maps.Modelling.prepare_data(data, inputs, outcome)
    predictions = np.load(maps.PREDICTION_PATH).mean(axis=1)
    differences = predictions[:, 1] - predictions[:, 0]
    summaries = subgroup_summary(features, differences)

    fig, axes_grid = plt.subplots(
        3,
        2,
        figsize=(FIGURE_WIDTH_MM * MM, 125 * MM),
        sharey=True,
    )
    axes = axes_grid.flatten()
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.985,
        bottom=0.11,
        hspace=0.78,
        wspace=0.32,
    )
    fig.supylabel("Mean Prediction Difference", x=0.025, fontsize=7)

    export_rows = []
    for ax, panel, (variable, xlabel, grouped) in zip(
        axes, "abcde", summaries
    ):
        x = np.arange(len(grouped))
        ax.errorbar(
            x,
            grouped["mean"],
            yerr=grouped["ci"],
            fmt="o",
            color="#1F77B4",
            markersize=4,
            linewidth=1,
            capsize=3,
        )
        ax.axhline(0, linestyle="--", color="red", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            grouped["group"].astype(str),
            rotation=45,
            ha="right",
            fontsize=5.5,
        )
        if variable == "Edu12_Ratio":
            xlabel = "Members with 12-Year Education\nor above Ratio"
        ax.set_xlabel(xlabel, labelpad=1, fontsize=6.3)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True)
        panel_label(ax, panel, parentheses=False, bold=True, fontsize=8)
        open_axes(ax)

        table = grouped.copy()
        table.insert(0, "variable", variable)
        export_rows.append(table)

    axes[5].axis("off")

    save_pdf(fig, output)
    return pd.concat(export_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Rev/revision/figures",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    data = maps.SettingForFeatures.data_load_combine_dataset()
    units = maps.load_spatial_units()
    cmap = LinearSegmentedColormap.from_list(
        "blue_green_yellow_red", ["blue", "green", "yellow", "red"], N=256
    )

    wave_specs = [
        (1, maps.make_wave_values(data, None, 1.0), "Total Respondents", 0, 1000),
        (
            2,
            maps.make_wave_values(
                data, maps.SettingForFeatures.return_output_variables()[0], 100.0
            ),
            "Disease Probability (%)",
            0,
            60,
        ),
        (
            3,
            maps.make_wave_values(data, "DisasterExpInd", 1.0),
            "Types of Natural Disasters",
            0,
            8,
        ),
        (
            4,
            maps.make_wave_values(data, "HeardClimate_Dummy", 100.0),
            "Percentage of Population with Climate Knowledge (%)",
            0,
            70,
        ),
    ]
    for number, values, label, vmin, vmax in wave_specs:
        export_wave_map(
            units,
            values,
            output_dir / f"Figure{number}.pdf",
            label,
            vmin,
            vmax,
            cmap,
        )

    export_pdp(output_dir / "Figure5.pdf", with_knowledge=False)
    export_pdp(output_dir / "Figure6.pdf", with_knowledge=True)

    difference_frame, _ = maps.make_knowledge_difference(data, units)
    export_difference_map(difference_frame, output_dir / "Figure7.pdf", cmap)
    subgroup_table = export_subgroups(data, output_dir / "Figure8.pdf")
    subgroup_table.to_csv(output_dir / "Figure8_values.csv", index=False)

    outputs = {}
    for number in range(1, 9):
        path = output_dir / f"Figure{number}.pdf"
        outputs[path.name] = {
            "caption": CAPTIONS[number],
            "sha256": sha256(path),
            "width_mm": FIGURE_WIDTH_MM,
            "format": "PDF",
        }

    manifest = {
        "policy": {
            "one_figure_per_file": True,
            "width_mm": FIGURE_WIDTH_MM,
            "panel_labels": "lowercase letters; parentheses retained for map panels",
            "figure_titles_or_captions_embedded": False,
            "four_sided_graph_box": False,
            "all_composite_panels_on_one_page": True,
            "data_or_model_recomputed": False,
        },
        "sources": {
            "administrative_boundary": str(maps.ADMIN_PATH.relative_to(PROJECT_ROOT)),
            "ecobelt": str(maps.ECOBELT_PATH.relative_to(PROJECT_ROOT)),
            "knowledge_prediction_array": str(maps.PREDICTION_PATH.relative_to(PROJECT_ROOT)),
            "pdp_array": "results/pdp_array_DisasterExpInd.npy",
            "pdp_without_knowledge": "results/pdp_array_without_knowledge.npy",
            "pdp_with_knowledge": "results/pdp_array_with_knowledge.npy",
        },
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
