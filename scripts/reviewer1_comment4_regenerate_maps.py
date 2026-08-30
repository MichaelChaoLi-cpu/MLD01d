#!/usr/bin/env python3
"""Regenerate the five Nepal maps for Reviewer 1 Comment 4.

The administrative boundary is read from the updated Nepal province layer.
The ecological-belt layer is the study-generated Mountain/Hill/Terai layer
derived from JAXA global DSM elevation data.  This script changes geometry
only; it preserves the existing aggregation definitions, colour scales, and
figure layout used by the manuscript figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mld01d-r1c4-matplotlib")

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

import Modelling  # noqa: E402
import SettingForFeatures  # noqa: E402


ADMIN_PATH = PROJECT_ROOT / "data/raw/SpatialMaps/02_PROVINCE/PROVINCE.shp"
ECOBELT_PATH = (
    PROJECT_ROOT
    / "data/raw/SpatialMaps/nepal_ecobelt_data/3_class_shape/Ecobelts_3Class.shp"
)
PREDICTION_PATH = PROJECT_ROOT / "results/health_prediction_of_HeardClimate_Dummy.npy"

OUTPUT_NAMES = {
    "respondents": "fig01_observation_distribution.jpg",
    "health": "fig02_health.jpg",
    "disaster": "fig03_natural_disaster.jpg",
    "knowledge": "fig04_knowledge_perc.jpg",
    "knowledge_difference": "fig06_spatial_effect.jpg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spatial_units() -> gpd.GeoDataFrame:
    ecobelt = gpd.read_file(ECOBELT_PATH).to_crs(epsg=4326)
    province = gpd.read_file(ADMIN_PATH)[["Province", "geometry"]].to_crs(epsg=4326)

    province["Province"] = province["Province"].replace(
        {
            "Madesh": "Madhesh",
            "Sudur Pashchim": "Sudurpaschim",
        }
    )
    ecobelt = ecobelt.rename(columns={"Ecobelt": "EcoBelt", "Eco_Belt": "EcoBelt"})

    units = gpd.overlay(ecobelt[["EcoBelt", "geometry"]], province, how="intersection")
    units = units[["Province", "EcoBelt", "geometry"]].copy()
    units = units.sort_values(["Province", "EcoBelt"]).reset_index(drop=True)

    if units.crs is None or units.crs.to_epsg() != 4326:
        raise RuntimeError(f"Unexpected output CRS: {units.crs}")
    if units.geometry.is_empty.any() or units.geometry.isna().any():
        raise RuntimeError("Spatial intersection contains empty or missing geometry")
    if not units.geometry.is_valid.all():
        raise RuntimeError("Spatial intersection contains invalid geometry")
    if units.duplicated(["Province", "EcoBelt"]).any():
        raise RuntimeError("Spatial intersection contains duplicate province–EcoBelt units")

    return units


def make_wave_values(
    data: pd.DataFrame,
    variable: str | None,
    multiplier: float,
) -> pd.DataFrame:
    columns = ["Prov", "EcoBelt", "Year"]
    if variable is not None:
        columns.append(variable)
    values = data[columns].copy()
    values = values.rename(columns={"Prov": "Province"})

    if variable is None:
        values["value"] = 1
        grouped = values.groupby(["Province", "EcoBelt", "Year"], as_index=False)["value"].sum()
    else:
        grouped = (
            values.groupby(["Province", "EcoBelt", "Year"], as_index=False)[variable]
            .mean()
            .rename(columns={variable: "value"})
        )
        grouped["value"] *= multiplier

    wide = grouped.pivot(index=["Province", "EcoBelt"], columns="Year", values="value")
    wide = wide.rename(columns={2016: "2016", 2022: "2022"}).reset_index()
    return wide


def add_unit_labels(ax: plt.Axes, frame: gpd.GeoDataFrame) -> None:
    for _, row in frame.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        point = geometry.centroid
        ax.text(
            point.x,
            point.y,
            f"{row['Province']}\n{row['EcoBelt']}",
            fontsize=7,
            ha="center",
            va="center",
        ).set_path_effects(
            [
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal(),
            ]
        )


def render_wave_map(
    units: gpd.GeoDataFrame,
    values: pd.DataFrame,
    output_path: Path,
    panel_title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    cmap: LinearSegmentedColormap,
) -> gpd.GeoDataFrame:
    frame = units.merge(values, on=["Province", "EcoBelt"], how="left")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(18, 18),
        gridspec_kw={"hspace": 0.1, "wspace": 0.05},
    )

    for ax, year, panel in zip(axes, ["2016", "2022"], ["a", "b"]):
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
            missing_kwds={"color": "grey", "edgecolor": "black"},
        )
        add_unit_labels(ax, frame)
        ax.set_title(f"({panel}): {panel_title} in {year}", fontsize=11, loc="left")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axis("on")

    colorbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015])
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(colorbar_label, fontsize=12)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return frame


def make_knowledge_difference(
    data: pd.DataFrame,
    units: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    inputs = SettingForFeatures.return_input_variables()
    outcome = SettingForFeatures.return_output_variables()[0]
    features, _ = Modelling.prepare_data(data, inputs, outcome)
    predictions = np.load(PREDICTION_PATH)

    if predictions.shape != (len(features), 10, 2):
        raise RuntimeError(
            f"Unexpected prediction shape {predictions.shape}; expected {(len(features), 10, 2)}"
        )

    mean_predictions = predictions.mean(axis=1)
    locations = data.loc[features.index, ["Prov", "EcoBelt"]].rename(columns={"Prov": "Province"})
    values = locations.copy()
    values["negative_health_proba"] = mean_predictions[:, 0]
    values["positive_health_proba"] = mean_predictions[:, 1]
    values["difference"] = values["positive_health_proba"] - values["negative_health_proba"]
    regional = values.groupby(["Province", "EcoBelt"], as_index=False).mean(numeric_only=True)
    frame = units.merge(regional, on=["Province", "EcoBelt"], how="left")
    return frame, regional


def render_knowledge_difference(
    frame: gpd.GeoDataFrame,
    output_path: Path,
    cmap: LinearSegmentedColormap,
) -> None:
    vmin, vmax = -0.07, -0.01
    fig, ax = plt.subplots(figsize=(15, 10))
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
        missing_kwds={"color": "grey", "edgecolor": "black"},
    )
    add_unit_labels(ax, frame)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("on")

    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.02, shrink=0.6)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("Effects", fontsize=12)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Rev/analysis/reviewer-1-comment-4-maps",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = SettingForFeatures.data_load_combine_dataset()
    units = load_spatial_units()
    cmap = LinearSegmentedColormap.from_list(
        "blue_green_yellow_red", ["blue", "green", "yellow", "red"], N=256
    )

    figure_specs = [
        (
            "respondents",
            make_wave_values(data, None, 1.0),
            "Observation Spatial Distribution",
            "Total Respondents",
            0,
            1000,
        ),
        (
            "health",
            make_wave_values(data, SettingForFeatures.return_output_variables()[0], 100.0),
            "Disease Probability Spatial Distribution",
            "Disease Probability (%)",
            0,
            60,
        ),
        (
            "disaster",
            make_wave_values(data, "DisasterExpInd", 1.0),
            "Average Experienced Natural Disasters",
            "Types of Natural Disasters",
            0,
            8,
        ),
        (
            "knowledge",
            make_wave_values(data, "HeardClimate_Dummy", 100.0),
            "Percentage of Population with Climate Knowledge",
            "Percentage of Population with Climate Knowledge (%)",
            0,
            70,
        ),
    ]

    value_tables: list[pd.DataFrame] = []
    for key, values, panel_title, label, vmin, vmax in figure_specs:
        output_path = output_dir / OUTPUT_NAMES[key]
        frame = render_wave_map(
            units,
            values,
            output_path,
            panel_title,
            label,
            vmin,
            vmax,
            cmap,
        )
        exported = frame.drop(columns="geometry").copy()
        exported.insert(0, "figure", key)
        value_tables.append(exported)

    difference_frame, difference_values = make_knowledge_difference(data, units)
    render_knowledge_difference(
        difference_frame,
        output_dir / OUTPUT_NAMES["knowledge_difference"],
        cmap,
    )
    difference_export = difference_frame.drop(columns="geometry").copy()
    difference_export.insert(0, "figure", "knowledge_difference")
    value_tables.append(difference_export)
    pd.concat(value_tables, ignore_index=True).to_csv(output_dir / "map_values.csv", index=False)

    survey_units = set(zip(data["Prov"], data["EcoBelt"]))
    geometry_units = set(zip(units["Province"], units["EcoBelt"]))
    missing_geometry = sorted(survey_units - geometry_units)
    if missing_geometry:
        raise RuntimeError(f"Survey units missing from map geometry: {missing_geometry}")

    outputs = {}
    for filename in OUTPUT_NAMES.values():
        path = output_dir / filename
        with Image.open(path) as image:
            outputs[filename] = {
                "sha256": sha256(path),
                "pixels": list(image.size),
                "dpi": list(image.info.get("dpi", ())),
            }

    manifest = {
        "inputs": {
            str(ADMIN_PATH.relative_to(PROJECT_ROOT)): sha256(ADMIN_PATH),
            str(ECOBELT_PATH.relative_to(PROJECT_ROOT)): sha256(ECOBELT_PATH),
            str(PREDICTION_PATH.relative_to(PROJECT_ROOT)): sha256(PREDICTION_PATH),
        },
        "source_note": {
            "administrative_boundary": "Updated Government of Nepal administrative boundary supplied in the project map-data directory.",
            "ecobelt": "Study-generated Mountain/Hill/Terai layer derived from JAXA global DSM elevation data.",
        },
        "geometry": {
            "crs": str(units.crs),
            "rows": len(units),
            "valid_rows": int(units.geometry.is_valid.sum()),
            "empty_rows": int(units.geometry.is_empty.sum()),
            "bounds": units.total_bounds.tolist(),
            "survey_units": len(survey_units),
            "survey_units_missing_geometry": missing_geometry,
            "geometry_units_without_survey_data": sorted(geometry_units - survey_units),
        },
        "knowledge_difference": {
            "households": len(data),
            "regional_rows": len(difference_values),
            "overall_mean": float(
                np.load(PREDICTION_PATH).mean(axis=1)[:, 1].mean()
                - np.load(PREDICTION_PATH).mean(axis=1)[:, 0].mean()
            ),
        },
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
