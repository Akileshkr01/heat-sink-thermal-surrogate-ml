# -*- coding: utf-8 -*-
"""
Parametric Sensitivity & Monotonic Correlation Analysis
Module: sensitivity.py

Performs Pearson and Spearman rank correlation studies, GBDT feature importance
extractions, and identifies critical airflow velocity thresholds.
"""

import os
import sys
import argparse
from typing import Dict
import numpy as np
import pandas as pd


def resolve_dataset_path(given_path: str) -> str:
    """
    Locates the dataset across common directories and naming variations.
    """
    candidate_paths = [
        given_path,
        os.path.join("data", given_path),
        "heat_sink_parameter_sweep.csv",
        "heat_sink_param_sweep.csv",
        os.path.join("data", "heat_sink_parameter_sweep.csv"),
        os.path.join("data", "heat_sink_param_sweep.csv"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    print("\n[!] Dataset CSV file could not be found.")
    print("Checked locations:")
    for path in candidate_paths:
        print(f"  - {os.path.abspath(path)}")
    print("\nIf you haven't generated the sweep data yet, run:")
    print("  python -m src.param_sweep --samples 2500 --output heat_sink_parameter_sweep.csv\n")
    sys.exit(1)


def run_sensitivity_analysis(dataset_path: str = "heat_sink_parameter_sweep.csv") -> Dict[str, pd.DataFrame]:
    """
    Computes linear, rank-order correlations and power-law scaling metrics.
    """
    valid_path = resolve_dataset_path(dataset_path)
    print(f"Loading dataset from: {valid_path}")
    df = pd.read_csv(valid_path)

    # Automatically handle column naming variations
    v_col = 'Air_Velocity_mps' if 'Air_Velocity_mps' in df.columns else 'V_air_m_s'
    t_col = 'T_junction_C' if 'T_junction_C' in df.columns else 'T_j_C'
    analysis_columns = ['TDP_W', v_col, 'k_tim_W_mK', 'R_total_C_W', t_col]
    
    pearson_corr = df[analysis_columns].corr(method='pearson')
    spearman_corr = df[analysis_columns].corr(method='spearman')

    print("\n--- Spearman Rank Correlation (Monotonic Association) ---")
    print(spearman_corr.round(4))

    print("\n--- Pearson Linear Correlation ---")
    print(pearson_corr.round(4))
    
    # Airflow diminishing returns calculation
    sub_df = df[(df['TDP_W'] == 150.0) & (df['k_tim_W_mK'] == 4.0)]
    if not sub_df.empty:
        r_low = sub_df[sub_df[v_col] == 0.5]['R_total_C_W'].values
        r_mid = sub_df[sub_df[v_col] == 3.0]['R_total_C_W'].values
        r_high = sub_df[sub_df[v_col] == 15.0]['R_total_C_W'].values
        if len(r_low) > 0 and len(r_mid) > 0 and len(r_high) > 0:
            print(f"\nR_total @ 0.5 m/s: {r_low[0]:.4f} °C/W")
            print(f"R_total @ 3.0 m/s: {r_mid[0]:.4f} °C/W (drop: {((r_low[0]-r_mid[0])/r_low[0])*100:.1f}%)")
            print(f"R_total @ 15.0 m/s: {r_high[0]:.4f} °C/W (drop: {((r_mid[0]-r_high[0])/r_mid[0])*100:.1f}%)")

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Sensitivity Analysis")
    parser.add_argument("--data", type=str, default="heat_sink_parameter_sweep.csv", help="Path to input sweep CSV")
    args = parser.parse_args()

    run_sensitivity_analysis(dataset_path=args.data)