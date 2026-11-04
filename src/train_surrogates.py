# -*- coding: utf-8 -*-
"""
Machine Learning Surrogate Model Training & Benchmarking
Module: train_surrogates.py

Trains and evaluates Gradient Boosted Decision Trees (GBDT) and
Multi-Layer Perceptrons (MLP) for predicting total thermal resistance
and processor junction temperature.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard evaluation metrics: MAE, RMSE, and R² score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}


def train_and_evaluate(
    dataset_path: str = "heat_sink_parameter_sweep.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains dual surrogate architectures across both targets and benchmarks test accuracy.
    """
    valid_path = resolve_dataset_path(dataset_path)
    print(f"Loading dataset from: {valid_path}")
    df = pd.read_csv(valid_path)

    # Automatically handle column naming variations
    v_col = 'Air_Velocity_mps' if 'Air_Velocity_mps' in df.columns else 'V_air_m_s'
    t_col = 'T_junction_C' if 'T_junction_C' in df.columns else 'T_j_C'
    features = ['TDP_W', v_col, 'k_tim_W_mK']
    
    X = df[features]
    y_R = df['R_total_C_W']
    y_T = df[t_col]

    X_train, X_test, y_R_train, y_R_test, y_T_train, y_T_test = train_test_split(
        X, y_R, y_T, test_size=test_size, random_state=random_state
    )

    models = {
        'GBDT_R_total': GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04, random_state=random_state
        ),
        'GBDT_T_junction': GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04, random_state=random_state
        ),
        'MLP_R_total': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu', max_iter=1500,
                random_state=random_state, early_stopping=True
            ))
        ]),
        'MLP_T_junction': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu', max_iter=1500,
                random_state=random_state, early_stopping=True
            ))
        ])
    }

    print("Fitting surrogate models...")
    models['GBDT_R_total'].fit(X_train, y_R_train)
    models['GBDT_T_junction'].fit(X_train, y_T_train)
    models['MLP_R_total'].fit(X_train, y_R_train)
    models['MLP_T_junction'].fit(X_train, y_T_train)

    # Benchmark evaluations on unseen holdout test partition
    pred_gb_R = models['GBDT_R_total'].predict(X_test)
    pred_mlp_R = models['MLP_R_total'].predict(X_test)
    pred_gb_T = models['GBDT_T_junction'].predict(X_test)
    pred_mlp_T = models['MLP_T_junction'].predict(X_test)

    results_table = [
        {'Target': 'R_total (°C/W)', 'Model': 'Gradient Boosting (GBDT)', **compute_metrics(y_R_test, pred_gb_R)},
        {'Target': 'R_total (°C/W)', 'Model': 'Multi-Layer Perceptron (MLP)', **compute_metrics(y_R_test, pred_mlp_R)},
        {'Target': 'T_junction (°C)', 'Model': 'Gradient Boosting (GBDT)', **compute_metrics(y_T_test, pred_gb_T)},
        {'Target': 'T_junction (°C)', 'Model': 'Multi-Layer Perceptron (MLP)', **compute_metrics(y_T_test, pred_mlp_T)},
    ]

    df_results = pd.DataFrame(results_table)
    print("\nBenchmark Evaluation Results on Holdout Test Set:")
    print(df_results.to_string(index=False))

    return models, df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Surrogate Models")
    parser.add_argument("--data", type=str, default="heat_sink_parameter_sweep.csv", help="Path to input sweep CSV")
    args = parser.parse_args()

    train_and_evaluate(dataset_path=args.data)