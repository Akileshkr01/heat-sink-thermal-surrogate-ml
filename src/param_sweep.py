# -*- coding: utf-8 -*-
"""
Design of Experiments (DoE) & Parameter Sweep Pipeline
Module: param_sweep.py

Implements space-filling sampling (Latin Hypercube & Sobol sequences)
across multi-dimensional operational bounds and exports clean simulation datasets.
"""

import os
import argparse
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import qmc
from src.analytical_model import PlateFinHeatSink

try:
    from src.analytical_model import PlateFinHeatSink
except ImportError:
    from src.analytical_model import HeatSinkAnalyticalModel as PlateFinHeatSink

def generate_parameter_sweep(
    n_samples: int = 10000,
    output_filepath: str = "data/heat_sink_param_sweep.csv",
    sampling_method: str = "sobol",
    random_seed: int = 42,
    r_jc_val: float = 0.10
) -> pd.DataFrame:
    """
    Performs space-filling sampling across operational parameters and evaluates thermal metrics.

    Parameter Bounds:
        - TDP: [30.0, 250.0] W
        - V_air: [0.5, 15.0] m/s
        - k_tim: [1.0, 12.0] W/m·K
    """
    lower_bounds = [30.0, 0.5, 1.0]
    upper_bounds = [250.0, 15.0, 12.0]

    if sampling_method.lower() == "sobol":
        sampler = qmc.Sobol(d=3, scramble=True, seed=random_seed)
        raw_samples = sampler.random(n=n_samples)
    elif sampling_method.lower() == "lhs":
        sampler = qmc.LatinHypercube(d=3, seed=random_seed)
        raw_samples = sampler.random(n=n_samples)
    else:
        raise ValueError("sampling_method must be either 'sobol' or 'lhs'")

    scaled_samples = qmc.scale(raw_samples, lower_bounds, upper_bounds)
    
    try:
        model = PlateFinHeatSink(R_jc=r_jc_val)
    except TypeError:
        model = PlateFinHeatSink()

    records = []
    for row in scaled_samples:
        tdp_val, v_val, ktim_val = row
        
        try:
            res = model.evaluate(TDP=tdp_val, V_air=v_val, k_tim=ktim_val, include_base_conduction=True)
        except TypeError:
            res = model.evaluate(tdp=tdp_val, air_velocity=v_val, k_tim=ktim_val)
        
        records.append({
            'TDP_W': np.round(tdp_val, 3),
            'V_air_m_s': np.round(v_val, 3),
            'k_tim_W_mK': np.round(ktim_val, 3),
            'Re': np.round(res.get('Re', 0.0), 3),
            'Nu': np.round(res.get('Nu', 0.0), 4),
            'h_conv_W_m2K': np.round(res.get('h_conv_W_m2K', res.get('h_conv', 0.0)), 3),
            'R_tim_C_W': np.round(res.get('R_tim_C_W', res.get('R_tim', 0.0)), 6),
            'R_cond_C_W': np.round(res.get('R_cond_C_W', res.get('R_cond', 0.0)), 6),
            'R_conv_C_W': np.round(res.get('R_conv_C_W', res.get('R_conv', 0.0)), 6),
            'R_hs_C_W': np.round(res.get('R_hs_C_W', res.get('R_hs', 0.0)), 6),
            'R_total_C_W': np.round(res.get('R_total_C_W', res.get('R_total', 0.0)), 6),
            'T_j_C': np.round(res.get('T_j_C', res.get('T_junction_C', res.get('T_j', 0.0))), 4)
        })

    df = pd.DataFrame(records)

    # Ensure parent output directory exists
    parent_dir = os.path.dirname(output_filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    df.to_csv(output_filepath, index=False)
    print(f"Generated {len(df)} design points exported to: {output_filepath}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Heat Sink Parameter Sweep Dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of DoE samples")
    parser.add_argument("--output", type=str, default="data/heat_sink_param_sweep.csv", help="Destination CSV path")
    parser.add_argument("--method", type=str, default="sobol", choices=["sobol", "lhs"], help="Sampling method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility")
    args = parser.parse_args()

    generate_parameter_sweep(
        n_samples=args.samples,
        output_filepath=args.output,
        sampling_method=args.method,
        random_seed=args.seed
    )