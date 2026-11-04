# -*- coding: utf-8 -*-
"""
Physics-Based Analytical Heat Sink Calculation Engine
Module: analytical_model.py

Formulates the 1D/quasi-2D thermal resistance network for forced-convection
plate-fin heat sinks, incorporating base conduction resistance, verified wetted
area geometry, and transition-flow Nusselt number interpolation.
"""

import argparse
from typing import Dict, Any
import numpy as np


class PlateFinHeatSink:
    """
    Analytical thermal resistance network model for forced convection plate-fin heat sinks.
    
    Standard references:
    - Incropera & DeWitt, Fundamentals of Heat and Mass Transfer
    - Kraus & Bar-Cohen, Design and Analysis of Heat Sinks
    """
    def __init__(
        self,
        L_die: float = 0.0525,         # Die length (m)
        W_die: float = 0.0450,         # Die width (m)
        Thickness_die: float = 0.0022, # Die thickness (m)
        L_sink: float = 0.090,         # Heat sink length (m)
        W_sink: float = 0.116,         # Heat sink width (m)
        H_sink: float = 0.027,         # Total heat sink height (m)
        Fin_Thickness: float = 0.8e-3, # Fin thickness (m)
        N_fins: int = 60,              # Number of fins
        Base_Thickness: float = 2.5e-3,# Base plate thickness (m)
        k_Al: float = 167.0,           # Aluminum thermal conductivity (W/m·K)
        t_tim: float = 0.1e-3,         # TIM thickness (m)
        R_jc: float = 0.10,            # Junction-to-case resistance (°C/W)
        T_ambient: float = 25.0        # Ambient coolant temperature (°C)
    ) -> None:
        self.L_die = L_die
        self.W_die = W_die
        self.Thickness_die = Thickness_die
        self.A_die = L_die * W_die

        self.L = L_sink
        self.W = W_sink
        self.H = H_sink
        self.t_fin = Fin_Thickness
        self.N_fins = N_fins
        self.t_base = Base_Thickness
        self.h_fin = H_sink - Base_Thickness

        self.k_Al = k_Al
        self.t_tim = t_tim
        self.R_jc = R_jc
        self.T_ambient = T_ambient

        # Thermophysical properties of dry air at 25 °C (1 atm)
        self.k_air = 0.0262    # W/m·K
        self.nu_air = 1.57e-5  # m²/s
        self.Pr_air = 0.71

        # Inter-fin spacing and hydraulic dimensions
        self.s_f = (self.W - (self.N_fins * self.t_fin)) / (self.N_fins - 1)
        self.D_h = (2.0 * self.s_f * self.h_fin) / (self.s_f + self.h_fin)

        # Total exposed wetted area: fin flanks + exposed base channels + fin tips
        self.A_fin_flanks = self.N_fins * (2.0 * self.h_fin * self.L)
        self.A_base_exposed = (self.N_fins - 1) * self.s_f * self.L
        self.A_fin_tips = self.N_fins * self.t_fin * self.L
        self.A_total = self.A_fin_flanks + self.A_base_exposed + self.A_fin_tips

        # Fixed normal conduction resistance through the base under the die footprint
        self.R_cond_base = self.t_base / (self.k_Al * self.A_die)

    def evaluate(
        self,
        TDP: float,
        V_air: float,
        k_tim: float,
        include_base_conduction: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate thermal performance for given operational boundary conditions.
        
        Args:
            TDP: Thermal Design Power (W)
            V_air: Approach air velocity (m/s)
            k_tim: Thermal interface material thermal conductivity (W/m·K)
            include_base_conduction: Toggle inclusion of R_cond through the base
        
        Returns:
            Dictionary with hydrodynamics, resistances, and junction temperature.
        """
        # Fin channel Reynolds number
        Re = (V_air * self.s_f) / self.nu_air

        # Nusselt number calculation
        # Laminar developing duct flow (Sieder-Tate)
        Nu_lam = 1.86 * ((Re * self.Pr_air * (2.0 * self.s_f) / self.L) ** (1.0 / 3.0))
        # Fully developed turbulent duct flow (Dittus-Boelter)
        Nu_turb = 0.023 * (Re ** 0.8) * (self.Pr_air ** 0.3)

        # Smooth transition interpolation across 2300 <= Re <= 3000 to prevent step discontinuities
        if Re < 2300.0:
            Nu = Nu_lam
        elif Re > 3000.0:
            Nu = Nu_turb
        else:
            gamma = (Re - 2300.0) / (3000.0 - 2300.0)
            weight = gamma * gamma * (3.0 - 2.0 * gamma)
            Nu = (1.0 - weight) * Nu_lam + weight * Nu_turb

        # Convective heat transfer coefficient
        h_conv = (Nu * self.k_air) / (2.0 * self.s_f)

        # Resistance network calculation
        R_conv = 1.0 / (h_conv * self.A_total)
        R_tim = self.t_tim / (k_tim * self.A_die)
        R_cond = self.R_cond_base if include_base_conduction else 0.0

        R_hs = R_conv + R_cond
        R_total = self.R_jc + R_tim + R_hs
        T_j = self.T_ambient + (TDP * R_total)

        return {
            'Re': float(Re),
            'Nu': float(Nu),
            'h_conv': float(h_conv),
            'R_tim': float(R_tim),
            'R_cond': float(R_cond),
            'R_conv': float(R_conv),
            'R_hs': float(R_hs),
            'R_total': float(R_total),
            'T_j': float(T_j)
        }

HeatSinkAnalyticalModel = PlateFinHeatSink

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Baseline Heat Sink Analytical Evaluation")
    parser.add_argument("--tdp", type=float, default=150.0, help="Thermal Design Power (W)")
    parser.add_argument("--vel", type=float, default=1.0, help="Approach air velocity (m/s)")
    parser.add_argument("--ktim", type=float, default=4.0, help="TIM thermal conductivity (W/m.K)")
    args = parser.parse_args()
    
    sink = PlateFinHeatSink()
    # Baseline benchmark test at TDP=150W, V=1m/s, k_tim=4 W/mK
    results = sink.evaluate(TDP=args.tdp, V_air=args.vel, k_tim=args.ktim, include_base_conduction=True)
    print("Baseline Analytical Evaluation:")
    for k, v in results.items():
        print(f"  {k:10s}: {v:.6f}")