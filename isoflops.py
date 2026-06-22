#!/usr/bin/env python3
"""
CS336 Assignment 3 - IsoFLOPs Scaling Laws
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def find_optimal_points(data):
    """For each compute budget, find run with minimum loss."""
    runs_by_compute = defaultdict(list)
    for run in data:
        runs_by_compute[run['compute_budget']].append(run)
    
    optimal_points = []
    for C in sorted(runs_by_compute.keys()):
        runs = runs_by_compute[C]
        best = min(runs, key=lambda r: r['final_loss'])
        N_opt = best['parameters']
        D_opt = C / (6 * N_opt)  # From C = 6*N*D
        optimal_points.append({
            'compute_budget': C,
            'N_opt': N_opt,
            'D_opt': D_opt,
            'loss': best['final_loss']
        })
    return optimal_points


def fit_power_law_logspace(C_values, y_values):
    """Fit y = a*C^b using log-space linear regression."""
    log_C = np.log10(C_values)
    log_y = np.log10(y_values)
    coeffs = np.polyfit(log_C, log_y, 1)
    b = coeffs[0]
    a = 10 ** coeffs[1]
    return a, b


def main():
    # 1. Load data
    data = load_data('data/isoflops_curves.json')
    
    # 2. Find optimal points
    optimal_points = find_optimal_points(data)
    
    # 3. Extract arrays
    C_values = np.array([p['compute_budget'] for p in optimal_points])
    N_opt_values = np.array([p['N_opt'] for p in optimal_points])
    D_opt_values = np.array([p['D_opt'] for p in optimal_points])
    
    # 4. Fit power laws in log-space
    a_N, b_N = fit_power_law_logspace(C_values, N_opt_values)
    a_D, b_D = fit_power_law_logspace(C_values, D_opt_values)
    
    print(f"N_opt(C) = {a_N:.4e} * C^{b_N:.4f}")
    print(f"D_opt(C) = {a_D:.4e} * C^{b_D:.4f}")
    
    # 5. Predictions
    for C in [1e23, 1e24]:
        N_pred = a_N * (C ** b_N)
        D_pred = a_D * (C ** b_D)
        print(f"C={C:.0e}: N_opt={N_pred:.2e}, D_opt={D_pred:.2e}")


if __name__ == '__main__':
    main()