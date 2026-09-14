# ============================================================
# TRACK PARETO-FRONT PERFORMANCE FOR EACH ITERATION
# Run this AFTER recalculating pareto_df
# ============================================================

from pymoo.indicators.hv import HV
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Create convergence-history table if it does not exist yet
# ------------------------------------------------------------

if "convergence_history" not in globals():

    convergence_history = pd.DataFrame(
        columns=[
            "Iteration",
            "Cumulative_Designs",
            "Pareto_Designs",
            "Min_CV_D",
            "Min_CV_F",
            "Min_ZDI",
            "Hypervolume"
        ]
    )


# ------------------------------------------------------------
# 2. Extract current objective values
# ------------------------------------------------------------

F_all = cumulative_df[
    ["CV_D", "CV_F", "ZDI"]
].to_numpy(dtype=float)

F_pareto = pareto_df[
    ["CV_D", "CV_F", "ZDI"]
].to_numpy(dtype=float)


# ------------------------------------------------------------
# 3. Hypervolume reference point
#
# IMPORTANT:
# Use the SAME reference point for every iteration.
#
# Here we create it the first time only and keep it unchanged.
# The reference point should be worse (larger) than all
# objective values because all 3 objectives are minimized.
# ------------------------------------------------------------

if "hv_reference_point" not in globals():

    hv_reference_point = (
        np.max(F_all, axis=0) * 1.10
    )

    print(
        "Fixed Hypervolume Reference Point:",
        hv_reference_point
    )


# ------------------------------------------------------------
# 4. Calculate Hypervolume
# ------------------------------------------------------------

hv_indicator = HV(
    ref_point=hv_reference_point
)

hypervolume = hv_indicator(
    F_pareto
)


# ------------------------------------------------------------
# 5. Current performance values
# ------------------------------------------------------------

current_row = pd.DataFrame([{

    "Iteration":
        iteration_number,

    "Cumulative_Designs":
        len(cumulative_df),

    "Pareto_Designs":
        len(pareto_df),

    "Min_CV_D":
        pareto_df["CV_D"].min(),

    "Min_CV_F":
        pareto_df["CV_F"].min(),

    "Min_ZDI":
        pareto_df["ZDI"].min(),

    "Hypervolume":
        hypervolume
}])


# ------------------------------------------------------------
# 6. Avoid duplicate entry if cell is accidentally rerun
# ------------------------------------------------------------

convergence_history = convergence_history[
    convergence_history["Iteration"]
    != iteration_number
]

convergence_history = pd.concat(
    [
        convergence_history,
        current_row
    ],
    ignore_index=True
)


# Sort by iteration
convergence_history = convergence_history.sort_values(
    "Iteration"
).reset_index(drop=True)


# ------------------------------------------------------------
# 7. Display convergence table
# ------------------------------------------------------------

display(
    convergence_history
)

# ============================================================
# SAVE CONVERGENCE HISTORY
# ============================================================

convergence_history.to_csv(
    "FinRay_Pareto_Convergence_History.csv",
    index=False
)

print(
    "Convergence history saved."
)

# ============================================================
# HYPERVOLUME CONVERGENCE PLOT
# ============================================================

import matplotlib.pyplot as plt

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    convergence_history[
        "Cumulative_Designs"
    ],

    convergence_history[
        "Hypervolume"
    ],

    marker="o"
)

plt.xlabel(
    "Cumulative Number of Designs"
)

plt.ylabel(
    "Hypervolume"
)

plt.title(
    "Pareto Front Hypervolume Convergence"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Pareto_Hypervolume_Convergence.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()