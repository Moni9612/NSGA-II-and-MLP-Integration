# ============================================================
# FINAL CONSTRAINED 3D PARETO FRONT
#
# Objectives:
#   Minimize CV_D
#   Minimize CV_F
#   Minimize ZDI
#
# Constraint:
#   Mean_Fn >= 10 N
#
# FINAL DESIGN SELECTION:
#   Minimum Distance to Utopia
#
# NEW:
#   + SELECT TOP 5 UTOPIA-NEAREST DESIGNS
#   + MARK TOP 5 ON 3D PLOT WITH A-E ARROWS
#   + PRINT THEIR GEOMETRIES
#   + PRINT THEIR OBJECTIVE VALUES
#   + PRINT THEIR MEAN FORCE VALUES
#   + FOR EACH OF THE TOP 5:
#       PRINT ALL 12 PREDICTED Fn
#       PRINT ALL 12 PREDICTED delta_x
#       PRINT ALL 12 PREDICTED delta_y
#       PRINT ALL 12 PREDICTED delta_z
#
# NO CSV EXPORT FOR 12-CONDITION PREDICTIONS
#
# DISPLAY RANGES:
#   CV_F : 0.00 : 0.05 : 0.25
#   ZDI  : 0.00 : 0.05 : 0.70
#
# DISPLAY COORDINATES:
#   X = CV_F
#   Y = CV_D
#   Z = ZDI
#
# CAMERA:
#   Reduced CV_F foreshortening
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


# ============================================================
# 2. COLUMN DEFINITIONS
# ============================================================

design_columns = [
    "t_rib1",
    "r_angle",
    "t_c",
    "N",
    "L"
]

objective_columns = [
    "CV_D",
    "CV_F",
    "ZDI"
]

MIN_MEAN_FORCE = 15.0

TOP_N_DESIGNS = 5


# ============================================================
# 3. SAFETY CHECKS
# ============================================================

required_columns = (
    design_columns
    +
    objective_columns
    +
    [
        "Mean_Fn",
        "Feasible"
    ]
)


missing_cumulative = [
    col
    for col in required_columns
    if col not in cumulative_df.columns
]


missing_pareto = [
    col
    for col in required_columns
    if col not in pareto_df.columns
]


if missing_cumulative:

    raise ValueError(
        "cumulative_df is missing columns: "
        + str(missing_cumulative)
    )


if missing_pareto:

    raise ValueError(
        "pareto_df is missing columns: "
        + str(missing_pareto)
    )


if len(pareto_df) == 0:

    raise ValueError(
        "pareto_df is empty."
    )


if "evaluate_geometry_with_mlp" not in globals():

    raise ValueError(
        "evaluate_geometry_with_mlp() is not defined. "
        "Run your MLP evaluation-function cell first."
    )


# ============================================================
# 4. VERIFY CONSTRAINED PARETO FRONT
# ============================================================

constraint_check = (
    pareto_df["Mean_Fn"]
    >=
    MIN_MEAN_FORCE
).all()


print("=" * 80)
print("CONSTRAINED PARETO FRONT CHECK")
print("=" * 80)


print(
    f"Force constraint: Mean_Fn >= "
    f"{MIN_MEAN_FORCE:.1f} N"
)


print(
    "Total evaluated designs:",
    len(cumulative_df)
)


print(
    "Total feasible evaluated designs:",
    (
        cumulative_df["Mean_Fn"]
        >=
        MIN_MEAN_FORCE
    ).sum()
)


print(
    "Total constrained Pareto designs:",
    len(pareto_df)
)


print(
    "Minimum Pareto Mean_Fn:",
    f"{pareto_df['Mean_Fn'].min():.3f} N"
)


print(
    "Maximum Pareto Mean_Fn:",
    f"{pareto_df['Mean_Fn'].max():.3f} N"
)


print(
    "All Pareto designs satisfy force constraint:",
    constraint_check
)


if not constraint_check:

    raise ValueError(
        "pareto_df contains infeasible designs. "
        "Recalculate pareto_df using constrained "
        "non-dominated sorting."
    )


# ============================================================
# 5. MINIMUM DISTANCE TO UTOPIA
#
# All three objectives are minimized.
# ============================================================

pareto_objectives = pareto_df[
    objective_columns
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# Min and max of each Pareto objective
# ------------------------------------------------------------

obj_min = pareto_objectives.min(
    axis=0
)


obj_max = pareto_objectives.max(
    axis=0
)


obj_range = (
    obj_max
    -
    obj_min
)


obj_range[
    obj_range == 0
] = 1.0


# ------------------------------------------------------------
# Min-max normalization
# ------------------------------------------------------------

F_norm = (
    pareto_objectives
    -
    obj_min
) / obj_range


# ------------------------------------------------------------
# Normalized utopia point
# ------------------------------------------------------------

utopia_point = np.array(
    [
        0.0,
        0.0,
        0.0
    ]
)


# ------------------------------------------------------------
# Euclidean distance to utopia
# ------------------------------------------------------------

utopia_distances = np.linalg.norm(
    F_norm
    -
    utopia_point,
    axis=1
)


# ============================================================
# 6. CREATE RANKED PARETO DATAFRAME
# ============================================================

ranked_pareto_df = (
    pareto_df.copy()
)


ranked_pareto_df[
    "Normalized_CV_D"
] = F_norm[:, 0]


ranked_pareto_df[
    "Normalized_CV_F"
] = F_norm[:, 1]


ranked_pareto_df[
    "Normalized_ZDI"
] = F_norm[:, 2]


ranked_pareto_df[
    "Utopia_Distance"
] = utopia_distances


ranked_pareto_df = (
    ranked_pareto_df
    .sort_values(
        by="Utopia_Distance",
        ascending=True
    )
    .reset_index(
        drop=True
    )
)


ranked_pareto_df[
    "Utopia_Rank"
] = np.arange(
    1,
    len(ranked_pareto_df) + 1
)


# ============================================================
# 7. EXTRACT BEST 5 DESIGNS
# ============================================================

n_best = min(
    TOP_N_DESIGNS,
    len(ranked_pareto_df)
)


best_5_df = (
    ranked_pareto_df
    .head(
        n_best
    )
    .copy()
)


# ============================================================
# 8. PRINT BEST 5 DESIGN SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("TOP 5 DESIGNS — MINIMUM DISTANCE TO UTOPIA")
print("=" * 100)


for _, row in best_5_df.iterrows():

    rank = int(
        row["Utopia_Rank"]
    )


    print(
        f"\n{'=' * 80}"
    )

    print(
        f"RANK {rank}"
    )

    print(
        f"{'=' * 80}"
    )


    print("\nGEOMETRY")


    print(
        f"t_rib1 = "
        f"{row['t_rib1']}"
    )


    print(
        f"r_angle = "
        f"{row['r_angle']}"
    )


    print(
        f"t_c = "
        f"{row['t_c']}"
    )


    print(
        f"N = "
        f"{int(row['N'])}"
    )


    print(
        f"L = "
        f"{row['L']}"
    )


    print("\nOBJECTIVES")


    print(
        f"CV_D = "
        f"{row['CV_D']:.6f}"
    )


    print(
        f"CV_F = "
        f"{row['CV_F']:.6f}"
    )


    print(
        f"ZDI  = "
        f"{row['ZDI']:.6f}"
    )


    print(
        f"Mean Fn = "
        f"{row['Mean_Fn']:.6f} N"
    )


    print(
        f"Utopia distance = "
        f"{row['Utopia_Distance']:.6f}"
    )


# ============================================================
# 9. PREDICT ALL 12 CONDITIONS FOR TOP 5
#
# evaluate_geometry_with_mlp() is expected to return:
#
# result["Fn"]
# result["delta_x"]
# result["delta_y"]
# result["delta_z"]
#
# each containing 12 values.
# ============================================================

print("\n" + "=" * 110)
print("12-CONDITION MLP PREDICTIONS FOR TOP 5 DESIGNS")
print("=" * 110)


top5_prediction_results = {}


for _, design_row in best_5_df.iterrows():

    rank = int(
        design_row["Utopia_Rank"]
    )


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = design_row[
        design_columns
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Predict all 12 operating conditions
    # --------------------------------------------------------

    result = evaluate_geometry_with_mlp(
        geometry
    )


    Fn_values = np.asarray(
        result["Fn"],
        dtype=float
    ).flatten()


    delta_x_values = np.asarray(
        result["delta_x"],
        dtype=float
    ).flatten()


    delta_y_values = np.asarray(
        result["delta_y"],
        dtype=float
    ).flatten()


    delta_z_values = np.asarray(
        result["delta_z"],
        dtype=float
    ).flatten()


    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    lengths = {
        "Fn": len(Fn_values),
        "delta_x": len(delta_x_values),
        "delta_y": len(delta_y_values),
        "delta_z": len(delta_z_values)
    }


    for name, length in lengths.items():

        if length != 12:

            raise ValueError(
                f"Rank {rank}: "
                f"{name} returned {length} values "
                f"instead of 12."
            )


    # --------------------------------------------------------
    # Store in memory
    # --------------------------------------------------------

    top5_prediction_results[
        rank
    ] = {
        "Fn":
            Fn_values,

        "delta_x":
            delta_x_values,

        "delta_y":
            delta_y_values,

        "delta_z":
            delta_z_values
    }


    # --------------------------------------------------------
    # Print design header
    # --------------------------------------------------------

    print(
        f"\n{'=' * 110}"
    )

    print(
        f"RANK {rank}"
    )

    print(
        f"{'=' * 110}"
    )


    print(
        f"Geometry: "
        f"t_rib1={design_row['t_rib1']}, "
        f"r_angle={design_row['r_angle']}, "
        f"t_c={design_row['t_c']}, "
        f"N={int(design_row['N'])}, "
        f"L={design_row['L']}"
    )


    print(
        f"CV_D={design_row['CV_D']:.6f}, "
        f"CV_F={design_row['CV_F']:.6f}, "
        f"ZDI={design_row['ZDI']:.6f}, "
        f"Utopia distance="
        f"{design_row['Utopia_Distance']:.6f}"
    )


    # --------------------------------------------------------
    # Print table header
    # --------------------------------------------------------

    print()

    print(
        f"{'Condition':<12}"
        f"{'Fn (N)':>14}"
        f"{'delta_x':>16}"
        f"{'delta_y':>16}"
        f"{'delta_z':>16}"
    )

    print(
        "-" * 74
    )


    # --------------------------------------------------------
    # Print all 12 conditions
    # --------------------------------------------------------

    for i in range(12):

        print(
            f"{i + 1:<12}"
            f"{Fn_values[i]:>14.6f}"
            f"{delta_x_values[i]:>16.6f}"
            f"{delta_y_values[i]:>16.6f}"
            f"{delta_z_values[i]:>16.6f}"
        )


    # --------------------------------------------------------
    # Print force summary
    # --------------------------------------------------------

    calculated_mean_fn = np.mean(
        Fn_values
    )


    calculated_std_fn = np.std(
        Fn_values,
        ddof=0
    )


    print(
        "\nPredicted force summary:"
    )


    print(
        f"Mean Fn = "
        f"{calculated_mean_fn:.6f} N"
    )


    print(
        f"Std Fn  = "
        f"{calculated_std_fn:.6f} N"
    )


print("\n" + "=" * 110)


# ============================================================
# 10. OPTIONAL DISPLAY TABLE FOR EACH DESIGN
#
# This does NOT save anything.
# It only displays five tables in Colab.
# ============================================================

for _, design_row in best_5_df.iterrows():

    rank = int(
        design_row["Utopia_Rank"]
    )


    values = (
        top5_prediction_results[
            rank
        ]
    )


    condition_table = pd.DataFrame(
        {
            "Condition":
                np.arange(
                    1,
                    13
                ),

            "Fn":
                values["Fn"],

            "delta_x":
                values["delta_x"],

            "delta_y":
                values["delta_y"],

            "delta_z":
                values["delta_z"]
        }
    )


    # --------------------------------------------------------
    # Add actual operating-condition labels if available
    # --------------------------------------------------------

    if (
        "operating_conditions"
        in globals()
        and
        len(operating_conditions) == 12
    ):

        if (
            "position"
            in operating_conditions.columns
        ):

            condition_table.insert(
                1,
                "position",
                operating_conditions[
                    "position"
                ].to_numpy()
            )


        if (
            "orientation"
            in operating_conditions.columns
        ):

            insert_position = (
                2
                if
                "position"
                in condition_table.columns
                else
                1
            )


            condition_table.insert(
                insert_position,
                "orientation",
                operating_conditions[
                    "orientation"
                ].to_numpy()
            )


    print(
        f"\nRANK {rank} — 12 PREDICTED OPERATING CONDITIONS"
    )


    display(
        condition_table
    )


# ============================================================
# 11. SELECT RANK-1 DESIGN FOR PLOT
# ============================================================

selected_design = (
    ranked_pareto_df
    .iloc[0]
    .copy()
)


selected_CVD = float(
    selected_design[
        "CV_D"
    ]
)


selected_CVF = float(
    selected_design[
        "CV_F"
    ]
)


selected_ZDI = float(
    selected_design[
        "ZDI"
    ]
)


selected_mean_force = float(
    selected_design[
        "Mean_Fn"
    ]
)


selected_distance = float(
    selected_design[
        "Utopia_Distance"
    ]
)


# ============================================================
# 12. COMPLETE FEASIBLE POPULATION
# ============================================================

plot_cumulative_df = cumulative_df[

    cumulative_df[
        "Mean_Fn"
    ]
    >=
    MIN_MEAN_FORCE

].copy()


plot_pareto_df = (
    pareto_df.copy()
)


# ============================================================
# 13. EXTRACT COMPLETE DATA
# ============================================================

CVD_all_full = (
    plot_cumulative_df[
        "CV_D"
    ]
    .to_numpy(
        dtype=float
    )
)


CVF_all_full = (
    plot_cumulative_df[
        "CV_F"
    ]
    .to_numpy(
        dtype=float
    )
)


ZDI_all_full = (
    plot_cumulative_df[
        "ZDI"
    ]
    .to_numpy(
        dtype=float
    )
)


CVD_p_full = (
    plot_pareto_df[
        "CV_D"
    ]
    .to_numpy(
        dtype=float
    )
)


CVF_p_full = (
    plot_pareto_df[
        "CV_F"
    ]
    .to_numpy(
        dtype=float
    )
)


ZDI_p_full = (
    plot_pareto_df[
        "ZDI"
    ]
    .to_numpy(
        dtype=float
    )
)


# ============================================================
# 14. DISPLAY LIMITS
# ============================================================

CVD_MIN = 0.0


CVD_data_max = max(
    CVD_all_full.max(),
    CVD_p_full.max()
)


CVD_MAX = (
    np.ceil(
        CVD_data_max
        *
        10
    )
    /
    10
)


CVD_MAX = max(
    1.0,
    CVD_MAX
)


CVF_MIN = 0.00
CVF_MAX = 0.25
CVF_STEP = 0.05


ZDI_MIN = 0.00
ZDI_MAX = 0.70
ZDI_STEP = 0.05


# ============================================================
# 15. VISUAL DISPLAY FILTER
#
# Display only.
# Does not affect Pareto ranking.
# ============================================================

display_cumulative_df = plot_cumulative_df[

    (
        plot_cumulative_df[
            "CV_F"
        ]
        >=
        CVF_MIN
    )

    &

    (
        plot_cumulative_df[
            "CV_F"
        ]
        <=
        CVF_MAX
    )

    &

    (
        plot_cumulative_df[
            "ZDI"
        ]
        >=
        ZDI_MIN
    )

    &

    (
        plot_cumulative_df[
            "ZDI"
        ]
        <=
        ZDI_MAX
    )

].copy()


display_pareto_df = plot_pareto_df[

    (
        plot_pareto_df[
            "CV_F"
        ]
        >=
        CVF_MIN
    )

    &

    (
        plot_pareto_df[
            "CV_F"
        ]
        <=
        CVF_MAX
    )

    &

    (
        plot_pareto_df[
            "ZDI"
        ]
        >=
        ZDI_MIN
    )

    &

    (
        plot_pareto_df[
            "ZDI"
        ]
        <=
        ZDI_MAX
    )

].copy()


# ============================================================
# 16. DISPLAY COORDINATES
#
# X = CV_F
# Y = CV_D
# Z = ZDI
# ============================================================

X_all = (
    display_cumulative_df[
        "CV_F"
    ]
    .to_numpy(
        dtype=float
    )
)


Y_all = (
    display_cumulative_df[
        "CV_D"
    ]
    .to_numpy(
        dtype=float
    )
)


Z_all = (
    display_cumulative_df[
        "ZDI"
    ]
    .to_numpy(
        dtype=float
    )
)


X_p = (
    display_pareto_df[
        "CV_F"
    ]
    .to_numpy(
        dtype=float
    )
)


Y_p = (
    display_pareto_df[
        "CV_D"
    ]
    .to_numpy(
        dtype=float
    )
)


Z_p = (
    display_pareto_df[
        "ZDI"
    ]
    .to_numpy(
        dtype=float
    )
)


selected_X = selected_CVF
selected_Y = selected_CVD
selected_Z = selected_ZDI


# ============================================================
# 17. PARETO QUALITY SCORE
#
# Visualization only.
# ============================================================

full_pareto_objectives = np.column_stack(
    (
        CVD_p_full,
        CVF_p_full,
        ZDI_p_full
    )
)


pareto_plot_min = (
    full_pareto_objectives.min(
        axis=0
    )
)


pareto_plot_max = (
    full_pareto_objectives.max(
        axis=0
    )
)


pareto_plot_range = (
    pareto_plot_max
    -
    pareto_plot_min
)


pareto_plot_range[
    pareto_plot_range == 0
] = 1.0


pareto_normalized = (
    full_pareto_objectives
    -
    pareto_plot_min
) / pareto_plot_range


pareto_cost_full = np.mean(
    pareto_normalized,
    axis=1
)


pareto_quality_full = (
    1.0
    -
    pareto_cost_full
)


quality_min = (
    pareto_quality_full.min()
)


quality_max = (
    pareto_quality_full.max()
)


if quality_max > quality_min:

    pareto_quality_full = (
        pareto_quality_full
        -
        quality_min
    ) / (
        quality_max
        -
        quality_min
    )

else:

    pareto_quality_full = np.ones_like(
        pareto_quality_full
    )


plot_pareto_quality_df = (
    plot_pareto_df.copy()
)


plot_pareto_quality_df[
    "Plot_Quality"
] = pareto_quality_full


display_pareto_quality_df = (

    plot_pareto_quality_df[

        (
            plot_pareto_quality_df[
                "CV_F"
            ]
            >=
            CVF_MIN
        )

        &

        (
            plot_pareto_quality_df[
                "CV_F"
            ]
            <=
            CVF_MAX
        )

        &

        (
            plot_pareto_quality_df[
                "ZDI"
            ]
            >=
            ZDI_MIN
        )

        &

        (
            plot_pareto_quality_df[
                "ZDI"
            ]
            <=
            ZDI_MAX
        )

    ]

    .copy()
)


pareto_quality = (
    display_pareto_quality_df[
        "Plot_Quality"
    ]
    .to_numpy(
        dtype=float
    )
)


# ============================================================
# 18. CREATE 3D FIGURE
# ============================================================

fig = plt.figure(

    figsize=(
        14,
        11
    ),

    facecolor="white"
)


ax = fig.add_subplot(

    111,

    projection="3d"
)


# ============================================================
# 19. HIDE DEFAULT PANES
# ============================================================

ax.xaxis.pane.set_visible(
    False
)

ax.yaxis.pane.set_visible(
    False
)

ax.zaxis.pane.set_visible(
    False
)


ax.grid(
    False
)


try:

    ax.zaxis.line.set_visible(
        False
    )

except Exception:

    pass


# ============================================================
# 20. MANUAL PLANES
# ============================================================

floor_vertices = [[

    (
        CVF_MIN,
        CVD_MIN,
        ZDI_MIN
    ),

    (
        CVF_MAX,
        CVD_MIN,
        ZDI_MIN
    ),

    (
        CVF_MAX,
        CVD_MAX,
        ZDI_MIN
    ),

    (
        CVF_MIN,
        CVD_MAX,
        ZDI_MIN
    )

]]


floor = Poly3DCollection(

    floor_vertices,

    facecolor="lightgray",

    edgecolor="black",

    linewidth=0.8,

    alpha=0.18
)


ax.add_collection3d(
    floor
)


# ------------------------------------------------------------
# CV_D - ZDI wall
# ------------------------------------------------------------

wall_CVD_ZDI_vertices = [[

    (
        CVF_MIN,
        CVD_MIN,
        ZDI_MIN
    ),

    (
        CVF_MIN,
        CVD_MAX,
        ZDI_MIN
    ),

    (
        CVF_MIN,
        CVD_MAX,
        ZDI_MAX
    ),

    (
        CVF_MIN,
        CVD_MIN,
        ZDI_MAX
    )

]]


wall_CVD_ZDI = Poly3DCollection(

    wall_CVD_ZDI_vertices,

    facecolor="lightgray",

    edgecolor="black",

    linewidth=0.8,

    alpha=0.14
)


ax.add_collection3d(
    wall_CVD_ZDI
)


# ------------------------------------------------------------
# CV_F - ZDI wall
# ------------------------------------------------------------

wall_CVF_ZDI_vertices = [[

    (
        CVF_MIN,
        CVD_MIN,
        ZDI_MIN
    ),

    (
        CVF_MAX,
        CVD_MIN,
        ZDI_MIN
    ),

    (
        CVF_MAX,
        CVD_MIN,
        ZDI_MAX
    ),

    (
        CVF_MIN,
        CVD_MIN,
        ZDI_MAX
    )

]]


wall_CVF_ZDI = Poly3DCollection(

    wall_CVF_ZDI_vertices,

    facecolor="lightgray",

    edgecolor="black",

    linewidth=0.8,

    alpha=0.14
)


ax.add_collection3d(
    wall_CVF_ZDI
)


# ============================================================
# 21. GRID DEFINITIONS
# ============================================================

CVF_grid = np.arange(
    CVF_MIN,
    CVF_MAX + 0.001,
    CVF_STEP
)


CVD_grid = np.linspace(
    CVD_MIN,
    CVD_MAX,
    11
)


ZDI_grid = np.arange(
    ZDI_MIN,
    ZDI_MAX + 0.001,
    ZDI_STEP
)


# ============================================================
# 22. FLOOR GRID
# ============================================================

for cvf in CVF_grid:

    ax.plot(

        [
            cvf,
            cvf
        ],

        [
            CVD_MIN,
            CVD_MAX
        ],

        [
            ZDI_MIN,
            ZDI_MIN
        ],

        color="gray",

        linewidth=0.45,

        alpha=0.35
    )


for cvd in CVD_grid:

    ax.plot(

        [
            CVF_MIN,
            CVF_MAX
        ],

        [
            cvd,
            cvd
        ],

        [
            ZDI_MIN,
            ZDI_MIN
        ],

        color="gray",

        linewidth=0.45,

        alpha=0.35
    )


# ============================================================
# 23. CVD-ZDI WALL GRID
# ============================================================

for cvd in CVD_grid:

    ax.plot(

        [
            CVF_MIN,
            CVF_MIN
        ],

        [
            cvd,
            cvd
        ],

        [
            ZDI_MIN,
            ZDI_MAX
        ],

        color="gray",

        linewidth=0.45,

        alpha=0.30
    )


for zdi in ZDI_grid:

    ax.plot(

        [
            CVF_MIN,
            CVF_MIN
        ],

        [
            CVD_MIN,
            CVD_MAX
        ],

        [
            zdi,
            zdi
        ],

        color="gray",

        linewidth=0.45,

        alpha=0.30
    )


# ============================================================
# 24. CVF-ZDI WALL GRID
# ============================================================

for cvf in CVF_grid:

    ax.plot(

        [
            cvf,
            cvf
        ],

        [
            CVD_MIN,
            CVD_MIN
        ],

        [
            ZDI_MIN,
            ZDI_MAX
        ],

        color="gray",

        linewidth=0.45,

        alpha=0.30
    )


for zdi in ZDI_grid:

    ax.plot(

        [
            CVF_MIN,
            CVF_MAX
        ],

        [
            CVD_MIN,
            CVD_MIN
        ],

        [
            zdi,
            zdi
        ],

        color="gray",

        linewidth=0.45,

        alpha=0.30
    )


# ============================================================
# 25. FEASIBLE DESIGNS
# ============================================================

ax.scatter(

    X_all,
    Y_all,
    Z_all,

    s=25,

    facecolors="none",

    edgecolors="#3767B1",

    linewidth=0.8,

    alpha=0.60,

    depthshade=False,

    label=(
        f"Feasible Evaluated Designs "
        f"({len(plot_cumulative_df)})"
    )
)


# ============================================================
# 26. PARETO DESIGNS
# ============================================================

if len(X_p) > 0:

    pareto_scatter = ax.scatter(

        X_p,
        Y_p,
        Z_p,

        s=90,

        c=pareto_quality,

        cmap="turbo",

        vmin=0.0,

        vmax=1.0,

        edgecolor="white",

        linewidth=0.75,

        alpha=1.0,

        depthshade=False,

        label=(
            f"Constrained Pareto Designs "
            f"({len(plot_pareto_df)})"
        ),

        zorder=10
    )


# ============================================================
# 27. TOP-5 DESIGNS FOR ARROW ANNOTATION
#
# The previous Rank-1 red star is intentionally removed.
# A, B, C, D, E correspond to Utopia ranks 1, 2, 3, 4, 5.
# Actual arrows are added after the camera/projection settings
# so that their arrowheads land correctly on the 3D points.
# ============================================================

top5_plot_df = best_5_df.copy()

top5_plot_df["Arrow_Label"] = list(
    "ABCDE"[:len(top5_plot_df)]
)

print("\nTOP-5 PLOT LABELS")
print("-" * 40)

for _, row in top5_plot_df.iterrows():

    print(
        f"{row['Arrow_Label']} = "
        f"Rank {int(row['Utopia_Rank'])}"
    )


# ============================================================
# 28. AXIS LIMITS
# ============================================================

ax.set_xlim(
    CVF_MAX,
    CVF_MIN
)


ax.set_ylim(
    CVD_MAX,
    CVD_MIN
)


ax.set_zlim(
    ZDI_MIN,
    ZDI_MAX
)


# ============================================================
# 29. CVF TICKS
# ============================================================

CVF_ticks = np.arange(
    CVF_MIN,
    CVF_MAX + 0.001,
    CVF_STEP
)


ax.set_xticks(
    CVF_ticks
)


ax.set_xticklabels(
    [
        f"{value:.2f}"
        for value
        in CVF_ticks
    ]
)


# ============================================================
# 30. CVD TICKS
# ============================================================

CVD_ticks = np.linspace(
    CVD_MIN,
    CVD_MAX,
    6
)


ax.set_yticks(
    CVD_ticks
)


ax.set_yticklabels(
    [
        f"{value:.2f}"
        for value
        in CVD_ticks
    ]
)


# ============================================================
# 31. CUSTOM ZDI AXIS
# ============================================================

ax.set_zticks(
    [
        ZDI_MIN
    ]
)


ax.set_zticklabels(
    [
        ""
    ]
)


ax.set_zlabel(
    ""
)


ZDI_ticks = np.arange(
    ZDI_MIN,
    ZDI_MAX + 0.001,
    ZDI_STEP
)


ax.plot(

    [
        CVF_MIN,
        CVF_MIN
    ],

    [
        CVD_MIN,
        CVD_MIN
    ],

    [
        ZDI_MIN,
        ZDI_MAX
    ],

    color="black",

    linewidth=2.0,

    zorder=200
)


z_tick_length = (
    0.025
    *
    (
        CVF_MAX
        -
        CVF_MIN
    )
)


for z_tick in ZDI_ticks:

    ax.plot(

        [
            CVF_MIN,
            CVF_MIN
            +
            z_tick_length
        ],

        [
            CVD_MIN,
            CVD_MIN
        ],

        [
            z_tick,
            z_tick
        ],

        color="black",

        linewidth=1.0,

        zorder=200
    )


    ax.text(

        CVF_MIN,

        CVD_MIN
        -
        0.025
        *
        (
            CVD_MAX
            -
            CVD_MIN
        ),

        z_tick,

        f"{z_tick:.2f}",

        fontsize=9,

        horizontalalignment="right",

        verticalalignment="center",

        color="black",

        zorder=200
    )


# ============================================================
# 32. AXIS LABELS
# ============================================================

ax.set_xlabel(

    r"$CV_F=\sigma_F/\mu_F$",

    fontsize=14,

    fontweight="bold",

    labelpad=22
)


ax.set_ylabel(

    r"$CV_D=\sigma_D/\mu_D$",

    fontsize=14,

    fontweight="bold",

    labelpad=22
)


ax.text2D(

    -0.045,

    0.52,

    "ZDI",

    transform=ax.transAxes,

    rotation=90,

    fontsize=14,

    fontweight="bold",

    horizontalalignment="center",

    verticalalignment="center"
)


# ============================================================
# 33. CAMERA
#
# Reduced CVF foreshortening
# ============================================================

ax.view_init(

    elev=18,

    azim=165
)


# ============================================================
# 34. PERSPECTIVE
# ============================================================

try:

    ax.set_proj_type(

        "persp",

        focal_length=1.05
    )

except TypeError:

    ax.set_proj_type(
        "persp"
    )


# ============================================================
# 35. BOX ASPECT
# ============================================================

ax.set_box_aspect(

    (
        1.20,   # CV_F

        1.50,   # CV_D

        1.45    # ZDI
    )
)


# ============================================================
# 36. TOP-5 ARROWS AND A-E LABELS
#
# A = Rank 1
# B = Rank 2
# C = Rank 3
# D = Rank 4
# E = Rank 5
#
# Labels are stacked from top to bottom on the left side.
# Each red arrow points directly to the corresponding 3D point.
# ============================================================

# Force Matplotlib to finalize the current 3D projection matrix
# before converting the 3D points to 2D screen coordinates.
fig.canvas.draw()

# Fixed label positions in Axes coordinates.
# These keep A-E neatly ordered from top to bottom.
arrow_label_x = 0.14

arrow_label_y_positions = np.linspace(
    0.68,
    0.40,
    len(top5_plot_df)
)

for (
    (_, row),
    label_y
) in zip(
    top5_plot_df.iterrows(),
    arrow_label_y_positions
):

    label = row["Arrow_Label"]

    point_x = float(
        row["CV_F"]
    )

    point_y = float(
        row["CV_D"]
    )

    point_z = float(
        row["ZDI"]
    )

    # --------------------------------------------------------
    # Check whether this point lies inside the displayed range
    # --------------------------------------------------------

    point_visible = (

        point_x >= CVF_MIN
        and
        point_x <= CVF_MAX

        and

        point_z >= ZDI_MIN
        and
        point_z <= ZDI_MAX

        and

        point_y >= CVD_MIN
        and
        point_y <= CVD_MAX
    )

    if not point_visible:

        print(
            f"Warning: {label} "
            f"(Rank {int(row['Utopia_Rank'])}) "
            f"is outside the current plot display limits "
            f"and cannot be arrow-annotated visibly."
        )

        continue

    # --------------------------------------------------------
    # Project the 3D Pareto point into the 2D drawing plane
    # --------------------------------------------------------

    x_2d, y_2d, _ = proj3d.proj_transform(
        point_x,
        point_y,
        point_z,
        ax.get_proj()
    )

    # --------------------------------------------------------
    # Draw a small outline around the selected Pareto point.
    # This is NOT a star and does not cover the Pareto colour.
    # --------------------------------------------------------

    ax.scatter(

        [point_x],
        [point_y],
        [point_z],

        s=145,

        facecolors="none",

        edgecolors="black",

        linewidth=1.5,

        depthshade=False,

        zorder=120
    )

    # --------------------------------------------------------
    # Draw stacked A-E label and arrow
    # --------------------------------------------------------

    ax.annotate(

        label,

        xy=(
            x_2d,
            y_2d
        ),

        xycoords="data",

        xytext=(
            arrow_label_x,
            label_y
        ),

        textcoords=ax.transAxes,

        fontsize=17,

        fontweight="bold",

        color="red",

        horizontalalignment="center",

        verticalalignment="center",

        bbox=dict(

            boxstyle="round,pad=0.12",

            facecolor="white",

            edgecolor="none",

            alpha=0.78
        ),

        arrowprops=dict(

            arrowstyle="-|>",

            color="red",

            linewidth=2.4,

            mutation_scale=18,

            shrinkA=5,

            shrinkB=4,

            connectionstyle="arc3,rad=0.0"
        ),

        annotation_clip=False,

        zorder=300
    )


# ============================================================
# 37. TICK APPEARANCE
# ============================================================

ax.tick_params(

    axis="x",

    labelsize=12,

    pad=7
)


ax.tick_params(

    axis="y",

    labelsize=12,

    pad=7
)


# ============================================================
# 38. TITLE
# ============================================================

ax.set_title(

    f"Constrained Three-Objective Pareto Front "
    f"after {len(cumulative_df)} Evaluated Designs\n"

    r"$\mu_F \geq 15$ N | "
    r"Minimum-Distance-to-Utopia Selection",

    fontsize=16,

    fontweight="bold",

    pad=24
)


# ============================================================
# 39. LEGEND
# ============================================================

ax.legend(

    loc="upper right",

    bbox_to_anchor=(
        1.03,
        0.99
    ),

    fontsize=12,

    frameon=True
)


# ============================================================
# 40. COLORBAR
# ============================================================

if len(X_p) > 0:

    cbar = fig.colorbar(

        pareto_scatter,

        ax=ax,

        shrink=0.58,

        pad=0.09,

        aspect=25
    )


    cbar.set_label(

        "Normalized Pareto quality",

        fontsize=11,

        fontweight="bold",

        labelpad=10
    )


    cbar.set_ticks(
        [
            0.00,
            0.25,
            0.50,
            0.75,
            1.00
        ]
    )


    cbar.ax.tick_params(
        labelsize=9
    )


# ============================================================
# 41. OBJECTIVES + CONSTRAINT
# ============================================================

fig.text(

    0.765,

    0.145,

    "Optimization\n"

    r"$\downarrow\;CV_D$"
    "\n"

    r"$\downarrow\;CV_F$"
    "\n"

    r"$\downarrow\;ZDI$"

    "\n\n"

    "Constraint\n"

    r"$\mu_F \geq 15$ N",

    fontsize=10.5,

    bbox=dict(

        boxstyle="round,pad=0.5",

        facecolor="aliceblue",

        edgecolor="steelblue",

        alpha=0.9
    )
)


# ============================================================
# 42. POPULATION SUMMARY
# ============================================================

total_feasible_designs = len(
    plot_cumulative_df
)


total_constrained_pareto = len(
    plot_pareto_df
)


fig.text(

    0.765,

    0.275,

    "Design Population\n"

    f"Feasible evaluated: "
    f"{total_feasible_designs}\n"

    f"Constrained Pareto: "
    f"{total_constrained_pareto}",

    fontsize=10.5,

    bbox=dict(

        boxstyle="round,pad=0.5",

        facecolor="whitesmoke",

        edgecolor="gray",

        alpha=0.9
    )
)


# ============================================================
# 43. SELECTED DESIGN SUMMARY
# ============================================================

fig.text(

    0.765,

    0.385,

    "Rank 1 Design\n"

    f"Mean Fn: "
    f"{selected_mean_force:.2f} N\n"

    f"Utopia distance: "
    f"{selected_distance:.3f}",

    fontsize=10.5,

    bbox=dict(

        boxstyle="round,pad=0.5",

        facecolor="mistyrose",

        edgecolor="firebrick",

        alpha=0.9
    )
)


# ============================================================
# 44. FIGURE NOTE
# ============================================================

fig.text(

    0.08,

    0.050,

    r"All designs satisfy $\mu_F \geq 15$ N. "
    r"Display ranges: $CV_F\leq0.25$ and ZDI$\leq0.70$. "
    r"Utopia ranking uses the complete constrained Pareto front.",

    fontsize=9,

    style="italic"
)


# ============================================================
# 45. LAYOUT
# ============================================================

plt.subplots_adjust(

    left=0.08,

    right=0.84,

    bottom=0.11,

    top=0.89
)


# ============================================================
# 46. SAVE FIGURE
#
# Only the figure is saved.
# The 12-condition predictions are NOT saved to CSV.
# ============================================================

output_filename = (

    "Final_Constrained_Pareto_"
    "Top5_Utopia_Arrows_A_to_E.png"
)


plt.savefig(

    output_filename,

    dpi=600,

    facecolor="white",

    bbox_inches="tight"
)


print(
    f"\nFigure saved as: "
    f"{output_filename}"
)


# ============================================================
# 47. FINAL COMPACT TOP-5 SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("FINAL TOP-5 UTOPIA-NEAREST DESIGNS")
print("=" * 110)


for _, row in best_5_df.iterrows():

    rank = int(
        row["Utopia_Rank"]
    )


    values = (
        top5_prediction_results[
            rank
        ]
    )


    print(
        f"\nRank {rank}"
    )


    print(
        f"Geometry: "
        f"["
        f"{row['t_rib1']}, "
        f"{row['r_angle']}, "
        f"{row['t_c']}, "
        f"{int(row['N'])}, "
        f"{row['L']}"
        f"]"
    )


    print(
        f"Mean Fn = "
        f"{np.mean(values['Fn']):.6f} N"
    )


    print(
        "Fn      = "
        + np.array2string(
            values["Fn"],
            precision=6,
            separator=", "
        )
    )


    print(
        "delta_x = "
        + np.array2string(
            values["delta_x"],
            precision=6,
            separator=", "
        )
    )


    print(
        "delta_y = "
        + np.array2string(
            values["delta_y"],
            precision=6,
            separator=", "
        )
    )


    print(
        "delta_z = "
        + np.array2string(
            values["delta_z"],
            precision=6,
            separator=", "
        )
    )


print("\n" + "=" * 110)


# ============================================================
# 48. DISPLAY PLOT
# ============================================================

plt.show()
