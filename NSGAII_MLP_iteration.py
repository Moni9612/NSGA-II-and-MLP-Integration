# ============================================================
# ITERATION CODE FOR EACH CONSTRAINED NSGA-II + MLP LOOP
#
# IMPORTANT:
# Change ONLY iteration_number before each run
#
# Constraint:
# Mean_Fn >= 15 N
# ============================================================

iteration_number = 2
# ============================================================
# 1. GENERATE 20 NEW CANDIDATES
# ============================================================

new_candidates = generate_new_candidates(
    cumulative_df,
    n_candidates=20
)

new_candidates_df = pd.DataFrame(
    new_candidates,
    columns=geometry_columns
)

new_candidates_df["N"] = (
    new_candidates_df["N"]
    .astype(int)
)

print(
    "Generated candidates:",
    len(new_candidates_df)
)

display(
    new_candidates_df
)


# ============================================================
# 2. EVALUATE NEW CANDIDATES USING MLP
# ============================================================

new_evaluated_rows = []

condition_prediction_rows = []


for candidate_id, row in (
    new_candidates_df.iterrows()
):

    result = evaluate_geometry_with_mlp(
        row[
            geometry_columns
        ].to_numpy(
            dtype=float
        )
    )

    g = result[
        "geometry"
    ]


    # --------------------------------------------------------
    # DESIGN-LEVEL RESULTS
    # --------------------------------------------------------

    new_evaluated_rows.append({

        "t_rib1": g[0],
        "r_angle": g[1],
        "t_c": g[2],
        "N": int(g[3]),
        "L": g[4],

        "CV_D": result["CV_D"],
        "CV_F": result["CV_F"],
        "ZDI": result["ZDI"],

        # --------------------------------------------
        # FORCE CONSTRAINT INFORMATION
        # --------------------------------------------

        "Mean_Fn": result["Mean_Fn"],

        "Feasible": result["Feasible"],

        "source": "MLP",

        "iteration": iteration_number
    })


    # --------------------------------------------------------
    # SAVE ALL 12 OPERATING-CONDITION PREDICTIONS
    # --------------------------------------------------------

    for k, (_, c) in enumerate(
        operating_conditions.iterrows()
    ):

        condition_prediction_rows.append({

            "candidate": candidate_id + 1,

            "t_rib1": g[0],
            "r_angle": g[1],
            "t_c": g[2],
            "N": int(g[3]),
            "L": g[4],

            "position":
                c["position"],

            "orientation":
                c["orientation"],

            "Ux":
                result["delta_x"][k],

            "Uy":
                result["delta_y"][k],

            "Uz":
                result["delta_z"][k],

            "Fn_R":
                result["Fn"][k],

            "source":
                "MLP",

            "iteration":
                iteration_number
        })


# ============================================================
# 3. CREATE ITERATION DATAFRAMES
# ============================================================

iteration_df = pd.DataFrame(
    new_evaluated_rows
)

iteration_condition_df = pd.DataFrame(
    condition_prediction_rows
)


print(
    "\n========================================"
)

print(
    "ITERATION",
    iteration_number,
    "RESULTS"
)

print(
    "========================================"
)

display(
    iteration_df
)


# ============================================================
# 4. FORCE CONSTRAINT SUMMARY
# ============================================================

n_feasible = (
    iteration_df[
        "Feasible"
    ]
    .sum()
)

n_infeasible = (
    len(iteration_df)
    -
    n_feasible
)


print(
    "Feasible new designs:",
    n_feasible
)

print(
    "Infeasible new designs:",
    n_infeasible
)


print(
    "\nMean force range:"
)

print(
    "Minimum Mean Fn:",
    iteration_df[
        "Mean_Fn"
    ].min(),
    "N"
)

print(
    "Maximum Mean Fn:",
    iteration_df[
        "Mean_Fn"
    ].max(),
    "N"
)


# ============================================================
# 5. APPEND TO CUMULATIVE DATASET
# ============================================================

cumulative_df = pd.concat(
    [
        cumulative_df,
        iteration_df
    ],
    ignore_index=True
)


print(
    "\nCumulative design count:",
    len(cumulative_df)
)


print(
    "Total feasible designs:",
    cumulative_df[
        "Feasible"
    ].sum()
)

print(
    "Total infeasible designs:",
    (
        ~cumulative_df[
            "Feasible"
        ]
    ).sum()
)


# ============================================================
# 6. CALCULATE CONSTRAINED PARETO FRONT
# ============================================================

nd_idx = get_nondominated_indices(
    cumulative_df
)


pareto_df = (
    cumulative_df
    .iloc[
        nd_idx
    ]
    .reset_index(
        drop=True
    )
)


print(
    "\n========================================"
)

print(
    "CONSTRAINED PARETO FRONT"
)

print(
    "========================================"
)


print(
    "Current Pareto-set size:",
    len(pareto_df)
)


# ============================================================
# 7. VERIFY FORCE CONSTRAINT
# ============================================================

if len(pareto_df) > 0:

    print(
        "Minimum Pareto Mean Fn:",
        pareto_df[
            "Mean_Fn"
        ].min(),
        "N"
    )

    print(
        "Maximum Pareto Mean Fn:",
        pareto_df[
            "Mean_Fn"
        ].max(),
        "N"
    )

    print(
        "All Pareto designs satisfy Mean Fn >= 15 N:",
        (
            pareto_df[
                "Mean_Fn"
            ]
            >=
            MIN_MEAN_FORCE
        ).all()
    )


display(
    pareto_df
)


# ============================================================
# 8. OPTIONAL: DISPLAY INFEASIBLE DESIGNS FROM THIS ITERATION
# ============================================================

infeasible_iteration_df = (
    iteration_df[
        iteration_df[
            "Feasible"
        ]
        ==
        False
    ]
    .reset_index(
        drop=True
    )
)


print(
    "\nInfeasible designs generated in iteration",
    iteration_number,
    ":",
    len(infeasible_iteration_df)
)


if len(
    infeasible_iteration_df
) > 0:

    display(
        infeasible_iteration_df
    )