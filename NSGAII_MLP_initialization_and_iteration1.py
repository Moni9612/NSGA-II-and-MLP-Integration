# ============================================================
# FIN-RAY CONSTRAINED MULTI-OBJECTIVE OPTIMIZATION
#
# Objectives:
#   1. Minimize CV_D
#   2. Minimize CV_F
#   3. Minimize ZDI
#
# Constraint:
#   Mean normal force >= 15 N
#
#   μ_F >= 15 N
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow import keras

from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from google.colab import files


# ============================================================
# 2. FORCE CONSTRAINT
# ============================================================

# NEW: FORCE CONSTRAINT

MIN_MEAN_FORCE = 15.0   # N

print(
    "Minimum allowable mean grasp force:",
    MIN_MEAN_FORCE,
    "N"
)


# ============================================================
# 3. LOAD MLP MODEL AND SCALERS
# ============================================================

uploaded = files.upload()

model = keras.models.load_model(
    "finray_real_mlp_improved.keras"
)

X_scaler = joblib.load(
    "finray_X_scaler_new_improved.pkl"
)

Y_scaler = joblib.load(
    "finray_Y_scaler_new_improved.pkl"
)

print(
    "MLP and scalers loaded successfully."
)


# ============================================================
# 4. LOAD COMSOL / MLP DATASET
# ============================================================

uploaded = files.upload()

df = pd.read_csv(
    "MLP_NSGA_Dataset.csv"
)

required_columns = [
    "t_rib1",
    "r_angle",
    "t_c",
    "N",
    "L",
    "position",
    "orientation",
    "Ux",
    "Uy",
    "Uz",
    "Fn_R"
]

df = (
    df[required_columns]
    .dropna()
    .reset_index(drop=True)
)

print(
    "Cleaned dataset shape:",
    df.shape
)


# ============================================================
# 5. COLUMN DEFINITIONS
# ============================================================

geometry_columns = [
    "t_rib1",
    "r_angle",
    "t_c",
    "N",
    "L"
]

input_columns = [
    "t_rib1",
    "r_angle",
    "t_c",
    "N",
    "L",
    "position",
    "orientation"
]

output_columns = [
    "Ux",
    "Uy",
    "Uz",
    "Fn_R"
]

objective_columns = [
    "CV_D",
    "CV_F",
    "ZDI"
]


# ============================================================
# 6. OBJECTIVE CALCULATION
# ============================================================

def calculate_objectives_from_outputs(
    delta_x,
    delta_y,
    delta_z,
    Fn
):

    # --------------------------------------------------------
    # Objective 1: CV_D
    # --------------------------------------------------------

    D = (
        np.abs(delta_x)
        *
        np.abs(delta_y)
    )

    mu_D = np.mean(D)

    sigma_D = np.std(
        D,
        ddof=0
    )

    CV_D = (
        sigma_D
        /
        mu_D
    )


    # --------------------------------------------------------
    # Objective 2: CV_F
    # --------------------------------------------------------

    mu_F = np.mean(Fn)

    sigma_F = np.std(
        Fn,
        ddof=0
    )

    CV_F = (
        sigma_F
        /
        mu_F
    )


    # --------------------------------------------------------
    # Objective 3: ZDI
    # --------------------------------------------------------

    abs_delta_z = np.abs(
        delta_z
    )

    mu_Z = np.mean(
        abs_delta_z
    )

    sigma_Z = np.std(
        abs_delta_z,
        ddof=0
    )

    ZDI = (
        mu_Z
        *
        sigma_Z
    )


    # --------------------------------------------------------
    # Return mean force as well
    # --------------------------------------------------------

    return (
        CV_D,
        CV_F,
        ZDI,
        mu_F
    )


# ============================================================
# 7. CREATE INITIAL DESIGN-LEVEL DATASET
# ============================================================

initial_rows = []

for geometry_values, group in df.groupby(
    geometry_columns
):

    delta_x = group[
        "Ux"
    ].to_numpy()

    delta_y = group[
        "Uy"
    ].to_numpy()

    delta_z = group[
        "Uz"
    ].to_numpy()

    Fn = group[
        "Fn_R"
    ].to_numpy()


    (
        CV_D,
        CV_F,
        ZDI,
        mean_Fn
    ) = calculate_objectives_from_outputs(
        delta_x,
        delta_y,
        delta_z,
        Fn
    )


    row = dict(
        zip(
            geometry_columns,
            geometry_values
        )
    )


    row[
        "CV_D"
    ] = CV_D

    row[
        "CV_F"
    ] = CV_F

    row[
        "ZDI"
    ] = ZDI


    # --------------------------------------------------------
    # NEW: Mean grasp force
    # --------------------------------------------------------

    row[
        "Mean_Fn"
    ] = mean_Fn


    # --------------------------------------------------------
    # NEW: Feasibility
    # --------------------------------------------------------

    row[
        "Feasible"
    ] = (
        mean_Fn
        >=
        MIN_MEAN_FORCE
    )


    row[
        "source"
    ] = "COMSOL"

    row[
        "iteration"
    ] = 0


    initial_rows.append(
        row
    )


cumulative_df = pd.DataFrame(
    initial_rows
)


print(
    "Current number of geometries:",
    len(cumulative_df)
)

print(
    "Feasible designs (Mean Fn >= 15 N):",
    cumulative_df[
        "Feasible"
    ].sum()
)

print(
    "Infeasible designs (Mean Fn < 15 N):",
    (
        ~cumulative_df[
            "Feasible"
        ]
    ).sum()
)


display(
    cumulative_df.head()
)


# ============================================================
# 8. CHECK NUMBER OF CONDITIONS PER GEOMETRY
# ============================================================

geometry_counts = (
    df.groupby(
        geometry_columns
    )
    .size()
)

print(
    geometry_counts.value_counts()
)


# ============================================================
# 9. ALLOWED DESIGN VARIABLE VALUES
# ============================================================

tr_values = np.arange(
    1.5,
    5.0 + 0.001,
    0.25
)

alpha_values = np.arange(
    0,
    38 + 1,
    1
)

tw_values = np.arange(
    1.5,
    3.5 + 0.001,
    0.25
)

N_values = np.arange(
    1,
    8 + 1,
    1
)

L_values = np.arange(
    35,
    70 + 1,
    1
)


print(
    "tr:",
    tr_values
)

print(
    "alpha:",
    alpha_values
)

print(
    "tw:",
    tw_values
)

print(
    "N:",
    N_values
)

print(
    "L:",
    L_values
)


# ============================================================
# 10. SNAP GEOMETRY TO ALLOWED DESIGN SPACE
# ============================================================

def nearest_allowed(
    value,
    allowed_values
):

    index = np.argmin(
        np.abs(
            allowed_values
            -
            value
        )
    )

    return allowed_values[
        index
    ]


def snap_geometry(
    geometry
):

    tr, alpha, tw, N, L = geometry


    tr = nearest_allowed(
        tr,
        tr_values
    )

    alpha = nearest_allowed(
        alpha,
        alpha_values
    )

    tw = nearest_allowed(
        tw,
        tw_values
    )

    N = nearest_allowed(
        N,
        N_values
    )

    L = nearest_allowed(
        L,
        L_values
    )


    return np.array([
        tr,
        alpha,
        tw,
        int(N),
        L
    ])


# ============================================================
# 11. OPERATING CONDITIONS
# ============================================================

operating_conditions = (
    df[
        [
            "position",
            "orientation"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "position",
            "orientation"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "Operating conditions:",
    len(
        operating_conditions
    )
)

display(
    operating_conditions
)


# ============================================================
# 12. EVALUATE ONE GEOMETRY USING MLP
# ============================================================

def evaluate_geometry_with_mlp(
    geometry
):

    geometry = snap_geometry(
        geometry
    )

    tr, alpha, tw, N, L = geometry

    rows = []


    for _, condition in (
        operating_conditions.iterrows()
    ):

        rows.append([
            tr,
            alpha,
            tw,
            N,
            L,
            condition[
                "position"
            ],
            condition[
                "orientation"
            ]
        ])


    mlp_input = pd.DataFrame(
        rows,
        columns=input_columns
    )


    X_scaled = (
        X_scaler.transform(
            mlp_input
        )
    )


    Y_scaled = model.predict(
        X_scaled,
        verbose=0
    )


    Y_pred = (
        Y_scaler.inverse_transform(
            Y_scaled
        )
    )


    delta_x = Y_pred[
        :,
        0
    ]

    delta_y = Y_pred[
        :,
        1
    ]

    delta_z = Y_pred[
        :,
        2
    ]

    Fn = Y_pred[
        :,
        3
    ]


    (
        CV_D,
        CV_F,
        ZDI,
        mean_Fn
    ) = calculate_objectives_from_outputs(
        delta_x,
        delta_y,
        delta_z,
        Fn
    )


    # --------------------------------------------------------
    # NEW: determine force feasibility
    # --------------------------------------------------------

    feasible = (
        mean_Fn
        >=
        MIN_MEAN_FORCE
    )


    return {

        "geometry":
            geometry,

        "delta_x":
            delta_x,

        "delta_y":
            delta_y,

        "delta_z":
            delta_z,

        "Fn":
            Fn,

        "CV_D":
            CV_D,

        "CV_F":
            CV_F,

        "ZDI":
            ZDI,

        # NEW
        "Mean_Fn":
            mean_Fn,

        # NEW
        "Feasible":
            feasible
    }


# ============================================================
# 13. TEST MLP EVALUATION
# ============================================================

test_geometry = [
    2.5,
    20,
    2.5,
    5,
    50
]


test_result = (
    evaluate_geometry_with_mlp(
        test_geometry
    )
)


print(
    "Geometry:",
    test_result[
        "geometry"
    ]
)

print(
    "CV_D:",
    test_result[
        "CV_D"
    ]
)

print(
    "CV_F:",
    test_result[
        "CV_F"
    ]
)

print(
    "ZDI:",
    test_result[
        "ZDI"
    ]
)

print(
    "Mean Fn:",
    test_result[
        "Mean_Fn"
    ],
    "N"
)

print(
    "Feasible:",
    test_result[
        "Feasible"
    ]
)


# ============================================================
# 14. CONSTRAINED NON-DOMINATED SORTING
# ============================================================

def get_nondominated_indices(
    dataframe
):

    """
    Returns the non-dominated designs ONLY among
    designs satisfying:

        Mean_Fn >= MIN_MEAN_FORCE

    Therefore every design on the returned Pareto front
    automatically satisfies the grasp-force constraint.
    """


    # --------------------------------------------------------
    # NEW: Find feasible designs
    # --------------------------------------------------------

    feasible_mask = (
        dataframe[
            "Mean_Fn"
        ]
        >=
        MIN_MEAN_FORCE
    )


    feasible_positions = np.flatnonzero(
        feasible_mask.to_numpy()
    )


    # Safety check
    if len(
        feasible_positions
    ) == 0:

        print(
            "WARNING: No feasible designs "
            "satisfy Mean_Fn >=",
            MIN_MEAN_FORCE,
            "N"
        )

        return np.array(
            [],
            dtype=int
        )


    # --------------------------------------------------------
    # Pareto sorting ONLY over feasible population
    # --------------------------------------------------------

    feasible_df = (
        dataframe.iloc[
            feasible_positions
        ]
    )


    F = feasible_df[
        objective_columns
    ].to_numpy()


    local_nd_indices = (
        NonDominatedSorting()
        .do(
            F,
            only_non_dominated_front=True
        )
    )


    # Convert local positions back to cumulative_df positions
    nd_indices = (
        feasible_positions[
            local_nd_indices
        ]
    )


    return nd_indices


# ============================================================
# 15. INITIAL CONSTRAINED PARETO FRONT
# ============================================================

nd_indices = (
    get_nondominated_indices(
        cumulative_df
    )
)


pareto_df = (
    cumulative_df
    .iloc[
        nd_indices
    ]
    .reset_index(
        drop=True
    )
)


print(
    "Feasible non-dominated designs:",
    len(
        pareto_df
    )
)

print(
    "Minimum Mean_Fn on Pareto front:",
    pareto_df[
        "Mean_Fn"
    ].min()
    if len(
        pareto_df
    ) > 0
    else np.nan
)


display(
    pareto_df
)


# ============================================================
# 16. INITIAL PARENT POOL
# ============================================================

parent_pool = (
    pareto_df[
        geometry_columns
    ]
    .to_numpy(
        dtype=float
    )
)


print(
    "Parent pool size:",
    len(
        parent_pool
    )
)


# ============================================================
# 17. RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    42
)


# ============================================================
# 18. CROSSOVER + MUTATION
# ============================================================

def generate_child(
    parent1,
    parent2,
    mutation_probability=0.20
):

    # --------------------------------------------------------
    # Random blend crossover
    # --------------------------------------------------------

    alpha = rng.random(
        5
    )


    child = (
        alpha
        *
        parent1
        +
        (
            1
            -
            alpha
        )
        *
        parent2
    )


    # --------------------------------------------------------
    # Mutation
    # --------------------------------------------------------

    ranges = np.array([
        5.0 - 1.5,
        38.0 - 0.0,
        3.5 - 1.5,
        8.0 - 1.0,
        70.0 - 35.0
    ])


    for j in range(
        5
    ):

        if (
            rng.random()
            <
            mutation_probability
        ):

            child[
                j
            ] += (
                rng.normal(
                    0,
                    0.05
                    *
                    ranges[
                        j
                    ]
                )
            )


    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    lower = np.array([
        1.5,
        0,
        1.5,
        1,
        35
    ])


    upper = np.array([
        5.0,
        38,
        3.5,
        8,
        70
    ])


    child = np.clip(
        child,
        lower,
        upper
    )


    return snap_geometry(
        child
    )


# ============================================================
# 19. DUPLICATE CHECK
# ============================================================

def geometry_exists(
    geometry,
    dataframe
):

    geometry = snap_geometry(
        geometry
    )


    mask = (

        np.isclose(
            dataframe[
                "t_rib1"
            ],
            geometry[
                0
            ]
        )

        &

        np.isclose(
            dataframe[
                "r_angle"
            ],
            geometry[
                1
            ]
        )

        &

        np.isclose(
            dataframe[
                "t_c"
            ],
            geometry[
                2
            ]
        )

        &

        np.isclose(
            dataframe[
                "N"
            ],
            geometry[
                3
            ]
        )

        &

        np.isclose(
            dataframe[
                "L"
            ],
            geometry[
                4
            ]
        )
    )


    return mask.any()


# ============================================================
# 20. GENERATE NEW CANDIDATES
# ============================================================

def generate_new_candidates(
    cumulative_df,
    n_candidates=20
):

    # --------------------------------------------------------
    # Get constrained Pareto front
    # --------------------------------------------------------

    nd_indices = (
        get_nondominated_indices(
            cumulative_df
        )
    )


    # --------------------------------------------------------
    # First preference:
    # Feasible Pareto-optimal parents
    # --------------------------------------------------------

    if len(
        nd_indices
    ) >= 2:

        parents = (
            cumulative_df
            .iloc[
                nd_indices
            ][
                geometry_columns
            ]
            .to_numpy(
                dtype=float
            )
        )


    else:

        # ----------------------------------------------------
        # Safety fallback:
        # use all feasible designs
        # ----------------------------------------------------

        feasible_df = (
            cumulative_df[
                cumulative_df[
                    "Mean_Fn"
                ]
                >=
                MIN_MEAN_FORCE
            ]
        )


        if len(
            feasible_df
        ) >= 2:

            parents = (
                feasible_df[
                    geometry_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


        else:

            # ------------------------------------------------
            # Extremely unlikely fallback:
            # choose highest-force designs available.
            # This helps move population toward feasibility.
            # ------------------------------------------------

            print(
                "WARNING: Fewer than two feasible designs."
            )

            print(
                "Using highest-mean-force designs "
                "as temporary parent pool."
            )


            temporary_parent_df = (
                cumulative_df
                .sort_values(
                    "Mean_Fn",
                    ascending=False
                )
                .head(
                    max(
                        2,
                        min(
                            10,
                            len(
                                cumulative_df
                            )
                        )
                    )
                )
            )


            parents = (
                temporary_parent_df[
                    geometry_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


    print(
        "Parent pool used for candidate generation:",
        len(
            parents
        )
    )


    candidates = []

    attempts = 0

    max_attempts = 10000


    while (

        len(
            candidates
        )
        <
        n_candidates

        and

        attempts
        <
        max_attempts

    ):

        attempts += 1


        ids = rng.choice(
            len(
                parents
            ),
            size=2,
            replace=False
        )


        p1 = parents[
            ids[
                0
            ]
        ]

        p2 = parents[
            ids[
                1
            ]
        ]


        child = generate_child(
            p1,
            p2
        )


        # ----------------------------------------------------
        # Check against cumulative population
        # ----------------------------------------------------

        if geometry_exists(
            child,
            cumulative_df
        ):

            continue


        # ----------------------------------------------------
        # Check against newly generated candidates
        # ----------------------------------------------------

        duplicate_new = any(

            np.allclose(
                child,
                existing_child
            )

            for existing_child
            in candidates
        )


        if duplicate_new:

            continue


        candidates.append(
            child
        )


    if (
        len(
            candidates
        )
        <
        n_candidates
    ):

        print(
            "Warning:",
            "Only generated",
            len(
                candidates
            ),
            "unique candidates."
        )


    return np.array(
        candidates
    )


# ============================================================
# 21. GENERATE NEW CANDIDATE DESIGNS
# ============================================================

new_candidates = (
    generate_new_candidates(
        cumulative_df,
        n_candidates=20
    )
)


new_candidates_df = pd.DataFrame(
    new_candidates,
    columns=geometry_columns
)


new_candidates_df[
    "N"
] = (
    new_candidates_df[
        "N"
    ]
    .astype(
        int
    )
)


print(
    "Generated candidates:",
    len(
        new_candidates_df
    )
)


display(
    new_candidates_df
)


# ============================================================
# 22. EVALUATE NEW DESIGNS USING MLP
# ============================================================

# Initial COMSOL population = iteration 0
# First MLP-generated population = iteration 1

iteration_number = 1


new_evaluated_rows = []


for _, row in (
    new_candidates_df.iterrows()
):


    geometry = row[
        geometry_columns
    ].to_numpy(
        dtype=float
    )


    result = (
        evaluate_geometry_with_mlp(
            geometry
        )
    )


    g = result[
        "geometry"
    ]


    new_evaluated_rows.append({

        "t_rib1":
            g[
                0
            ],

        "r_angle":
            g[
                1
            ],

        "t_c":
            g[
                2
            ],

        "N":
            int(
                g[
                    3
                ]
            ),

        "L":
            g[
                4
            ],


        "CV_D":
            result[
                "CV_D"
            ],

        "CV_F":
            result[
                "CV_F"
            ],

        "ZDI":
            result[
                "ZDI"
            ],


        # ----------------------------------------------------
        # NEW: mean force
        # ----------------------------------------------------

        "Mean_Fn":
            result[
                "Mean_Fn"
            ],


        # ----------------------------------------------------
        # NEW: feasibility
        # ----------------------------------------------------

        "Feasible":
            result[
                "Feasible"
            ],


        "source":
            "MLP",

        "iteration":
            iteration_number
    })


iteration_df = pd.DataFrame(
    new_evaluated_rows
)


print(
    "\nNew iteration results:"
)

display(
    iteration_df
)


# ============================================================
# 23. ITERATION FORCE CONSTRAINT SUMMARY
# ============================================================

new_feasible = (
    iteration_df[
        "Feasible"
    ].sum()
)


new_infeasible = (
    len(
        iteration_df
    )
    -
    new_feasible
)


print(
    "Feasible new designs:",
    new_feasible
)

print(
    "Infeasible new designs:",
    new_infeasible
)


print(
    "Mean force range in new designs:"
)

print(
    "Minimum:",
    iteration_df[
        "Mean_Fn"
    ].min(),
    "N"
)

print(
    "Maximum:",
    iteration_df[
        "Mean_Fn"
    ].max(),
    "N"
)


# ============================================================
# 24. APPEND NEW DESIGNS TO CUMULATIVE DATASET
# ============================================================

cumulative_df = pd.concat(
    [
        cumulative_df,
        iteration_df
    ],
    ignore_index=True
)


print(
    "Cumulative design count:",
    len(
        cumulative_df
    )
)


# ============================================================
# 25. UPDATE CONSTRAINED PARETO FRONT
# ============================================================

nd_indices = (
    get_nondominated_indices(
        cumulative_df
    )
)


pareto_df = (
    cumulative_df
    .iloc[
        nd_indices
    ]
    .reset_index(
        drop=True
    )
)


print(
    "Pareto designs after iteration",
    iteration_number,
    ":",
    len(
        pareto_df
    )
)


# ============================================================
# 26. VERIFY FORCE CONSTRAINT
# ============================================================

print(
    "\n========================================"
)

print(
    "CONSTRAINED PARETO FRONT CHECK"
)

print(
    "========================================"
)


print(
    "Required mean force >=",
    MIN_MEAN_FORCE,
    "N"
)


if len(
    pareto_df
) > 0:

    print(
        "Lowest Pareto mean force:",
        pareto_df[
            "Mean_Fn"
        ].min(),
        "N"
    )

    print(
        "Highest Pareto mean force:",
        pareto_df[
            "Mean_Fn"
        ].max(),
        "N"
    )


    constraint_satisfied = (

        pareto_df[
            "Mean_Fn"
        ]
        >=
        MIN_MEAN_FORCE

    ).all()


    print(
        "All Pareto designs satisfy force constraint:",
        constraint_satisfied
    )


else:

    print(
        "No feasible Pareto designs found."
    )


print(
    "========================================"
)


display(
    pareto_df
)