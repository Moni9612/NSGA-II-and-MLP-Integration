# MLP Evaluation

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Fin-Ray MLP Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

Y_pred_scaled = model.predict(
    X_test_scaled
)

Y_pred = Y_scaler.inverse_transform(
    Y_pred_scaled
)

Y_test_actual = Y_test.to_numpy()

print(
    "Test predictions completed."
)

results = []

for i, output_name in enumerate(output_columns):

    y_true = Y_test_actual[:, i]
    y_pred = Y_pred[:, i]

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    results.append({

        "Output": output_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })

metrics_df = pd.DataFrame(
    results
)

display(metrics_df)

for i, output_name in enumerate(output_columns):

    y_true = Y_test_actual[:, i]

    y_pred = Y_pred[:, i]

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.7
    )

    minimum = min(
        y_true.min(),
        y_pred.min()
    )

    maximum = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )

    plt.xlabel(
        f"COMSOL {output_name}"
    )

    plt.ylabel(
        f"MLP Predicted {output_name}"
    )

    plt.title(
        f"COMSOL vs MLP: {output_name}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    comparison_df = (
    X_test
    .reset_index(drop=True)
    .copy()
)

for i, output_name in enumerate(output_columns):

    comparison_df[
        f"{output_name}_COMSOL"
    ] = Y_test_actual[:, i]

    comparison_df[
        f"{output_name}_MLP"
    ] = Y_pred[:, i]

    comparison_df[
        f"{output_name}_Error"
    ] = (
        Y_pred[:, i]
        -
        Y_test_actual[:, i]
    )

display(
    comparison_df.head(20)
)

fn_index = output_columns.index(
    "F_res"
)

F_res_true = Y_test_actual[
    :, fn_index
]

F_res_pred = Y_pred[
    :, fn_index
]

F_res_percentage_error = (
    np.abs(
        F_res_pred - F_res_true
    )
    /
    np.abs(F_res_true)
) * 100

print(
    "Mean F_res percentage error:",
    np.mean(F_res_percentage_error),
    "%"
)

print(
    "Median F_res percentage error:",
    np.median(F_res_percentage_error),
    "%"
)

print(
    "Maximum F_res percentage error:",
    np.max(F_res_percentage_error),
    "%"
)

#Saving the model parameters

derived_results = []

derived_pairs = [

    (
        "delta_x",
        delta_x_true,
        delta_x_pred
    ),
    (
        "delta_y",
        delta_y_true,
        delta_y_pred
    ),
    (
        "delta_z",
        delta_z_true,
        delta_z_pred
    ),
    (
        "F_res",
        F_res_true,
        F_res_pred
    )
]

for name, y_true, y_pred in derived_pairs:

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    derived_results.append({

        "Quantity": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })

derived_metrics_df = pd.DataFrame(
    derived_results
)

display(
    derived_metrics_df
)

model.save(
    "finray_real_mlp_improved.keras"
)

joblib.dump(
    X_scaler,
    "finray_X_scaler_new_improved.pkl"
)

joblib.dump(
    Y_scaler,
    "finray_Y_scaler_new_improved.pkl"
)

print(
    "Real Fin-Ray MLP and scalers saved."
)
