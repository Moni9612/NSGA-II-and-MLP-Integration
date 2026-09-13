early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=40,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(

    X_train_scaled,
    Y_train_scaled,

    validation_data=(
        X_val_scaled,
        Y_val_scaled
    ),

    epochs=500,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)