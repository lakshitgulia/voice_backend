import keras

print("Loading old model with Keras 3...")

model = keras.models.load_model(
    "models/dashboard_symbol_model.h5",
    compile=False
)

print("Saving compatible model...")

model.save("models/dashboard_symbol_model_fixed.keras")

print("Model conversion complete!")