import tensorflow as tf

# !!! REPLACE WITH THE NAME OF YOUR .h5 MODEL FILE !!!
H5_MODEL_PATH = 'plant_disease_model.h5' 
TFLITE_MODEL_PATH = 'model.tflite'

print(f"Loading Keras model from: {H5_MODEL_PATH}")
# Load the Keras .h5 model
model = tf.keras.models.load_model(H5_MODEL_PATH)

# Initialize the TFLite converter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- THIS IS THE CRUCIAL CHANGE FOR COMPATIBILITY ---
# This setting forces the converter to produce a model that is compatible 
# with the standard, built-in operations of older TFLite runtimes.
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS
]

print("Converting model to a compatible TFLite format...")
# Perform the conversion
tflite_model = converter.convert()

# Save the new, compatible .tflite model to disk
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"SUCCESS: Compatible model has been saved to: {TFLITE_MODEL_PATH}")
print("Please upload THIS NEW file to your Hugging Face Space.")