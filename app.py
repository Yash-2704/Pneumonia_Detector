import os
import time
import threading
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model

# Optimize TensorFlow for production deployment
# Disable GPU if not available (prevents CUDA errors)
tf.config.set_visible_devices([], 'GPU')
# Reduce memory usage - disable eager execution optimizations that use extra memory
tf.config.run_functions_eagerly(False)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "pneumonia_resnet50_final.h5")
DEFAULT_IMG_SIZE: Tuple[int, int] = (384, 384)

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me"
CORS(app, resources={r"/predict": {"origins": "*"}})

# Initialize model_ready event (will be set after model loads)
model_ready = threading.Event()

try:
    print("Loading model...")
    # Load model - Keras 3.x will compile on first use if needed
    model = load_model(MODEL_PATH)
    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and input_shape:
        # Functional models can expose a list of shapes
        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]
        if len(input_shape) >= 3:
            IMG_SIZE: Tuple[int, int] = (
                int(input_shape[1]) if input_shape[1] else DEFAULT_IMG_SIZE[0],
                int(input_shape[2]) if input_shape[2] else DEFAULT_IMG_SIZE[1],
            )
        else:
            IMG_SIZE = DEFAULT_IMG_SIZE
    else:
        IMG_SIZE = DEFAULT_IMG_SIZE
    
    # Warm up model in background thread to avoid blocking startup
    # This compiles the model so first request is faster
    def warmup_model():
        """Warm up model in background thread."""
        try:
            print("Warming up model in background (this may take a few minutes)...")
            dummy_input = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0, batch_size=1)
            model_ready.set()
            print("✅ Model warmed up and ready for predictions!")
        except Exception as warmup_exc:
            print(f"⚠️  Model warmup failed ({warmup_exc}), will compile on first request")
            model_ready.set()  # Still mark as ready, will compile on first request
    
    # Start warmup in background thread
    warmup_thread = threading.Thread(target=warmup_model, daemon=True)
    warmup_thread.start()
    print("Model loaded. Warming up in background...")
except OSError as exc:
    raise RuntimeError(f"Unable to load model at {MODEL_PATH}: {exc}") from exc


def prepare_image(file_storage) -> np.ndarray:
    """Replicate the medical preprocessing pipeline used during training."""
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    file_storage.seek(0)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Uploaded file is not a valid image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    if IMG_SIZE:
        image_rgb = cv2.resize(image_rgb, IMG_SIZE)

    image_rgb = image_rgb.astype(np.float32) / 255.0
    return np.expand_dims(image_rgb, axis=0)


def compute_pneumonia_probability(prediction: np.ndarray) -> float:
    """Convert model output to probability (sigmoid output by design)."""
    value = float(np.squeeze(prediction))
    if 0.0 <= value <= 1.0:
        return value
    return float(1 / (1 + np.exp(-value)))


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint - doesn't require model to be ready."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_ready": model_ready.is_set()
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    # Wait for model warmup to complete (with timeout to avoid hanging forever)
    if not model_ready.wait(timeout=600):  # 10 minute max wait
        return jsonify({"error": "Model is still initializing. Please try again in a moment."}), 503

    try:
        start = time.perf_counter()
        processed = prepare_image(file)
        # Use explicit batch_size=1 and steps=1 for memory efficiency
        raw_prediction = model.predict(processed, verbose=0, batch_size=1, steps=1)
        pneumonia_prob = compute_pneumonia_probability(raw_prediction)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:  # pylint: disable=broad-except
        import traceback
        print(f"Prediction error: {exc}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to process image: {exc}"}), 500

    diagnosis = "pneumonia" if pneumonia_prob >= 0.5 else "normal"
    confidence = pneumonia_prob if diagnosis == "pneumonia" else 1 - pneumonia_prob

    return jsonify(
        {
            "diagnosis": diagnosis,
            "probability": pneumonia_prob,
            "confidence": confidence,
            "inference_time_ms": elapsed_ms,
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)