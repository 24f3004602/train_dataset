import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.tflite")
LABEL_PATH = os.path.join(BASE_DIR, "models", "sign_to_prediction_index_map.json")

# Load Labels
with open(LABEL_PATH, 'r') as f:
    # We flip the map so we can look up by Index: { "0": "Apple", "52": "Cry" }
    raw_labels = json.load(f)
    LABEL_MAP = {str(v): k for k, v in raw_labels.items()}

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['landmarks']
        input_data = np.array(data, dtype=np.float32).reshape(1, 543, 3)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index']).copy()
        predictions = output_data.flatten()
        
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        
        # Get Label from our flipped map
        label = LABEL_MAP.get(str(predicted_idx), "Unknown")

        return jsonify({
            "index": predicted_idx,
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)