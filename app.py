import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow import keras

# -------------------------
# Initialize Flask
# -------------------------
app = Flask(__name__)

# -------------------------
# Load model
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'TCT_verifier_842x595.h5')
# careful handling if model doesn't exist yet
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    app.logger.warning(f"Warning loading model at {model_path}: {e}")
    model = None

# -------------------------
# Helper function
# -------------------------
def predict_image(img: Image.Image):
    if model is None:
        raise Exception("Model not loaded")
        
    img_resized = img.resize((595, 842))
    img_array = keras.preprocessing.image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)[0][0]
    
    if pred >= 0.9:
        status = 'TCT'
    elif pred >= 0.8:
        status = 'Low'
    else:
        status = 'Not TCT'
        
    return float(pred), status

# -------------------------
# API routes
# -------------------------
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Flask app is running"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file')
    results = []

    for f in files:
        try:
            img = Image.open(f).convert('RGB')
            pred, status = predict_image(img)
            results.append({
                "filename": f.filename,
                "prediction": pred,
                "status": status
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "error": str(e)
            })
    
    return jsonify(results)

# -------------------------
# Run server
# -------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
