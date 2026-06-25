import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Render test app running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')
    results = []

    for f in files:
        results.append({
            'filename': f.filename,
            'prediction': 0.5,
            'status': 'test'
        })

    return jsonify(results)
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
