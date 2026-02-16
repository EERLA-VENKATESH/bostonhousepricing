import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# ==========================
# Load Trained Model & Scaler
# ==========================
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# ==========================
# Home Route
# ==========================
@app.route('/')
def home():
    return render_template('home.html')


# ==========================
# API Prediction (JSON Input)
# ==========================
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']

    # Convert JSON values to numpy array
    input_data = np.array(list(data.values())).reshape(1, -1)

    # Scale
    scaled_data = scaler.transform(input_data)

    # Predict
    output = model.predict(scaled_data)

    return jsonify({
        "prediction": float(output[0])
    })


# ==========================
# Web Form Prediction
# ==========================
@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    data = [float(x) for x in request.form.values()]

    # Convert to numpy array & reshape
    input_data = np.array(data).reshape(1, -1)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    output = model.predict(scaled_data)

    return render_template(
        "home.html",
        prediction_text="The House price prediction is {:.2f}".format(output[0])
    )


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
