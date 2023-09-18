from flask import Flask, request, jsonify
import numpy as np
import pickle

# Starting FastApi
app = Flask(__name__)

# Loading k-means model
with open("final_KMEANS.pkl", "rb") as model_file:
    kmeans_model = pickle.load(model_file)

class InputData:
    def __init__(self, Length, Recency, Frequency,MONETARY ):
        self.Length = float(Length)
        self.Recency = float(Recency)
        self.Frequency = float(Frequency)
        self.MONETARY = float(MONETARY)

@app.route('/')
def index():
    return 'K-Means Model Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get json data
        data = request.get_json()

        # Create input data object
        input_data = InputData(data['Length'], data['Recency'], data['Frequency'], data['MONETARY'])

        # Use input data for clusturing
        input_features = [input_data.Length, input_data.Recency, input_data.Frequency, input_data.MONETARY]
        predicted_cluster = int(kmeans_model.predict([input_features])[0])

        return jsonify({"predicted_cluster": predicted_cluster})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
