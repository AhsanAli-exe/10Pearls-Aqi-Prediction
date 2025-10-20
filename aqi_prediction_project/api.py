from flask import Flask, jsonify
from predictor import generate_sample_features,predict_aqi

app = Flask(__name__)

@app.route('/predict',methods=['GET'])
def get_prediction():
    predictions = []
    for _ in range(3):
        sample_features = generate_sample_features()
        prediction = predict_aqi(sample_features)
        predictions.append(prediction)
    
    return jsonify({
        'day1_aqi': predictions[0],
        'day2_aqi': predictions[1],
        'day3_aqi': predictions[2],
    })

@app.route('/',methods=['GET'])
def health_check():
    return "Flask API is running!"


app.run(debug=True,port=5000)
