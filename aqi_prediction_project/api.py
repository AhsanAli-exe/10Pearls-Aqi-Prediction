from flask import Flask, jsonify, request
from predictor import predict_aqi_real_time, get_real_time_features
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_prediction():
    """
    Get AQI prediction for the next 3 days using real-time data
    """
    try:
        predictions = []
        prediction_times = []
        
        # Get current time for reference
        current_time = datetime.now()
        
        # For now, we'll use the same prediction for all 3 days
        # In a more sophisticated system, you'd use different time horizons
        for i in range(3):
            prediction = predict_aqi_real_time()
            
            if prediction is None:
                logger.error(f"Failed to get prediction for day {i+1}")
                return jsonify({
                    'error': 'Failed to generate prediction',
                    'status': 'error'
                }), 500
            
            predictions.append(prediction)
            prediction_times.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        response = {
            'day1_aqi': predictions[0],
            'day2_aqi': predictions[1],
            'day3_aqi': predictions[2],
            'prediction_times': prediction_times,
            'status': 'success',
            'timestamp': current_time.isoformat()
        }
        
        logger.info(f"Successfully generated predictions: {predictions}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/predict/single', methods=['GET'])
def get_single_prediction():
    """
    Get a single AQI prediction using real-time data
    """
    try:
        prediction = predict_aqi_real_time()
        
        if prediction is None:
            return jsonify({
                'error': 'Failed to generate prediction',
                'status': 'error'
            }), 500
        
        response = {
            'aqi': prediction,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_single_prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint with detailed status
    """
    try:
        # Test real-time data fetching
        features = get_real_time_features()
        data_status = "connected" if features is not None else "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'api': 'running',
            'real_time_data': data_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in health_check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API information
    """
    return jsonify({
        'message': 'AQI Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'Get 3-day AQI predictions',
            '/predict/single': 'Get single AQI prediction',
            '/health': 'Health check with detailed status',
            '/': 'API information'
        },
        'status': 'running'
    })

if __name__ == '__main__':
    logger.info("Starting AQI Prediction API...")
    app.run(debug=True, port=5000, host='0.0.0.0')
