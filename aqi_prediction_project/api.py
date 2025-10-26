from flask import Flask,jsonify,request
from predictor import predict_aqi,CITIES,generate_features_with_cached_data_and_time_progression
from datetime import datetime,timedelta
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict',methods=['GET','POST'])
def get_prediction():
    city = 'karachi'
    city_info = CITIES[city]
    lat,lon = city_info['lat'],city_info['lon']
    use_historical = False

    try:
        predictions = []
        base_time = datetime.now()
        from predictor import fetch_real_time_weather_data
        current_data = fetch_real_time_weather_data(lat,lon,use_cache=True)

        if current_data is None:
            return jsonify({'error': 'Failed to fetch real-time weather data'}),500

        for i in range(3):
            future_time = base_time + timedelta(days=i)
            features = generate_features_with_cached_data_and_time_progression(
                current_data,future_time,city,use_historical,day_offset=i
            )
            prediction = predict_aqi(features)
            predictions.append(round(prediction,2))

        return jsonify({
            'city': city_info['name'],
            'coordinates': {'lat': lat,'lon': lon},
            'predictions': {
                'day1_aqi': predictions[0],
                'day2_aqi': predictions[1],
                'day3_aqi': predictions[2],
            },
            'timestamp': base_time.isoformat(),
            'data_source': 'current_conditions_only',
            'note': 'Karachi-optimized model using real-time weather data only'
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}),500

@app.route('/cities', methods=['GET'])
def get_available_cities():
    return jsonify({
        'cities': [
            {
                'id': city_id,
                'name': city_info['name'],
                'coordinates': {
                    'lat': city_info['lat'],
                    'lon': city_info['lon']
                }
            }
            for city_id,city_info in CITIES.items()
        ]
    })

@app.route('/historical',methods=['GET'])
def get_historical_data():
    city = request.args.get('city','karachi').lower()
    days = int(request.args.get('days',7))

    if city not in CITIES:
        return jsonify({'error': f'City "{city}" not supported.'}), 400

    try:
        historical_file = os.path.join(os.path.dirname(__file__),'data',f'{city}_weather_1year.csv')

        if not os.path.exists(historical_file):
            return jsonify({'error': f'Historical data not available for {city}'}), 404

        df = pd.read_csv(historical_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = df[df['datetime'] >= cutoff_date].copy()

        if len(recent_data) == 0:
            return jsonify({'error': f'No data available for the last {days} days'}), 404

        if 'aqi' not in recent_data.columns:
            from feature_engineering import calculate_aqi
            recent_data['aqi'] = recent_data.apply(
                lambda row: calculate_aqi(
                    row.get('pm25',50),row.get('pm10',50),
                    row.get('o3',50),row.get('no2',50),
                    row.get('co',500),row.get('so2',20)
                ),axis=1
            )

        recent_data['date'] = recent_data['datetime'].dt.date
        daily_aqi = recent_data.groupby('date')['aqi'].agg(['mean','min','max']).round(2)

        historical_trend = []
        for date,row in daily_aqi.iterrows():
            historical_trend.append({
                'date': date.isoformat(),
                'avg_aqi': row['mean'],
                'min_aqi': row['min'],
                'max_aqi': row['max']
            })

        return jsonify({
            'city': CITIES[city]['name'],
            'days': days,
            'historical_data': historical_trend
        })

    except Exception as e:
        return jsonify({'error': f'Failed to fetch historical data: {str(e)}'}), 500

@app.route('/',methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_cities': len(CITIES),
        'endpoints': {
            'predict': '/predict?city=<city_name>',
            'cities': '/cities',
            'historical': '/historical?city=<city_name>&days=<number>'
        }
    })

@app.route('/weather/current',methods=['GET'])
def get_current_weather():
    city = request.args.get('city','karachi').lower()

    if city not in CITIES:
        return jsonify({'error': f'City "{city}" not supported.'}),400

    city_info = CITIES[city]
    lat,lon = city_info['lat'],city_info['lon']

    try:
        from predictor import fetch_real_time_weather_data
        current_data = fetch_real_time_weather_data(lat,lon)

        if current_data is None:
            return jsonify({'error': 'Failed to fetch current weather data'}),500

        return jsonify({
            'city': city_info['name'],
            'coordinates': {'lat': lat,'lon': lon},
            'current_weather': current_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'Failed to fetch current weather: {str(e)}'}),500

if __name__ == '__main__':
    app.run(debug=True,port=5000)
