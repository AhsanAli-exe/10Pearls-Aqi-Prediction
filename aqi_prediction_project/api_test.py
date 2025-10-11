import requests
import json

def get_aqi_data(city_name):
    url = f"https://api.waqi.info/feed/{city_name}/?token=demo"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    

city = "karachi"
aqi_data = get_aqi_data(city)
aqi_value = aqi_data['data']['aqi']
city_name = aqi_data['data']['city']['name']
print(f"The AQI value for {city_name} is {aqi_value}")

pollutants = aqi_data['data']['iaqi']
for pollutant, value in pollutants.items():
    print(f"{pollutant}: {value['v']}")
    
    