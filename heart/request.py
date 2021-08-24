import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Insulin':0, 'BMI':33.6, 'DiabetesPedigreeFunction':0.627,'Age':50})

print(r.json())