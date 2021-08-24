import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
   
   
    if (output==1):
        return render_template('index.html', prediction_text='possibility of getting cardiac problem,please take precaution')
    
    
    else:
        return render_template('index.html', prediction_text="you have good health, continue your healthy life style")
@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()