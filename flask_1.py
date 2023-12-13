import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import os

app = Flask(__name__)

MODEL_VERSION = 'RFC.pkl'
model_path = os.path.join(os.getcwd(), 'model_assets', MODEL_VERSION)
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index1.html') 

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if output == 0:
        output ='Laying'
    elif output == 1:
        output ='Sitting'
    elif output ==2:
        output ='Standing'
    else:
        output ='Walking'
        
    print(prediction)
    
    print(output)
            
    

    return render_template('index1.html', prediction_text='HUMAN : {}'.format(output))

if __name__ == "__main__":
    app.run(host="localhost", port=6002)