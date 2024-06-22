import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('vr_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    one = ['rat', 'Rat', 'RAT', 'hydrogel', 'Hydrogel', 'HYDROGEL', 'BSA']
    two = ['Lidocaine', 'human', 'Human', 'HUMAN', 'plastic', 'Plastic', 'PLASTIC']
    three = ['Rhodamine B']
    four = ['Caffeine']
    five = ['Copper ions']
    six = ['GHK peptide']
    int_features = []
    x = [0,66000,234.34,479.02,194.19,63.5,340.38]
    n=1
    for i in request.form.values():
        if i in one:
            int_features.append(1.0)
        elif i in two:
            int_features.append(2.0)
        elif i in three:
            int_features.append(3.0)
        elif i in four:
            int_features.append(4.0)
        elif i in five:
            int_features.append(5.0)
        elif i in six:
            int_features.append(6.0)
        elif n == 3:
            int_features.append(x[int(int_features[0])])
            int_features.append(float(i))
        else:
            int_features.append(float(i))
        n+=1
            
    final_features = np.array(int_features).reshape(1,-1)

    prediction = model.predict(final_features)
    
    if prediction[0] < 0:
        prediction[0] = 0
        
    if prediction[0] > 100:
        prediction[0] = 100
    
    pred_amt = np.round(((int_features[1]*prediction[0])/100), 2)
    
    return render_template('index.html', prediction_text='Predicted permeation amount: {}µg/cm2 ({}%)'.format(pred_amt, np.round(prediction[0], 2)))

if __name__ == "__main__":
    app.run(debug=True)

'''
'bsa', 'Bovine Serum Albumin', 'Bovine serum albumin', 'bovine serum albumin'
'lidocaine',  'LIDOCAINE', 
, 'rhodamine B', 'RHODAMINE B', 'rhodamine b', 'Rhodamine b'
'caffeine', , 'CAFFEINE'
'copper ions',  'COPPER IONS', 'Copper Ions'
, 'ghk peptide', 'GHK PEPTIDE', 'GHK Peptide'
'''