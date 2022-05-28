from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
import  pickle

app = Flask(__name__)

@app.route('/', methods = ['GET'])
@cross_origin()

def homepage():
    return render_template("index.html")


@app.route('/predict', methods = ['GET', 'POST'])
@cross_origin()

def prediction():
    if request.method == 'POST' :
        try:
            CRIM = float(request.form['CRIM'])
            ZN   = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B  = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])

            file = 'LR_model_assignment.pickle'
            model_load = pickle.load(open(file,'rb'))
            predictionn = model_load.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
            return render_template('results.html', predictionn = np.round(predictionn[0]))
        except Exception as e:
            print('the error is : ', e)
            print('something went wrong while processing the request')

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug = True)


