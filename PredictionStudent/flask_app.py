from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

models = pickle.load(open('models.pkl', 'rb'))
rf_model = models['RandomForest']
lr_model = models['LogisticRegression']
dt_model = models['DecisionTree']

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/Prediction")
def prediksi():
    return render_template('basic_elements.html')

@app.route("/data")
def data():
    data = pd.read_excel('train.xlsx')
    data = data.drop(columns=["IPK "])

    data_dict = data.to_dict(orient='records')
    return render_template('basic-table.html', data_dict=data_dict)

@app.route("/evaluasi")
def evaluasi():
    return render_template('chartjs.html')

@app.route('/Prediction', methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction_rf = rf_model.predict(features)
    prediction_lr = lr_model.predict(features)
    prediction_tree = dt_model.predict(features)

    return render_template("basic_elements.html", prediction_text_rf = "Prediksi menggunakan Random Forest : {} ".format(prediction_rf[0]),
     prediction_text_lr="Prediksi menggunakan Logistic Regression : {}".format(prediction_lr[0]),
     prediction_text_tree="Prediksi menggunakan Decision Tree Classifier : {}".format(prediction_tree[0]))

if __name__ =="__main__":
    app.run(debug=True)