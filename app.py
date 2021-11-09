from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('adult_income81.pkl')

@app.route('/')
def first():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

    cols = ['age', 'education', 'marital-status', 'sex', 'hours-per-week',
       'country', 'gain/loss', 'Local-gov', 'Private', 'Self-emp-inc',
       'Self-emp-not-inc', 'State-gov', 'No-Income', 'Armed-Forces',
       'Craft-repair', 'Exec-managerial', 'Farming-fishing',
       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
       'Priv-house-serv', 'Private', 'Prof-specialty', 'Protective-serv',
       'Sales', 'Tech-support', 'Transport-moving', 'Asian-Pac-Islander',
       'Black', ' Other', 'White']

    dct = {'Male': 1, 'Female': 0, 'Never-married': 0, 'Divorced': 0, 'Widowed': 0, 'Married': 1,
    'Primary-School': 1, 'High-School': 2, 'Some-college': 3, 'Bachelors': 4, ' HS-grad': 5,
    'Prof-school': 6, 'Associate': 7, 'Masters': 8, 'Doctorate': 9}

    age = request.form["age"]
    education = request.form["education"]
    marital_Status = request.form["marital-Status"]
    gender = request.form["gender"]
    country = request.form["country"]
    hoursPerWeek = request.form["hoursPerWeek"]
    gainOrLoss = request.form["gainOrLoss"]
    workClass = request.form["workClass"]
    occupation = request.form["occupation"]
    race = request.form["race"]

    value = [0 for i in range(len(cols))]

    value[0] = np.log(int(age))
    value[1] = dct[education]
    value[2] = dct[marital_Status]
    value[3] = dct[gender]
    value[4] = int(hoursPerWeek)
    if country == 'United-States':
        value[5] = 1
    value[6] = np.log1p(int(gainOrLoss))
    if workClass in cols:
        value[cols.index(workClass)] = 1
    if occupation in cols:
        value[cols.index(occupation)] = 1
    if race in cols:
        value[cols.index(race)] = 1

    print(value)
    prediction = model.predict([value])

    output = ''
    if prediction[0] == 0:
        output = '<=50K'
    else:
        output = '>50K'

    return render_template('index.html',prediction_text="Your Income is {}".format(output))

if __name__ == '__main__':
    app.run(debug = True)
