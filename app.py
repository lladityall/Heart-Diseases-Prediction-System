from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model.pkl", 'rb') as file:
    trained_models = pickle.load(file)


def predict_heart_disease(features, model_name):
    model_to_use = trained_models[model_name]
    prediction = model_to_use.predict(features)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trtbps = int(request.form['trtbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalachh = int(request.form['thalachh'])
    exng = int(request.form['exng'])
    oldpeak = float(request.form['oldpeak'])
    slp = int(request.form['slp'])
    caa = int(request.form['caa'])
    thall = int(request.form['thall'])

    features = [[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]

    prediction = predict_heart_disease(features, "Random Forest")

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
