import pandas as pd
from flask import Flask, render_template,request
from flask_bootstrap import Bootstrap
import pickle

app = Flask(__name__)
Bootstrap(app)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET','POST'])
def server():
   if request.method == 'GET':
       return (render_template('index.html'))

   if request.method == 'POST':
        SeniorCitizen= request.form['SeniorCitizen']
        Partner = request.form['Partner']
        Dependants = request.form['Dependants']
        Tenure = request.form['Tenure']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        MonthlyCharges = request.form['MonthlyCharges']

        frame = pd.DataFrame([[SeniorCitizen,Partner,Dependants,Tenure
        ,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,Contract,
        PaperlessBilling,PaymentMethod,MonthlyCharges]],columns=['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges'])

        prediction = model.predict(frame)
        prob = model.predict_proba(frame)[0]
        factor = prob[0] *100
        confidence_factor = str(round(factor, 1)) +'%'
        if prediction == 0:
            result = 'YES'
        else:
            result = 'NO'

        print(prediction)
        return (render_template('index.html',result=result, confidence_factor=confidence_factor)) 

       














if __name__ == '__main__':
    app.run(debug=True)

    


