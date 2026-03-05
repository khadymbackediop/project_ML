from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("random_forest_diabetes.pkl")
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

@app.route('/', methods=['GET', 'POST'])
def home():
    
    prediction_text = None
    predicted = False

    if request.method == 'POST':
        try:
            if all(request.form.get(col) for col in columns):
                features = [float(request.form[col]) for col in columns]
                prediction = model.predict(np.array([features]))
                
                prediction_text = "🔴 le patient est Diabétique" if prediction[0] == 1 else "🟢 le patient n'est pas diabétique"
                predicted = True
            else:
                
                prediction_text = None
                predicted = False
        except Exception as e:
            prediction_text = f"Erreur : {str(e)}"
            predicted = True
    
    # On renvoie toujours les variables, qu'elles soient None ou remplies
    return render_template('index.html', columns=columns, prediction_text=prediction_text, predicted=predicted)

if __name__ == "__main__":
    app.run(debug=True)