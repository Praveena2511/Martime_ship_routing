from flask import Flask, render_template, request
import joblib
import re
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load pre-trained model and scaler
# Load the Keras model
model = load_model(r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\fuel_prediction_model.h5')
# Replace with your model path
scaler = joblib.load(r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/problem')
def problem():
    return render_template('problem.html')

@app.route('/route-planning')
def route_planning():
    return render_template('route_planning.html')

@app.route('/fuel-prediction', methods=['GET', 'POST'])
def fuel_prediction():
    if request.method == 'POST':
        try:
            features = [float(request.form[key]) for key in request.form.keys()]
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            fuel_consumed = round(float(prediction[0][0]), 2)
            return render_template('fuel_prediction.html', fuel_consumed=fuel_consumed)
        except Exception as e:
            return render_template('fuel_prediction.html', error=f"Error: {str(e)}")
    return render_template('fuel_prediction.html')

@app.route('/route-comparison')
def route_comparison():
    return render_template('route_comparison.html')

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)