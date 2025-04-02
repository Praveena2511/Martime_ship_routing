from flask import Flask, render_template, request
import joblib  # Assuming the model and scaler are saved using joblib
import re
from tensorflow.keras.models import load_model

# Load pre-trained model and scaler
# Load the Keras model
model = load_model(r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\fuel_prediction_model.h5')
# Replace with your model path
scaler = joblib.load(r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\scaler.pkl')  # Replace with your scaler path

app = Flask(__name__)

@app.route('/')
def index():
    # Read and parse the contents of palma.txt
    with open(r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\out\Res_Palma_Barna.txt', 'r') as file:
        content = file.read()

    # Helper function to extract values using regex
    def extract_value(pattern, content):
        match = re.search(pattern, content)
        return match.group(1) if match else 'Not Found'

    # Extract key details from the file
    initial_velocity = extract_value(r'Initial Velocity in knots = (\d+\.\d+)', content)
    wen_form = extract_value(r'WEN_form = (\d+)', content)
    departure_node = extract_value(r'Departure:  Node = (\d+)', content)
    departure_coords_match = re.search(r'coordinates  \((\d+\.\d+),(\d+\.\d+)\)', content)
    departure_coords = f"({departure_coords_match.group(1)}, {departure_coords_match.group(2)})" if departure_coords_match else 'Not Found'
    departure_time = extract_value(r'Departure time \(day-month-year hour:min\): (\d+-\d+-\d+ \d+:\d+)', content)
    arrival_node = extract_value(r'Arrival:     Node = (\d+)', content)
    arrival_coords_match = re.search(r'coordinates  \((\d+\.\d+),(\d+\.\d+)\)', content)
    arrival_coords = f"({arrival_coords_match.group(1)}, {arrival_coords_match.group(2)})" if arrival_coords_match else 'Not Found'
    geodetic_distance = extract_value(r'Geodetic Distance \(in milles\): (\d+\.\d+)', content)

    sailed_hours_optimized = extract_value(r'Route Optimized:\s+(\d+\.\d+)', content)
    sailed_miles_optimized = extract_value(r'Route Optimized:\s+\d+\.\d+\s+(\d+\.\d+)', content)
    sailed_hours_min_distance = extract_value(r'Route Minimum Distance:\s+(\d+\.\d+)', content)
    sailed_miles_min_distance = extract_value(r'Route Minimum Distance:\s+\d+\.\d+\s+(\d+\.\d+)', content)
    sailed_hours_min_distance_no_waves = extract_value(r'Route Minimum Distance \(without waves\):\s+(\d+\.\d+)', content)
    sailed_miles_min_distance_no_waves = extract_value(r'Route Minimum Distance \(without waves\):\s+\d+\.\d+\s+(\d+\.\d+)', content)

    # Render index.html with extracted data
    return render_template('main.html',
                           initial_velocity=initial_velocity,
                           wen_form=wen_form,
                           departure_node=departure_node,
                           departure_coords=departure_coords,
                           departure_time=departure_time,
                           arrival_node=arrival_node,
                           arrival_coords=arrival_coords,
                           geodetic_distance=geodetic_distance,
                           sailed_hours_optimized=sailed_hours_optimized,
                           sailed_miles_optimized=sailed_miles_optimized,
                           sailed_hours_min_distance=sailed_hours_min_distance,
                           sailed_miles_min_distance=sailed_miles_min_distance,
                           sailed_hours_min_distance_no_waves=sailed_hours_min_distance_no_waves,
                           sailed_miles_min_distance_no_waves=sailed_miles_min_distance_no_waves)


@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    if request.method == 'POST':
        # Get user inputs from the form
        try:
            #form_data={key:request.form[key] for key in request.form.keys()}
            form_data = request.form
            features = [
                float(request.form['speed_over_ground']),
                float(request.form['speed_through_water']),
                float(request.form['course']),
                float(request.form['heading']),
                float(request.form['main_engine_rpm']),
                float(request.form['main_engine_power']),
                float(request.form['main_engine_torque']),
                float(request.form['mean_draft']),
                float(request.form['draft_aft']),
                float(request.form['draft_fwd']),
                float(request.form['rate_of_turn']),
                float(request.form['wave_height']),
                float(request.form['wind_speed']),
                float(request.form['swell_height']),
                float(request.form['current_speed'])
            ]

            # Preprocess input features
            features_scaled = scaler.transform([features])

            # Predict fuel consumption
            prediction = model.predict(features_scaled)
            fuel_consumed = round(float(prediction[0][0]), 2)
            print("Features (scaled):", features_scaled)
            print("Prediction:", prediction)

            # Render result.html with the calculated fuel consumption
            return render_template('calculate.html', fuel_consumed=fuel_consumed,form_data=request.form)
        except Exception as e:
            return render_template('calculate.html', error=f"Error: {str(e)}", form_data = request.form)

    # Render calculate.html for GET requests
    return render_template('calculate.html', form_data = {})


if __name__ == '__main__':
    app.run(debug=True)
