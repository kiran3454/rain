from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/rainfall_model.pkl')  # Ensure this path is correct

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect input data
            data = [
                float(request.form.get(field)) for field in [
                    'pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint',
                    'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'
                ]
            ]
            # Make prediction
            prediction = model.predict(np.array(data).reshape(1, -1))[0]
            
            # Redirect to results page with the prediction (converted to string)
            return redirect(url_for('result', prediction=str(prediction)))
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}")

    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction', None)
    
    # Convert the prediction into a readable string (e.g., "Rain" or "No Rain")
    if prediction == '1':
        prediction_text = 'Rain'
    elif prediction == '0':
        prediction_text = 'No Rain'
    else:
        prediction_text = 'Error in prediction'
    
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
