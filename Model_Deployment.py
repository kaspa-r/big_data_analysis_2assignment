# Model Deployment

from flask import Flask, request, jsonify, render_template
import pickle
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # Or use OneHotEncoder
from sklearn import ensemble

app = Flask(__name__)

try:
    model_GBC = pickle.load(open('models/GradientBoostingClassifier.pkl', 'rb'))
    model_LGBM = pickle.load(open('models/LGBMClassifier.pkl', 'rb'))
    model_XGB = pickle.load(open('models/XGBoostClassifier.pkl', 'rb'))
    models = {
              'GradientBoostingClassifier': model_GBC,
              'LGBMClassifier': model_LGBM,
              'XGBoostClassifier': model_XGB}
except FileNotFoundError as e:
    print(f"Error loading model file: {e}")
    models = {} # Initialize as empty if loading fails


WORK_TYPE_OPTIONS = ['Govt_job', 'children', 'Private', 'Self-employed', 'Never_worked']
SMOKING_STATUS_OPTIONS = ['Unknown', 'formerly smoked', 'smokes', 'never smoked']



RANDOM_RANGES = {
    'gender':(0, 1),
    'age' : (0, 100),
    'hypertension': (0, 1),
    'heart_disease': (0, 1),
    'ever_married': (0, 1),
    'work_type' : WORK_TYPE_OPTIONS,
    'residence_type': (0, 1),
    'avg_glucose_level': (55.0, 272.0),
    'bmi': (10.0, 80.0),
    "smoking_status" : SMOKING_STATUS_OPTIONS
    }

@app.route('/random_data')
def get_random_data():
    random_data = {}
    for feature, value_range in RANDOM_RANGES.items():
        if feature in ['work_type', 'smoking_status']:
            random_data[feature] = random.choice(value_range)
        elif isinstance(value_range, tuple):
            min_val, max_val = value_range
            if isinstance(min_val, int) and isinstance(max_val, int):
                random_data[feature] = random.randint(min_val, max_val)
            else:
                random_data[feature] = round(random.uniform(min_val, max_val), 2)
        else:
            # Handle other cases if needed
            random_data[feature] = None
    return jsonify(random_data)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = request.form['work_type']  # Get the string work type
        residence_type = int(request.form['residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi']) if request.form['bmi'] else 0.0
        smoking_status = request.form['smoking_status']
        selected_model_name = request.form['model_select']
        selected_model = models.get(selected_model_name)

        if selected_model:
            input_features = pd.DataFrame([{'gender' : gender,
                                            'age' : age,
                                            'hypertension' : hypertension,
                                            'heart_disease' : heart_disease,
                                            'ever_married' : ever_married,
                                            'work_type' : work_type,
                                            'residence_type' : residence_type,
                                            'avg_glucose_level' : avg_glucose_level,
                                            'bmi' : bmi,
                                            'smoking_status': smoking_status}])

            # Make the prediction
            prediction = selected_model.predict(input_features)[0]

            # Determine the prediction outcome
            if prediction == 1:
                result = 'Likely to have a stroke'
            else:
                result = 'Not likely to have a stroke'

            return render_template('index.html', result=result, selected_model=selected_model_name)
        else:
            return render_template('index.html', error="Selected model not found.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
