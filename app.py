from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_yield_prediction_model.joblib')
model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


def make_input_df(src):
    """Build input DataFrame expected by the saved model from incoming data (dict or ImmutableMultiDict).
    Expected/used columns (based on training script):
    'Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide',
    'Season_Category', 'Decade', 'Production'
    """
    # Helper to get value from dict-like or form data
    def g(k, default=''):
        v = src.get(k)
        return v if v is not None else default

    # parse numeric fields safely
    def fnum(k, default=0.0):
        v = g(k, '')
        try:
            return float(v)
        except Exception:
            return float(default)

    year = fnum('Crop_Year', 0)
    decade = int(year // 10) * 10 if year else 0

    season = g('Season', '')
    season_mapping = {
        'Kharif': 'Monsoon',
        'Rabi': 'Winter',
        'Autumn': 'Autumn',
        'Summer': 'Summer',
        'Whole Year': 'Year-round',
        'Winter': 'Winter'
    }
    season_category = season_mapping.get(season, 'Unknown')

    data = {
        'Crop': [g('Crop', '')],
        'Crop_Year': [year],
        'Season': [season],
        'State': [g('State', '')],
        'Area': [fnum('Area', 0.0)],
        'Annual_Rainfall': [fnum('Annual_Rainfall', 0.0)],
        'Fertilizer': [fnum('Fertilizer', 0.0)],
        'Pesticide': [fnum('Pesticide', 0.0)],
        'Season_Category': [season_category],
        'Decade': [decade],
        'Production': [0]
    }
    return pd.DataFrame(data)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'JSON body required'}), 400

    df = make_input_df(payload)
    mdl = load_model()
    try:
        pred = mdl.predict(df)
        value = float(pred[0])
        return jsonify({'prediction': value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def form_predict():
    form = request.form
    df = make_input_df(form)
    mdl = load_model()
    try:
        pred = mdl.predict(df)
        value = round(float(pred[0]), 4)
        return render_template('index.html', prediction=value, input=form)
    except Exception as e:
        return render_template('index.html', error=str(e), input=form)


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
