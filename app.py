from flask import Flask, request, render_template, jsonify, send_file
import joblib
import pandas as pd
import os
import io

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


def validate_and_prepare(payload):
    """Validate incoming payload (dict-like) and return (df, errors).

    errors is None on success or a dict of field->message on failure.
    """
    errors = {}
    # Required fields
    required = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    for f in required:
        if payload.get(f) in (None, ''):
            errors[f] = 'This field is required.'

    # Numeric checks
    numeric_fields = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    for f in numeric_fields:
        v = payload.get(f)
        try:
            float(v)
        except Exception:
            errors[f] = 'Must be a number.'

    if errors:
        return None, errors

    # If ok, build DataFrame
    df = make_input_df(payload)
    return df, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'JSON body required'}), 400

    df, errors = validate_and_prepare(payload)
    if errors is not None:
        return jsonify({'errors': errors}), 400

    mdl = load_model()
    try:
        pred = mdl.predict(df)
        value = round(float(pred[0]), 4)
        return jsonify({'prediction': value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def form_predict():
    form = request.form
    payload = form.to_dict()
    df, errors = validate_and_prepare(payload)
    if errors is not None:
        return jsonify({'errors': errors}), 400

    mdl = load_model()
    try:
        pred = mdl.predict(df)
        value = round(float(pred[0]), 4)
        return jsonify({'prediction': value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """Accepts a CSV file upload (multipart/form-data, field 'file') or JSON list in body.
    Returns JSON list of predictions corresponding to input rows.
    """
    # If JSON list
    if request.is_json:
        payload = request.get_json()
        if not isinstance(payload, list):
            return jsonify({'error': 'Expected a JSON list of records for batch prediction.'}), 400
        records = payload
        try:
            df = pd.DataFrame(records)
        except Exception as e:
            return jsonify({'error': 'Could not parse JSON list: ' + str(e)}), 400
    else:
        # file upload expected
        if 'file' not in request.files:
            return jsonify({'error': "Missing file (field name 'file') for CSV upload."}), 400
        f = request.files['file']
        try:
            df = pd.read_csv(f)
        except Exception as e:
            return jsonify({'error': 'Could not read CSV file: ' + str(e)}), 400

    # Ensure required columns exist or are added
    # We'll try to be forgiving: if columns missing, set defaults where possible
    for col in ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']:
        if col not in df.columns:
            # fill with defaults
            if col in ['Crop', 'Season', 'State']:
                df[col] = ''
            else:
                df[col] = 0

    # Compute Season_Category and Decade and Production
    def map_season(s):
        mapping = {
            'Kharif': 'Monsoon',
            'Rabi': 'Winter',
            'Autumn': 'Autumn',
            'Summer': 'Summer',
            'Whole Year': 'Year-round',
            'Winter': 'Winter'
        }
        return mapping.get(s, 'Unknown')

    df['Season_Category'] = df['Season'].apply(map_season)
    try:
        df['Decade'] = (df['Crop_Year'].astype(float) // 10 * 10).astype(int)
    except Exception:
        df['Decade'] = 0
    df['Production'] = 0

    # Make predictions
    mdl = load_model()
    try:
        preds = mdl.predict(df)
        preds = [round(float(p), 4) for p in preds]
        # Return predictions paired with row index
        out = [{'index': int(i), 'prediction': p} for i, p in enumerate(preds)]
        return jsonify({'predictions': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # For local development - allow overriding host/port/debug via environment variables
    port = int(os.environ.get('PORT', os.environ.get('APP_PORT', '5000')))
    host = os.environ.get('HOST', '0.0.0.0')
    debug_env = os.environ.get('FLASK_DEBUG', os.environ.get('DEBUG', '1'))
    debug = str(debug_env).lower() in ('1', 'true', 'yes')
    print(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(debug=debug, host=host, port=port)
