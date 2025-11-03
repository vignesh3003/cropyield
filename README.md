# Crop Yield Predictor — Flask API + Frontend

This small app serves the saved crop yield model and provides a minimal web UI.

Quick start (macOS / zsh):

1. Create a virtualenv and activate it

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
python app.py
```

Open http://localhost:5000 in your browser. Use the form or the AJAX button to get predictions.

Notes:

- The app loads `crop_yield_prediction_model.joblib` from the repository root. Keep that file present.
- The JSON endpoint is `POST /api/predict` and expects a JSON object with the form fields.
