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

## Progressive Web App (mobile-installable)

This app now supports PWA install on modern mobile browsers.

- Open the site on your phone's browser (e.g. http://<machine-ip>:5001). Use the browser menu and choose "Add to Home screen" or tap the install button if presented.
- The app will behave like a native app (standalone window) when launched from the home screen.
- Offline: the app shell is cached by a service worker so the UI loads offline; API calls require network (the service worker attempts network-first for /api/ requests).

Notes:

- If testing on your device, ensure the server is reachable from the device (use local network IP or deploy to a public host).
- For stricter offline batch predictions you'd need to ship a client-side model or implement background sync — I can help if you want that.
