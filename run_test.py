"""Simple integration test using Flask test client to call /api/predict."""
from app import app


def run_sample_test():
    client = app.test_client()
    sample = {
        "Crop": "Rice",
        "Crop_Year": 2018,
        "Season": "Rabi",
        "State": "Kerala",
        "Area": 247,
        "Annual_Rainfall": 198,
        "Fertilizer": 70,
        "Pesticide": 4,
    }
    resp = client.post('/api/predict', json=sample)
    print('Status code:', resp.status_code)
    print('Response:', resp.get_json())


if __name__ == '__main__':
    run_sample_test()
