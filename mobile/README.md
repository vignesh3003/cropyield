# Crop Yield Mobile (Expo)

This is a minimal Expo scaffold that calls the Flask API you have in the repo.

Quick start (macOS, zsh)

1. Install dependencies

```bash
cd mobile
npm install
```

2. Configure the API base URL

Open `mobile/config.js` and set `BASE_URL` to the address where your Flask server is reachable from the device or emulator.

- Android emulator: use `http://10.0.2.2:5001`
- iOS simulator: use `http://localhost:5001`
- Real device: use your machine's LAN IP like `http://192.168.1.10:5001` (ensure firewall allows connections)

3. Start the Expo dev server

```bash
npm run start
# then press 'a' to open Android emulator or 'i' for iOS simulator, or scan QR with Expo Go on your phone
```

Notes

- This scaffold is intentionally minimal. It provides a single-screen form and calls `/api/predict`.
- For a production app you'll want to add input validation, error handling, styling, and token-based auth if you expose the API publicly.
