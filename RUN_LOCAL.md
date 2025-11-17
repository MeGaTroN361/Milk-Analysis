# Run the project locally (quick guide)

1. Set API secret:
   - macOS / Linux:
     ```bash
     export API_SECRET_KEY="your_test_key_here"
     ```
   - Windows PowerShell:
     ```powershell
     $env:API_SECRET_KEY = "your_test_key_here"
     ```

2. Create virtualenv and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start Flask (from project root where app.py exists):
   ```bash
   export FLASK_APP=app.py
   flask run --host=0.0.0.0 --port=5000
   ```

4. Test locally with sim_device:
   - Edit API_URL in sim_device.py if you need to point to LAN IP.
   - Run:
     ```bash
     python sim_device.py
     ```

5. To test with ESP32:
   - Open `esp32_post.ino` in Arduino IDE, set WIFI_SSID, WIFI_PASS, SERVER_URL, API_KEY.
   - Flash to device, open Serial Monitor at 115200.

Notes:
- If sending from a different device (ESP32 on LAN), make sure your PC's firewall allows incoming connections to port 5000.
- If you see 401 Unauthorized from Flask, ensure the X-API-KEY matches the environment variable used to start Flask.
