import os

# --- API SECURITY ---
# This key must be included in the header of any request to the /api/predict endpoint
# In a real application, this should be a long, randomly generated secret string.
API_SECRET_KEY = os.environ.get('API_SECRET_KEY', 'T3stKey_92#Flask!')

# --- IDEAL MILK QUALITY RANGES ---
# Used for visualizations on the Dash dashboard
IDEAL_RANGES = {
    "temperature": {"min": 36.5, "max": 39.0},
    "ph": {"min": 6.5, "max": 6.8},
    "conductivity": {"min": 4.0, "max": 6.0},
    "fat_content": {"min": 3.5, "max": 5.0}
}

# --- SERVER STATE ---
# Do not change. This is the initial state for the listening workflow.
LISTENING_STATE = {
    "cow_id": None,
    "timestamp": None,
    "status": "idle"  # States: idle, listening, received
}
