# app.py — Smart Farm (updated)
# - Fix: api_predict now persists when both vitals+milk are present
# - Add: /records_json/<cow_id> (NaN->null) for table/graph
# - Add: Shared-device mode alongside per-tag, selectable via env
# - Keep: persistent device/tasks tables and all existing UI routes

import os
import time
import json
import math
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, Response, abort
)
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd

from config import API_SECRET_KEY

# Optional ML (loaded if present)
try:
    import joblib
except Exception:
    joblib = None
from collections import deque

GLOBAL_TASKS = deque(maxlen=200)  # shared device tasks (FIFO)

def log(msg, *args):
    try:
        app.logger.info(msg, *args)
    except Exception:
        print(msg % args if args else msg)


# =============================================================================
# CONFIG
# =============================================================================
app = Flask(__name__)
# --- Listen state & pending buffers ---
from collections import defaultdict
import time

# One listen session per cow; supports multiple cows concurrently
LISTENING = defaultdict(lambda: {
    "status": "idle",     # "idle" | "listening"
    "vitals": False,      # did we get vitals in this session?
    "milk": False,        # did we get milk in this session?
    "started_at": 0.0,    # epoch seconds
})
PENDING = defaultdict(dict)  # cow_id -> merged pending dict
SESSION_TTL = 5 * 60         # auto-expire listen after 5 minutes of inactivity


# --- Change these for your environment ---
app.secret_key = os.environ.get("SECRET_KEY", "123")
DB_PATH = os.environ.get("DB_PATH", "farm.db")
# X-API-KEY expected from ESP32 firmware on device routes
app.config["API_SECRET_KEY"] = os.environ.get("API_SECRET_KEY", "123")

# Online heartbeat window for device presence (>= long-poll)
ONLINE_TTL_SECONDS = int(os.environ.get("ONLINE_TTL", "180"))

# Device mode:
#   SHARED_DEVICE_MODE=true  -> ESP polls with ?device=<SHARED_DEVICE_ID>
#   SHARED_DEVICE_MODE=false -> ESP polls with ?tag=<cow_tag>
# SHARED_DEVICE_MODE = os.environ.get("SHARED_DEVICE_MODE", "false").lower() == "true"
SHARED_DEVICE_MODE = True

SHARED_DEVICE_ID   = os.environ.get("SHARED_DEVICE_ID", "esp32_1")

# =============================================================================
# IDEAL RANGES & ADVICE
# =============================================================================
IDEAL = {
    "temperature":  {"min": 36.5, "max": 39.0},   # °C
    "ph":           {"min": 6.5,  "max": 6.8},
    "turbidity":    {"min": 0.3,  "max": 0.7},
    "conductivity": {"min": 4.0,  "max": 6.0},
    "fat_content":  {"min": 3.5,  "max": 5.0},
    "heart_rate":   {"min": 50,   "max": 90},     # bpm
}
import time
from collections import defaultdict, deque

# Session/config knobs
SESSION_TTL = 5 * 60  # seconds

# In-memory state
LISTENING = defaultdict(lambda: {"status": "idle", "vitals": False, "milk": False, "started_at": 0.0})
PENDING   = defaultdict(dict)
GLOBAL_TASKS = deque(maxlen=200)  # make sure it's a deque, not a list

def _expire_session_if_needed(cow_id: int) -> None:
    """Auto-expire a listening session and clear buffer after TTL."""
    s = LISTENING.get(cow_id)
    if not s:
        return
    try:
        if s.get("status") == "listening":
            started = float(s.get("started_at") or 0.0)
            if (time.time() - started) > SESSION_TTL:
                s.update({"status": "idle", "vitals": False, "milk": False})
                PENDING[cow_id].clear()
    except Exception:
        # never let this crash request handling
        pass


DISEASE_INFO = {
    "healthy": {
        "title": "Healthy",
        "description": "All parameters are within the ideal ranges.",
        "advice": "Keep routine monitoring and good nutrition."
    },
    "mastitis_risk": {
        "title": "Mastitis Risk",
        "description": "Indicators suggest udder inflammation.",
        "advice": "Improve udder hygiene; consider CMT; consult vet if persists."
    },
    "ketosis_risk": {
        "title": "Ketosis Risk",
        "description": "Energy balance issue suspected.",
        "advice": "Increase energy-dense feed; check ketones; vet check if needed."
    },
    "metritis_risk": {
        "title": "Metritis Risk",
        "description": "Possible uterine/reproductive tract infection.",
        "advice": "Observe discharge/fever; consult a veterinarian."
    },
    "udder_infection": {
        "title": "Udder Infection",
        "description": "Signs consistent with an udder infection.",
        "advice": "Isolate, hold milk, and seek veterinary care."
    },
    "warning": {
        "title": "Warning",
        "description": "One metric is out of the ideal range.",
        "advice": "Monitor and recheck soon."
    },
    "risk": {
        "title": "At Risk",
        "description": "Multiple metrics are out of the ideal range.",
        "advice": "Consider a prompt veterinary check."
    }
}

# =============================================================================
# OPTIONAL: Load ML models if present
# =============================================================================
fat_model = None
disease_model = None
if joblib:
    for p in ("models/fat_model.pkl", "fat_model.pkl"):
        if os.path.exists(p):
            try:
                fat_model = joblib.load(p)
                app.logger.info("Loaded fat_model from %s", p)
                break
            except Exception as e:
                app.logger.warning("Failed loading fat_model from %s: %s", p, e)
    for p in ("models/disease_model.pkl", "disease_model.pkl"):
        if os.path.exists(p):
            try:
                disease_model = joblib.load(p)
                app.logger.info("Loaded disease_model from %s", p)
                break
            except Exception as e:
                app.logger.warning("Failed loading disease_model from %s: %s", p, e)

# =============================================================================
# DB HELPERS & BOOTSTRAP
# =============================================================================
def db_conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c
@app.post("/_test_buffer/<int:cow_id>")
def _test_buffer(cow_id):
    LISTENING[cow_id].update({"status":"listening","vitals":True,"milk":True,"started_at":time.time()})
    PENDING[cow_id].update({
        "temperature": 37.6,
        "heart_rate":  72,
        "ph":          6.82,
        "turbidity":   0.44,
        "conductivity":4.9,
        "fat_content": 5.2,
        "predicted_fat": 5.18
    })
    return jsonify({"ok": True})

def ensure_tables():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        email TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cow_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag_id TEXT UNIQUE NOT NULL,
        age INTEGER,
        breed TEXT,
        notes TEXT,
        owner_user_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (owner_user_id) REFERENCES users(user_id)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS milk_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cow_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        temperature REAL,
        ph REAL,
        turbidity REAL,
        conductivity REAL,
        fat_content REAL,
        heart_rate REAL,
        predicted_fat REAL,
        health_status TEXT,
        FOREIGN KEY (cow_id) REFERENCES cow_details(id)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cow_owner ON cow_details(owner_user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mr_cow ON milk_records(cow_id);")
    conn.commit()
    conn.close()

# Persistent device & task tables (kept from your version)
def ensure_device_tables():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS devices (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         tag_id TEXT UNIQUE NOT NULL,
         cow_id INTEGER,
         api_key TEXT,
         last_seen DATETIME,
         created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS device_tasks (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         device_tag TEXT NOT NULL,
         task TEXT NOT NULL,
         status TEXT NOT NULL DEFAULT 'queued',
         created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
         delivered_at DATETIME,
         result TEXT
      )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_device_tasks_tag ON device_tasks(device_tag)")
    conn.commit()
    conn.close()

def ensure_admin():
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users;")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?,?,?)",
            ("admin", generate_password_hash("admin123"), "Default Admin"),
        )
        conn.commit()
        app.logger.info("Default admin created: admin / admin123")
    conn.close()

ensure_tables()
ensure_device_tables()
ensure_admin()

# =============================================================================
# AUTH DECORATORS
# =============================================================================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper

def require_device_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        want = app.config.get("API_SECRET_KEY")
        have = request.headers.get("X-API-KEY")
        if want and have != want:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# =============================================================================
# DEVICE PRESENCE & TASK QUEUE
# =============================================================================
DEVICE_LAST_SEEN = {}   # in-memory mirror for quick checks

# shared-device in-memory queue (when SHARED_DEVICE_MODE=true)
# GLOBAL_TASKS = []  # list of {"task": "vitals"|"milk", "tag": "<cow_tag>"}
SHARED_LAST_SEEN = {}

# listening/pending buffers
listening_state = {
    "status": None,          # None | "listening" | "received"
    "cow_id": None,
    "mode": None,            # "vitals" | "milk"
    "next": None,            # redirect url
    "ts": None               # start time
}
pending_readings = {}  # {cow_id: {"vitals": {...}, "milk": {...}}}

def _cow_tag(cow_id: int):
    conn = db_conn()
    row = conn.execute("SELECT tag_id FROM cow_details WHERE id=?", (cow_id,)).fetchone()
    conn.close()
    return row["tag_id"] if row else None

def _online_tag(tag: str) -> bool:
    # prefer DB last_seen
    conn = db_conn()
    row = conn.execute("SELECT last_seen FROM devices WHERE tag_id=?", (tag,)).fetchone()
    conn.close()
    if row and row["last_seen"]:
        try:
            dt = datetime.strptime(row["last_seen"], "%Y-%m-%d %H:%M:%S")
            age = (datetime.utcnow() - dt).total_seconds()
            return age < ONLINE_TTL_SECONDS
        except Exception:
            pass
    ts = DEVICE_LAST_SEEN.get(tag)
    return bool(ts and (time.time() - ts) < ONLINE_TTL_SECONDS)

def _online_shared() -> bool:
    last = SHARED_LAST_SEEN.get(SHARED_DEVICE_ID)
    return bool(last and (time.time() - last) < ONLINE_TTL_SECONDS)

def _start_listening(cow_id: int, mode: str, next_url: str):
    listening_state.update({
        "status": "listening",
        "cow_id": cow_id,
        "mode": mode,
        "next": next_url,
        "ts": time.time()
    })

def _expire_listening_if_needed():
    if listening_state["status"] == "listening":
        if time.time() - (listening_state.get("ts") or 0) > 120:
            listening_state.update({"status": None, "cow_id": None, "mode": None, "next": None, "ts": None})

# =============================================================================
# DEVICE API (ESP32) — X-API-KEY protected
# =============================================================================
@app.route("/device/ping", methods=["POST"])
@require_device_key
def device_ping():
    """
    Shared mode: {"device_id":"esp32_1"}
    Per-tag mode: {"tag":"admin_cow1"}
    """
    data = request.get_json(silent=True) or {}
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    if SHARED_DEVICE_MODE:
        device_id = (data.get("device_id") or "").strip()
        if not device_id:
            return jsonify({"error": "missing device_id"}), 400
        SHARED_LAST_SEEN[device_id] = time.time()
        return jsonify({"ok": True, "mode": "shared", "device_id": device_id})

    # per-tag
    tag = (data.get("tag") or "").strip()
    if not tag:
        return jsonify({"error": "missing tag"}), 400
    # upsert device row
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM devices WHERE tag_id=?", (tag,))
    if cur.fetchone():
        cur.execute("UPDATE devices SET last_seen=? WHERE tag_id=?", (now, tag))
    else:
        cur.execute("INSERT INTO devices (tag_id, last_seen) VALUES (?, ?)", (tag, now))
    conn.commit()
    conn.close()
    DEVICE_LAST_SEEN[tag] = time.time()
    return jsonify({"ok": True, "mode": "per_tag", "tag": tag})

@app.route("/device/get_task", methods=["GET"])
@require_device_key
def device_get_task():
    """
    Shared mode: /device/get_task?device=<SHARED_DEVICE_ID>
    Per-tag mode: /device/get_task?tag=<cow_tag>
    """
    device_param = (request.args.get("device") or "").strip()
    tag_param    = (request.args.get("tag") or "").strip()

    # SHARED DEVICE MODE
    if SHARED_DEVICE_MODE:
        if device_param != SHARED_DEVICE_ID:
            return jsonify({"error":"wrong_device","expect":SHARED_DEVICE_ID}), 400
        # presence touch
        SHARED_LAST_SEEN[device_param] = time.time()

        # immediate
        if GLOBAL_TASKS:
            task = GLOBAL_TASKS.pop(0)
            app.logger.info("DELIVERED(shared) %s", task)
            return jsonify(task)

        # long poll
        deadline = time.time() + 60.0
        last_touch = time.time()
        while time.time() < deadline:
            if time.time() - last_touch >= 5.0:
                SHARED_LAST_SEEN[device_param] = time.time()
                last_touch = time.time()
            if GLOBAL_TASKS:
                task = GLOBAL_TASKS.pop(0)
                app.logger.info("DELIVERED(shared longpoll) %s", task)
                return jsonify(task)
            time.sleep(0.2)
        return jsonify({"task": None})

    # PER-TAG MODE
    if not tag_param:
        return jsonify({"error":"missing tag"}), 400

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn = db_conn()
    cur = conn.cursor()
    # update last_seen
    cur.execute("UPDATE devices SET last_seen=? WHERE tag_id=?", (now, tag_param))
    conn.commit()

    # immediate
    row = cur.execute(
        "SELECT id, task, result FROM device_tasks WHERE device_tag=? AND status='queued' ORDER BY created_at ASC LIMIT 1",
        (tag_param,)
    ).fetchone()
    if row:
        cur.execute("UPDATE device_tasks SET status='delivered', delivered_at=? WHERE id=?", (now, row["id"]))
        conn.commit()
        conn.close()
        payload = None
        try:
            payload = json.loads(row["result"]) if row["result"] else None
        except Exception:
            payload = None
        return jsonify({"task": row["task"], "payload": payload})

    conn.close()
    # long poll
    deadline = time.time() + 60.0
    while time.time() < deadline:
        time.sleep(0.4)
        conn = db_conn()
        cur = conn.cursor()
        row = cur.execute(
            "SELECT id, task, result FROM device_tasks WHERE device_tag=? AND status='queued' ORDER BY created_at ASC LIMIT 1",
            (tag_param,)
        ).fetchone()
        if row:
            now2 = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("UPDATE device_tasks SET status='delivered', delivered_at=? WHERE id=?", (now2, row["id"]))
            conn.commit()
            conn.close()
            payload = None
            try:
                payload = json.loads(row["result"]) if row["result"] else None
            except Exception:
                payload = None
            return jsonify({"task": row["task"], "payload": payload})
        conn.close()
    return jsonify({"task": None})

# Back-compat alias
@app.get("/device/wait_task")
def device_wait_task():
    # Require API key (keep this ON for ESP/sim)
    key_ok = request.headers.get("X-API-KEY") == app.config.get("API_SECRET_KEY")
    if not key_ok:
        # Helpful log – if you see these, add the header in your firmware/sim
        log("wait_task unauthorized: missing/bad X-API-KEY from %s", request.remote_addr)
        return jsonify({"error":"unauthorized"}), 401

    device = request.args.get("device")
    if SHARED_DEVICE_MODE:
        if device != SHARED_DEVICE_ID:
            log("wait_task wrong device=%s expected=%s", device, SHARED_DEVICE_ID)
            return jsonify({"task": None}), 200

        # 1) Serve from explicit queue first (preferred)
        if GLOBAL_TASKS:
            task = GLOBAL_TASKS.popleft()
            log("DISPATCH task=%s tag=%s cow=%s", task.get("task"), task.get("tag"), task.get("cow_id"))
            return jsonify({"task": task["task"], "tag": task["tag"]}), 200

        # 2) Fallback: scan LISTENING (in case a task was created before this patch)
        for cid, s in list(LISTENING.items()):
            _expire_session_if_needed(cid)
            if s["status"] != "listening":
                continue
            tag = _cow_tag(cid)
            if not tag:
                continue
            if not s["vitals"]:
                log("DISPATCH (scan) task=vitals tag=%s cow=%s", tag, cid)
                return jsonify({"task":"vitals","tag": tag}), 200
            if not s["milk"]:
                log("DISPATCH (scan) task=milk tag=%s cow=%s", tag, cid)
                return jsonify({"task":"milk","tag": tag}), 200

        # no work
        return jsonify({"task": None}), 200

    # (Tag-mode not used here; OK to return no task)
    return jsonify({"task": None}), 200

# ---- Predicted fat helper (model or heuristic) ----
def _safe_float(x):
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None

_MODEL = {"loaded": False, "fn": None}
def _load_predictor_once():
    if _MODEL["loaded"]:
        return _MODEL["fn"]
    # Try model_pipeline.py if present
    try:
        from model_pipeline import load_model, predict as model_predict
        mdl = load_model()
        def _predict_with_model(feats: dict):
            # Expect features like: temperature, ph, turbidity, conductivity, heart_rate, fat_content(optional)
            # Map safely; model should handle missing gracefully, else fill 0
            keys = ["temperature","ph","turbidity","conductivity","heart_rate","fat_content"]
            x = {k: (_safe_float(feats.get(k)) or 0.0) for k in keys}
            y = model_predict(mdl, x)  # should return float
            try:
                y = float(y)
            except Exception:
                return None
            if y != y:  # NaN
                return None
            return round(y, 2)
        _MODEL["fn"] = _predict_with_model
        _MODEL["loaded"] = True
        return _MODEL["fn"]
    except Exception:
        # Fallback heuristic if no model
        def _heuristic(feats: dict):
            t  = _safe_float(feats.get("temperature"))
            ph = _safe_float(feats.get("ph"))
            tu = _safe_float(feats.get("turbidity"))
            co = _safe_float(feats.get("conductivity"))
            hr = _safe_float(feats.get("heart_rate"))
            # Need at least two features
            vals = [v for v in [t, ph, tu, co, hr] if v is not None]
            if len(vals) < 2:
                return None
            # Simple, stable heuristic centered near 4.5–6.0%
            base = 4.8
            if ph is not None: base += (ph - 6.8) * 0.6
            if co is not None: base += (5.0 - co) * 0.25
            if tu is not None: base -= tu * 0.2
            if t  is not None: base += (37.5 - t) * 0.15
            # clamp to reasonable range
            base = max(2.0, min(7.5, base))
            return round(base, 2)
        _MODEL["fn"] = _heuristic
        _MODEL["loaded"] = True
        return _MODEL["fn"]

def predict_fat_server(features: dict):
    """Return a 2-decimal float or None."""
    fn = _load_predictor_once()
    try:
        y = fn(features)
        if y is None: 
            return None
        y = float(y)
        if y != y: 
            return None
        return round(y, 2)
    except Exception:
        return None

@app.post("/api/predict")
@require_device_key
def api_predict():

    """
    Called by device (ESP32 or simulator). When a listen session is active for a cow,
    we BUFFER the fields in PENDING[cow_id] and DO NOT write to DB yet.
    If no session is active, we keep legacy behavior (insert immediately).
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "bad_json"}), 400

    tag = (payload.get("tag") or "").strip()
    if not tag:
        return jsonify({"error": "missing_tag"}), 400

    # Resolve cow_id by tag
    conn = db_conn()
    cow = conn.execute("SELECT id FROM cow_details WHERE tag_id=?", (tag,)).fetchone()
    if not cow:
        conn.close()
        return jsonify({"error": "unknown_tag"}), 404
    cow_id = int(cow["id"])

    # fields we accept
    fields = {
        "temperature": payload.get("temperature"),
        "heart_rate": payload.get("heart_rate"),
        "ph": payload.get("ph"),
        "turbidity": payload.get("turbidity"),
        "conductivity": payload.get("conductivity"),
        "fat_content": payload.get("fat_content"),
        "predicted_fat": payload.get("predicted_fat"),
        "health_status": payload.get("health_status"),
    }
    # clean NaN-like values -> None
    for k, v in list(fields.items()):
        try:
            if isinstance(v, float) and (v != v):  # NaN
                fields[k] = None
        except Exception:
            pass

    # Is there an active listen session for this cow?
    s = LISTENING[cow_id]
    if s["status"] == "listening":
        # merge fields into buffer
        for k, v in fields.items():
            if v is not None:
                PENDING[cow_id][k] = v

        # If predicted_fat not sent by device, compute it and stash in PENDING for the form
        if PENDING[cow_id].get("predicted_fat") is None:
            pf = predict_fat_server({
                "temperature":  PENDING[cow_id].get("temperature"),
                "ph":           PENDING[cow_id].get("ph"),
                "turbidity":    PENDING[cow_id].get("turbidity"),
                "conductivity": PENDING[cow_id].get("conductivity"),
                "heart_rate":   PENDING[cow_id].get("heart_rate"),
                "fat_content":  PENDING[cow_id].get("fat_content"),
            })
            if pf is not None:
                PENDING[cow_id]["predicted_fat"] = pf

        if fields.get("temperature") is not None or fields.get("heart_rate") is not None:
            s["vitals"] = True
        if any(fields.get(k) is not None for k in ("ph","turbidity","conductivity","fat_content","predicted_fat")):
            s["milk"] = True
        s["started_at"] = time.time()
        return jsonify({"status":"buffered","vitals":s["vitals"],"milk":s["milk"], "predicted_fat": PENDING[cow_id].get("predicted_fat")}), 200


    # --- LEGACY MODE (no active listening): insert immediately as before ---
    try:
        conn.execute("""
            INSERT INTO milk_records
              (cow_id, timestamp, temperature, ph, turbidity, conductivity, heart_rate,
               fat_content, predicted_fat, health_status, source)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, 'device')
        """, (cow_id, fields["temperature"], fields["ph"], fields["turbidity"], fields["conductivity"],
              fields["heart_rate"], fields["fat_content"], fields["predicted_fat"], fields["health_status"]))
        conn.commit()
    finally:
        conn.close()
    return jsonify({"status": "saved", "cow_id": cow_id}), 200


def _merge_predict_and_write(cow_id: int, buf: dict):
    """
    Combine vitals + milk, run ML (if available), compute fallback status/advice,
    write one row to milk_records, clear buffer for this cow, and respond.
    """
    features = {
        "temperature": buf["vitals"]["temperature"],
        "ph":          buf["milk"]["ph"],
        "turbidity":   buf["milk"]["turbidity"],
        "conductivity":buf["milk"]["conductivity"],
        "heart_rate":  buf["vitals"]["heart_rate"],
    }

    predicted_fat = None
    predicted_status = None
    if joblib:
        try:
            X = pd.DataFrame([features])
            if fat_model is not None:
                predicted_fat = float(fat_model.predict(X)[0])
            if disease_model is not None:
                predicted_status = str(disease_model.predict(X)[0])
        except Exception as e:
            app.logger.error("Model prediction error: %s", e)

    fat_final = buf["milk"].get("fat_content")
    if fat_final is None:
        fat_final = predicted_fat

    # Fallback rule-based status if classifier missing/uncertain
    if not predicted_status or str(predicted_status).strip().lower() == "none":
        alerts = []
        for k, rng in IDEAL.items():
            if k in features and features[k] is not None:
                try:
                    v = float(features[k])
                    mn, mx = rng.get("min"), rng.get("max")
                    if mn is not None and mx is not None and (v < mn or v > mx):
                        alerts.append(k)
                except Exception:
                    pass
        if len(alerts) == 0:
            predicted_status = "healthy"
        elif len(alerts) == 1:
            predicted_status = "warning"
        else:
            predicted_status = "risk"

    conn = db_conn()
    try:
        conn.execute("""
            INSERT INTO milk_records
                (cow_id, timestamp, temperature, ph, turbidity, conductivity,
                 fat_content, heart_rate, predicted_fat, health_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cow_id,
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            features["temperature"], features["ph"], features["turbidity"],
            features["conductivity"], fat_final, features["heart_rate"],
            predicted_fat, predicted_status
        ))
        conn.commit()
    finally:
        conn.close()

    # Clear buffer for this cow
    pending_readings.pop(cow_id, None)
    # Flip listening to received so the UI can proceed
    if listening_state["status"] == "listening" and listening_state["cow_id"] == cow_id:
        listening_state["status"] = "received"

    # Best-effort: mark last delivered device_task done (per-tag mode only)
    if not SHARED_DEVICE_MODE:
        try:
            conn2 = db_conn()
            cur2 = conn2.cursor()
            tag = _cow_tag(cow_id)
            if tag:
                cur2.execute("""
                   SELECT id FROM device_tasks
                   WHERE device_tag=? AND status='delivered'
                   ORDER BY delivered_at DESC LIMIT 1
                """, (tag,))
                t = cur2.fetchone()
                if t:
                    cur2.execute("UPDATE device_tasks SET status='done', result=? WHERE id=?",
                                 (json.dumps({'saved': True, 'ts': datetime.utcnow().isoformat()}), t["id"]))
                    conn2.commit()
        except Exception as e:
            app.logger.warning("Failed to mark device_task done: %s", e)
        finally:
            try: conn2.close()
            except: pass

    return jsonify({
        "status": "merged_saved",
        "predicted_fat": predicted_fat,
        "disease": predicted_status
    })

# =============================================================================
# DEVICE CONTROL (UI) — login protected
# =============================================================================
@app.route("/device/ready/<int:cow_id>")
@login_required
def device_ready(cow_id):
    tag = _cow_tag(cow_id)
    if not tag:
        return jsonify({"ready": False, "reason": "cow_not_found"}), 404
    if SHARED_DEVICE_MODE:
        return jsonify({"ready": _online_shared(), "mode": "shared", "device": SHARED_DEVICE_ID})
    return jsonify({"ready": _online_tag(tag), "mode": "per_tag", "tag": tag})

@app.post("/device/enqueue/vitals/<int:cow_id>")
@login_required
def device_enqueue_vitals(cow_id):
    s = LISTENING[cow_id]
    s["status"]="listening"; s["vitals"]=False; s.setdefault("milk", False); s["started_at"]=time.time()
    for k in ("temperature","heart_rate"): PENDING[cow_id].pop(k, None)

    # NEW: enqueue a device task now (so the ESP32 gets it right away)
    tag = _cow_tag(cow_id)
    if tag and SHARED_DEVICE_MODE:
        GLOBAL_TASKS.append({"device": SHARED_DEVICE_ID, "task":"vitals", "tag": tag, "cow_id": cow_id, "ts": time.time()})
        log("ENQUEUED task=vitals tag=%s cow=%s", tag, cow_id)

    return jsonify({"ok":True,"mode":"vitals","cow_id":cow_id})


@app.post("/device/enqueue/milk/<int:cow_id>")
@login_required
def device_enqueue_milk(cow_id):
    s = LISTENING[cow_id]
    s["status"]="listening"; s["milk"]=False; s.setdefault("vitals", False); s["started_at"]=time.time()
    for k in ("ph","turbidity","conductivity","fat_content","predicted_fat"): PENDING[cow_id].pop(k, None)

    # NEW: enqueue a device task now
    tag = _cow_tag(cow_id)
    if tag and SHARED_DEVICE_MODE:
        GLOBAL_TASKS.append({"device": SHARED_DEVICE_ID, "task":"milk", "tag": tag, "cow_id": cow_id, "ts": time.time()})
        log("ENQUEUED task=milk tag=%s cow=%s", tag, cow_id)

    return jsonify({"ok":True,"mode":"milk","cow_id":cow_id})


@app.route("/device/diag/<int:cow_id>")
@login_required
def device_diag(cow_id):
    tag = _cow_tag(cow_id)
    if not tag:
        return jsonify({"error": "cow_not_found"}), 404
    conn = db_conn()
    row = conn.execute("SELECT last_seen FROM devices WHERE tag_id=?", (tag,)).fetchone()
    conn.close()
    age = None
    online = False
    if row and row["last_seen"]:
        try:
            dt = datetime.strptime(row["last_seen"], "%Y-%m-%d %H:%M:%S")
            age = round((datetime.utcnow() - dt).total_seconds(), 1)
            online = age < ONLINE_TTL_SECONDS
        except Exception:
            pass
    return jsonify({
        "tag": tag,
        "online": online if not SHARED_DEVICE_MODE else _online_shared(),
        "mode": "shared" if SHARED_DEVICE_MODE else "per_tag",
        "shared_device_id": SHARED_DEVICE_ID,
        "global_queue_len": len(GLOBAL_TASKS),
        "queued_task_count": _count_queued_tasks_for_tag(tag)
    })

def _count_queued_tasks_for_tag(tag):
    if SHARED_DEVICE_MODE:
        return sum(1 for t in GLOBAL_TASKS if t.get("tag") == tag)
    conn = db_conn()
    cur = conn.cursor()
    r = cur.execute("SELECT COUNT(*) as c FROM device_tasks WHERE device_tag=? AND status='queued'", (tag,)).fetchone()
    conn.close()
    return r["c"] if r else 0

@app.route("/health")
def health():
    return "ok", 200

# =============================================================================
# LISTENING VIEWS (UI)
# =============================================================================
@app.route("/listen_vitals/<int:cow_id>")
@login_required
def listen_vitals(cow_id):
    s = LISTENING[cow_id]
    s["status"] = "listening"
    s["vitals"] = False
    s.setdefault("milk", False)
    s["started_at"] = time.time()
    # Clear only vitals fields so any milk data buffered stays
    for k in ("temperature", "heart_rate"):
        PENDING[cow_id].pop(k, None)
    return redirect(url_for("connect", cow_id=cow_id, next=request.args.get("next")))


@app.route("/listen_milk/<int:cow_id>")
@login_required
def listen_milk(cow_id):
    s = LISTENING[cow_id]
    s["status"] = "listening"
    s["milk"] = False
    s.setdefault("vitals", False)
    s["started_at"] = time.time()
    # Clear only milk fields so any vitals data buffered stays
    for k in ("ph","turbidity","conductivity","fat_content","predicted_fat"):
        PENDING[cow_id].pop(k, None)
    return redirect(url_for("connect", cow_id=cow_id, next=request.args.get("next")))

# If not already imported near the top:
from flask import render_template, url_for, abort, request

def _flag_in_range(name, val):
    """Return True (in range), False (out), or None if not applicable."""
    try:
        if val is None:
            return None
        v = float(val)
    except Exception:
        return None
    rng = IDEAL.get(name)
    if not rng:
        return None
    if rng.get("min") is not None and v < rng["min"]:
        return False
    if rng.get("max") is not None and v > rng["max"]:
        return False
    return True
@app.route("/cow/<int:cow_id>")
@login_required
def cow_dashboard(cow_id):
    """
    Cow page with:
      - Latest readings summary (top)
      - Toggle: Table (server-side) / Graph (client-side)
      - Table shows newest-first recent rows with green/red coloring
    Use `?view=server` (or `?view=table`) to land on the table tab, `?view=graph` for graph tab.
    """
    # which tab is active
    view = (request.args.get("view") or "").lower()
    server_table = view in ("server", "table")

    conn = db_conn()
    cow = conn.execute("SELECT id, tag_id, age, breed FROM cow_details WHERE id=?", (cow_id,)).fetchone()
    if not cow:
        conn.close()
        abort(404)

    # Latest record (for the top summary)
    latest = conn.execute("""
      SELECT id, timestamp, temperature, ph, turbidity, conductivity, heart_rate,
             fat_content, predicted_fat, health_status
        FROM milk_records
       WHERE cow_id=?
       ORDER BY datetime(timestamp) DESC
       LIMIT 1
    """, (cow_id,)).fetchone()

    # Status & disease info
    status = (latest["health_status"].strip().lower().replace(" ", "_")
              if latest and latest["health_status"] else None)
    if not status:
        alerts = []
        if latest:
            # derive simple status from IDEAL ranges
            for k, rng in IDEAL.items():
                try:
                    v = latest[k] if k in latest.keys() else None
                except Exception:
                    v = None
                if v is None:
                    continue
                try:
                    vv = float(v)
                    if rng.get("min") is not None and rng.get("max") is not None:
                        if vv < rng["min"] or vv > rng["max"]:
                            alerts.append(k)
                except Exception:
                    pass
        status = "healthy" if len(alerts)==0 else ("warning" if len(alerts)==1 else "risk")

    disease_info = DISEASE_INFO.get(status, {
        "title": status.title() if status else "Unknown",
        "description": "—",
        "advice": "Review trends and consult a veterinarian if concerned."
    })

    # ---------- ALWAYS build recent_records for the server-side table ----------
    # Pull newest-first rows; avoid SQLite date() quirks.
    df_recent = pd.read_sql_query("""
      SELECT id, timestamp, temperature, ph, turbidity, conductivity, heart_rate,
             fat_content, predicted_fat, health_status
        FROM milk_records
       WHERE cow_id=?
       ORDER BY datetime(timestamp) DESC
       LIMIT 200
    """, conn, params=(cow_id,))

    recent_records = []
    if not df_recent.empty:
        df_recent = df_recent.where(pd.notna(df_recent), None)
        for _, r in df_recent.iterrows():
            fat = r.get("fat_content")
            if fat is None:
                fat = r.get("predicted_fat")

            rec = {
                "id":           r["id"],
                "timestamp":    str(r["timestamp"]),
                "temperature":  r["temperature"],
                "heart_rate":   r["heart_rate"],
                "ph":           r["ph"],
                "turbidity":    r["turbidity"],
                "conductivity": r["conductivity"],
                "fat":          fat,
                "health_status": r.get("health_status")
            }
            # in-range flags for green/red styling
            rec["ok_temperature"]  = _flag_in_range("temperature",  rec["temperature"])
            rec["ok_heart_rate"]   = _flag_in_range("heart_rate",   rec["heart_rate"])
            rec["ok_ph"]           = _flag_in_range("ph",           rec["ph"])
            rec["ok_turbidity"]    = _flag_in_range("turbidity",    rec["turbidity"])
            rec["ok_conductivity"] = _flag_in_range("conductivity", rec["conductivity"])
            rec["ok_fat"]          = _flag_in_range("fat_content",  rec["fat"])
            recent_records.append(rec)

    # Graph fetch URL
    records_json = url_for("records_json", cow_id=cow_id) if "records_json" in app.view_functions else None

    # Listen links (only if routes exist)
    vitals_url = url_for("listen_vitals", cow_id=cow_id, next=url_for("cow_dashboard", cow_id=cow_id)) if "listen_vitals" in app.view_functions else None
    milk_url   = url_for("listen_milk",   cow_id=cow_id, next=url_for("cow_dashboard", cow_id=cow_id)) if "listen_milk"   in app.view_functions else None

    conn.close()
    return render_template(
        "cow_dashboard.html",
        cow=cow,
        latest=latest,
        status=status,
        disease_info=disease_info,
        ideal=IDEAL,
        records_json_url=records_json,
        server_table=server_table,
        recent_records=recent_records,
        vitals_url=vitals_url,
        milk_url=milk_url
    )

# Also support /listen_*?cow_id=...
@app.route("/listen_vitals")
@login_required
def listen_vitals_qp():
    cow_id = request.args.get("cow_id", type=int)
    if not cow_id:
        abort(400)
    return listen_vitals(cow_id)

@app.route("/listen_milk")
@login_required
def listen_milk_qp():
    cow_id = request.args.get("cow_id", type=int)
    if not cow_id:
        abort(400)
    return listen_milk(cow_id)

@app.get("/check_listen_status")
@login_required
def check_listen_status():
    # Optionally accept ?cow_id to scope
    try:
        cow_id = int(request.args.get("cow_id")) if request.args.get("cow_id") else None
    except Exception:
        cow_id = None

    def snapshot(cid):
        s = LISTENING[cid]
        # expire stale sessions
        if s["status"] == "listening" and (time.time() - s["started_at"] > SESSION_TTL):
            s.update({"status":"idle","vitals":False,"milk":False})
            PENDING[cid].clear()
        return {"cow_id": cid, "status": s["status"], "vitals": s["vitals"], "milk": s["milk"]}

    if cow_id is not None:
        return jsonify(snapshot(cow_id))

    # no cow_id -> return all active sessions
    active = {cid: snapshot(cid) for cid in list(LISTENING.keys())}
    return jsonify(active)

@app.route("/cancel_listen", methods=["POST"])
@login_required
def cancel_listen():
    listening_state.update({"status": None, "cow_id": None, "mode": None, "next": None, "ts": None})
    return jsonify({"status": "cancelled"})

# =============================================================================
# AUTH (UI)
# =============================================================================
@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        conn = db_conn()
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["user_id"]
            session["username"] = row["username"]
            return redirect(request.args.get("next") or url_for("index"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET","POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if not username or not password:
            error = "Username and password required"
        else:
            try:
                conn = db_conn()
                conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                             (username, generate_password_hash(password)))
                conn.commit()
                conn.close()
                flash("Account created. Please login.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                error = "Username already exists"
    return render_template("register.html", error=error)

@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

# =============================================================================
# HERD DASHBOARD (UI)
# =============================================================================
@app.route("/")
@login_required
def index():
    conn = db_conn()
    user_id = session.get("user_id")
    rows = conn.execute("""
        SELECT cd.id, cd.tag_id, cd.age, cd.breed,
               (SELECT mr.health_status FROM milk_records mr
                 WHERE mr.cow_id = cd.id
                 ORDER BY mr.timestamp DESC LIMIT 1) AS last_status,
               (SELECT printf('%.1f', mr.temperature) FROM milk_records mr
                 WHERE mr.cow_id = cd.id
                 ORDER BY mr.timestamp DESC LIMIT 1) AS last_temp
        FROM cow_details cd
        WHERE (cd.owner_user_id IS NULL OR cd.owner_user_id = ?)
        ORDER BY cd.tag_id
    """, (user_id,)).fetchall()
    conn.close()
    return render_template("index.html", cows=rows)

# =============================================================================
# COW CRUD & MANUAL RECORDS (UI)
# =============================================================================
@app.route("/add_cow", methods=["GET","POST"])
@login_required
def add_cow():
    if request.method == "POST":
        tag_id = request.form.get("tag_id","").strip()
        age = request.form.get("age")
        breed = request.form.get("breed")
        user_id = session.get("user_id")
        if not tag_id:
            flash("Tag ID is required", "danger")
            return redirect(url_for("add_cow"))
        conn = db_conn()
        try:
            conn.execute("INSERT INTO cow_details (tag_id, age, breed, owner_user_id) VALUES (?,?,?,?)",
                         (tag_id, age, breed, user_id))
            conn.commit()
            flash("Cow added.", "success")
        except sqlite3.IntegrityError:
            flash("Tag ID already exists", "danger")
        finally:
            conn.close()
        return redirect(url_for("index"))
    return render_template("add_cow.html")

@app.route("/edit_cow/<int:cow_id>", methods=["GET","POST"])
@login_required
def edit_cow(cow_id):
    conn = db_conn()
    cow = conn.execute("SELECT * FROM cow_details WHERE id=?", (cow_id,)).fetchone()
    if not cow:
        conn.close()
        return "Cow not found", 404
    if request.method == "POST":
        tag_id = request.form.get("tag_id","").strip()
        age = request.form.get("age")
        breed = request.form.get("breed")
        conn.execute("UPDATE cow_details SET tag_id=?, age=?, breed=? WHERE id=?",
                     (tag_id, age, breed, cow_id))
        conn.commit()
        conn.close()
        flash("Cow updated.", "success")
        return redirect(url_for("index"))
    conn.close()
    return render_template("edit_cow.html", cow=cow)

@app.route("/_deletecow/<int:cow_id>", methods=["POST"])
@login_required
def delete_cow(cow_id):
    conn = db_conn()
    conn.execute("DELETE FROM milk_records WHERE cow_id=?", (cow_id,))
    conn.execute("DELETE FROM cow_details WHERE id=?", (cow_id,))
    conn.commit()
    conn.close()
    flash("Cow and its records deleted.", "warning")
    return redirect(url_for("index"))

@app.route("/add_record/<int:cow_id>", methods=["GET","POST"])
@login_required
def add_record(cow_id):
    conn = db_conn()
    cow = conn.execute("SELECT * FROM cow_details WHERE id=?", (cow_id,)).fetchone()
    if not cow:
        conn.close()
        return "Cow not found", 404
    if request.method == "POST":
        def fget(k):
            v = request.form.get(k)
            return float(v) if v not in (None,"") else None

        temp = fget("temperature")
        ph = fget("ph")
        turb = fget("turbidity")
        cond = fget("conductivity")
        hr   = fget("heart_rate")
        fat  = fget("fat_content")

        predicted_fat = None
        predicted_status = None
        try:
            X = pd.DataFrame([[temp, ph, turb, cond, hr]], columns=["temperature","ph","turbidity","conductivity","heart_rate"])
            if fat_model is not None and all(v is not None for v in [temp, ph, turb, cond, hr]):
                predicted_fat = float(fat_model.predict(X)[0])
            if disease_model is not None and all(v is not None for v in [temp, ph, turb, cond, hr]):
                predicted_status = str(disease_model.predict(X)[0])
        except Exception as e:
            app.logger.warning("ML error (manual add): %s", e)
        if fat is None:
            fat = predicted_fat

        if not predicted_status:
            alerts = []
            for name, rng in IDEAL.items():
                val = {"temperature": temp, "ph": ph, "turbidity": turb, "conductivity": cond, "heart_rate": hr}.get(name)
                if val is None: continue
                if rng.get("min") is not None and rng.get("max") is not None:
                    if val < rng["min"] or val > rng["max"]:
                        alerts.append(name)
            predicted_status = "healthy" if len(alerts)==0 else ("warning" if len(alerts)==1 else "risk")

        conn.execute("""
            INSERT INTO milk_records
                (cow_id, temperature, ph, turbidity, conductivity, fat_content, heart_rate, predicted_fat, health_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cow_id, temp, ph, turb, cond, fat, hr, predicted_fat, predicted_status))
        conn.commit()
        conn.close()
        flash("Record saved.", "success")
        return redirect(url_for("cow_dashboard", cow_id=cow_id))
    conn.close()
    return render_template("add_record.html", cow=cow)

@app.route("/delete_record/<int:cow_id>/<int:record_id>", methods=["POST"])
@login_required
def delete_record(cow_id, record_id):
    conn = db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM milk_records WHERE id=? AND cow_id=?", (record_id, cow_id))
        if not cur.fetchone():
            return jsonify({"error": "record_not_found"}), 404
        cur.execute("DELETE FROM milk_records WHERE id=? AND cow_id=?", (record_id, cow_id))
        conn.commit()
    finally:
        conn.close()

    nxt = request.args.get("next") or url_for("cow_dashboard", cow_id=cow_id)

    # If it looks like an AJAX/fetch request, return JSON. Otherwise redirect.
    xrw = (request.headers.get("X-Requested-With") or "").lower()
    accept = (request.headers.get("Accept") or "").lower()
    if "xmlhttprequest" in xrw or "fetch" in xrw or "application/json" in accept:
        return jsonify({"deleted": True, "redirect": nxt})

    return redirect(nxt)

@app.route("/edit_record/<int:cow_id>/<int:record_id>", methods=["POST"])
@login_required
def edit_record(cow_id, record_id):
    conn = db_conn()
    try:
        def fget(k):
            v = request.form.get(k)
            return float(v) if v not in (None,"") else None
        vals = {
            "temperature": fget("temperature"),
            "ph": fget("ph"),
            "turbidity": fget("turbidity"),
            "conductivity": fget("conductivity"),
            "heart_rate": fget("heart_rate"),
            "fat_content": fget("fat_content"),
        }
        conn.execute("""
            UPDATE milk_records
               SET temperature=?, ph=?, turbidity=?, conductivity=?, heart_rate=?, fat_content=?
             WHERE id=? AND cow_id=?
        """, (vals["temperature"], vals["ph"], vals["turbidity"], vals["conductivity"],
              vals["heart_rate"], vals["fat_content"], record_id, cow_id))
        conn.commit()
    finally:
        conn.close()
    flash("Record updated.", "success")
    return redirect(url_for("cow_dashboard", cow_id=cow_id))
@app.route("/profile")
@login_required
def profile():
    return redirect(url_for("index"))

@app.route("/records/save/<int:cow_id>", methods=["POST"])
@login_required
def records_save(cow_id):
    def _num(name, cast=float):
        v = (request.form.get(name) or "").strip()
        if v == "":
            return None
        try:
            return cast(v)
        except Exception:
            return None

    temperature  = _num("temperature")
    heart_rate   = _num("heart_rate", int)
    ph           = _num("ph")
    turbidity    = _num("turbidity")
    conductivity = _num("conductivity")
    fat_content  = _num("fat_content")

    features = {
        "temperature":  temperature,
        "ph":           ph,
        "turbidity":    turbidity,
        "conductivity": conductivity,
        "heart_rate":   heart_rate,
    }

    # Predict fat if the user left it blank and the model is available
    predicted_fat = None
    if fat_content is None and 'fat_model' in globals() and fat_model:
        try:
            X = pd.DataFrame([features])
            predicted_fat = float(fat_model.predict(X)[0])
        except Exception:
            predicted_fat = None

    # Lightweight health classification (or your model if available)
    health_status = None
    if 'disease_model' in globals() and disease_model:
        try:
            X = pd.DataFrame([features])
            health_status = str(disease_model.predict(X)[0])
        except Exception:
            health_status = None
    if not health_status or health_status.lower() == "none":
        alerts = []
        for k, rng in IDEAL.items():
            v = features.get(k)
            if v is None:
                continue
            try:
                if (rng.get("min") is not None and v < rng["min"]) or \
                   (rng.get("max") is not None and v > rng["max"]):
                    alerts.append(k)
            except Exception:
                pass
        health_status = "healthy" if len(alerts) == 0 else ("warning" if len(alerts) == 1 else "risk")
            # --- compute predicted_fat ---
    # If user entered fat_content, we can still store a model estimate as predicted_fat.
    predicted_fat = predict_fat_server({
        "temperature":  temperature,
        "ph":           ph,
        "turbidity":    turbidity,
        "conductivity": conductivity,
        "heart_rate":   heart_rate,
        "fat_content":  fat_content  # model may use it or ignore it
    })


    conn = db_conn()
    try:
        conn.execute("""
          INSERT INTO milk_records
            (cow_id, timestamp, temperature, ph, turbidity, conductivity, heart_rate,
             fat_content, predicted_fat, health_status, source)
          VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')
        """, (cow_id, temperature, ph, turbidity, conductivity, heart_rate,
              fat_content, predicted_fat, health_status))
        conn.commit()
    finally:
        conn.close()

    # Clear unified buffers for this cow after saving
    PENDING[cow_id].clear()
    LISTENING[cow_id].update({"status": "idle", "vitals": False, "milk": False})

    return redirect(request.args.get("next") or url_for("cow_dashboard", cow_id=cow_id))

# =============================================================================
# CONNECT VIEW & COW DASHBOARD (UI)
# =============================================================================
@app.route("/connect/<int:cow_id>")
@login_required
def connect(cow_id):
    conn = db_conn()
    cow = conn.execute("SELECT * FROM cow_details WHERE id=?", (cow_id,)).fetchone()
    conn.close()
    if not cow: abort(404)
    return render_template("connect.html", cow=cow, ideal=IDEAL,
                           next=request.args.get("next"))

@app.route("/_oneoff_fix_nan_predicted_fat")
def oneoff_fix_nan():
    conn = db_conn()
    cur = conn.cursor()
    # In SQLite, NaN != NaN evaluates to TRUE, so we can find NaNs like this:
    cur.execute("UPDATE milk_records SET predicted_fat = NULL WHERE predicted_fat != predicted_fat;")
    # Repeat for any other columns you want to sanitize:
    for col in ("temperature","ph","turbidity","conductivity","heart_rate","fat_content"):
        cur.execute(f"UPDATE milk_records SET {col} = NULL WHERE {col} != {col};")
    conn.commit()
    conn.close()
    return "OK: cleaned NaNs -> NULL", 200

# =============================================================================
# DATA: JSON / EXPORT (NEW)
# =============================================================================
@app.route("/records_json/<int:cow_id>")
@login_required
def records_json(cow_id):
    conn = db_conn()
    df = pd.read_sql_query("""
      SELECT id, timestamp, temperature, ph, turbidity, conductivity, heart_rate,
             fat_content, predicted_fat, health_status
        FROM milk_records
       WHERE cow_id=?
       ORDER BY timestamp ASC
    """, conn, params=(cow_id,))
    conn.close()

    if df.empty:
        return jsonify([])

    # Make timestamps JSON-serializable
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype(str)

    # --- HARD SANITIZATION LAYER ---
    # 1) Ensure object dtype so .where writes real Python None (not numpy NaN)
    df = df.astype(object)

    # 2) Replace numpy/pandas NaN/NaT with None
    df = df.where(pd.notna(df), None).replace({np.nan: None})

    # 3) Double-check row-by-row (catches any weird leftovers)
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean[k] = None
            else:
                clean[k] = v
        records.append(clean)

    # 4) Serialize with allow_nan=False to forbid accidental NaN emission
    payload = json.dumps(records, allow_nan=False)
    return Response(payload, mimetype="application/json")

@app.route("/export/cow/<int:cow_id>.csv")
@login_required
def export_cow_csv(cow_id):
    conn = db_conn()
    df = pd.read_sql_query("""
      SELECT *
        FROM milk_records
       WHERE cow_id=?
       ORDER BY timestamp DESC
    """, conn, params=(cow_id,))
    conn.close()
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cow_{cow_id}_records.csv"}
    )

# =============================================================================
# DEVICE ASSIGN / PENDING / CONFIRM SAVE (kept from your file, small tweaks)
# =============================================================================
@app.route("/device/assign_tag/<mode>/<int:cow_id>", methods=["POST"])
@login_required
def device_assign_tag_and_enqueue(mode, cow_id):
    if mode not in ("vitals", "milk"):
        return jsonify({"error":"invalid_mode"}), 400

    new_tag = _cow_tag(cow_id)
    if not new_tag:
        return jsonify({"error":"cow_not_found"}), 404

    if SHARED_DEVICE_MODE:
        # In shared mode, no 'set_tag' task is needed—device is shared.
        GLOBAL_TASKS.append({"task": mode, "tag": new_tag})
        return jsonify({"queued": True, "mode": mode, "tag": new_tag, "mode_type": "shared"})

    # per-tag: find any online control device (most recent last_seen)
    conn = db_conn()
    cur = conn.cursor()
    cutoff = (datetime.utcnow() - timedelta(seconds=ONLINE_TTL_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
    ctrl = cur.execute("SELECT tag_id FROM devices WHERE last_seen >= ? ORDER BY last_seen DESC LIMIT 1", (cutoff,)).fetchone()
    if not ctrl:
        conn.close()
        return jsonify({"error":"no_online_device", "msg":"No device is currently online to assign tag"}), 409
    control_tag = ctrl["tag_id"]

    payload = json.dumps({"new_tag": new_tag})
    cur.execute("INSERT INTO device_tasks (device_tag, task, status, result) VALUES (?, 'set_tag', 'queued', ?)", (control_tag, payload))
    cur.execute("INSERT INTO device_tasks (device_tag, task, status) VALUES (?, ?, 'queued')", (new_tag, mode))
    conn.commit()
    conn.close()

    app.logger.info("ASSIGN_TAG enqueued control=%s -> new=%s mode=%s", control_tag, new_tag, mode)
    return jsonify({"queued": True, "control_tag": control_tag, "new_tag": new_tag, "mode": mode, "mode_type": "per_tag"})

@app.route("/device/pending/<int:cow_id>")
@login_required
def device_pending(cow_id):
    """Return buffered readings for this cow (not yet confirmed)."""
    # (optional) expire old sessions
    # _expire_session_if_needed(cow_id)

    # Grab the buffer
    data = PENDING.get(cow_id, {})

    # Sanitize: NaN -> None to keep JSON valid
    clean = {}
    for k, v in data.items():
        try:
            if v is None:
                clean[k] = None
            elif isinstance(v, float) and (v != v):  # NaN
                clean[k] = None
            else:
                clean[k] = v
        except Exception:
            clean[k] = None

    # (optional) log to verify form-fill
    app.logger.info("device_pending %s -> %s", cow_id, clean)

    return jsonify(clean)



@app.route("/device/confirm_save/<int:cow_id>", methods=["POST"])
@login_required
def device_confirm_save(cow_id):
    data = request.get_json(silent=True) or {}
    buf = pending_readings.get(cow_id)
    if not buf or (not buf.get("vitals") and not buf.get("milk")):
        return jsonify({"error": "no_pending_data"}), 400

    vit = buf.get("vitals", {})
    mil = buf.get("milk", {})

    def pick(k):
        if k in data and data[k] is not None:
            try: return float(data[k])
            except Exception: return None
        if k in ("temperature", "heart_rate"):
            return vit.get(k)
        return mil.get(k)

    features = {
        "temperature": pick("temperature"),
        "ph": pick("ph"),
        "turbidity": pick("turbidity"),
        "conductivity": pick("conductivity"),
        "heart_rate": pick("heart_rate"),
    }
    fat_final = pick("fat_content")

    predicted_fat = None
    predicted_status = None
    if joblib:
        try:
            X = pd.DataFrame([features])
            if fat_model is not None:
                predicted_fat = float(fat_model.predict(X)[0])
            if disease_model is not None:
                predicted_status = str(disease_model.predict(X)[0])
        except Exception as e:
            app.logger.warning("ML predict error on confirm_save: %s", e)

    if fat_final is None:
        fat_final = predicted_fat

    if not predicted_status or str(predicted_status).strip().lower() == "none":
        alerts = []
        for k, rng in IDEAL.items():
            if k in features and features[k] is not None:
                try:
                    v = float(features[k])
                    mn, mx = rng.get("min"), rng.get("max")
                    if mn is not None and mx is not None and (v < mn or v > mx):
                        alerts.append(k)
                except Exception:
                    pass
        predicted_status = "healthy" if len(alerts)==0 else ("warning" if len(alerts)==1 else "risk")

    conn = db_conn()
    try:
        conn.execute("""
            INSERT INTO milk_records
                (cow_id, timestamp, temperature, ph, turbidity, conductivity,
                 fat_content, heart_rate, predicted_fat, health_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cow_id,
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            features["temperature"], features["ph"], features["turbidity"],
            features["conductivity"], fat_final, features["heart_rate"],
            predicted_fat, predicted_status
        ))
        conn.commit()
    finally:
        conn.close()

    pending_readings.pop(cow_id, None)

    if not SHARED_DEVICE_MODE:
        try:
            conn2 = db_conn()
            cur2 = conn2.cursor()
            tag = _cow_tag(cow_id)
            if tag:
                cur2.execute("""
                   SELECT id FROM device_tasks
                   WHERE device_tag=? AND status='delivered'
                   ORDER BY delivered_at DESC LIMIT 1
                """, (tag,))
                t = cur2.fetchone()
                if t:
                    cur2.execute("UPDATE device_tasks SET status='done', result=? WHERE id=?",
                                 (json.dumps({'saved_by': session.get("username"), 'ts': datetime.utcnow().isoformat()}), t["id"]))
                    conn2.commit()
        except Exception as e:
            app.logger.warning("Failed to mark device_task done on confirm_save: %s", e)
        finally:
            try: conn2.close()
            except: pass

    return jsonify({"status": "saved", "health_status": predicted_status, "predicted_fat": predicted_fat})

@app.route("/device/clear_pending/<int:cow_id>", methods=["POST"])
@login_required
def device_clear_pending(cow_id):
    pending_readings.pop(cow_id, None)
    return jsonify({"cleared": True})

# =============================================================================
# DEBUG
# =============================================================================
@app.route("/_debug_state")
def debug_state():
    conn = db_conn()
    cur = conn.cursor()
    devices = [dict(r) for r in cur.execute("SELECT * FROM devices").fetchall()]
    tasks = [dict(r) for r in cur.execute("SELECT * FROM device_tasks ORDER BY created_at DESC LIMIT 200").fetchall()]
    conn.close()
    return jsonify({
        "devices": devices, "device_tasks": tasks,
        "listening_state": listening_state,
        "pending_readings_keys": list(pending_readings.keys()),
        "shared_mode": SHARED_DEVICE_MODE,
        "shared_device_id": SHARED_DEVICE_ID,
        "global_queue_len": len(GLOBAL_TASKS)
    })

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    ensure_tables()
    ensure_device_tables()
    ensure_admin()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
