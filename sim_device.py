# sim_device.py — simulate ESP32 device against your Flask server
# Modes:
#   - shared (default): device=esp32_1, waits for tasks and uses provided 'tag'
#   - tag: identifies as one tag and waits for tasks for that tag
#
# Endpoints used:
#   POST /device/ping
#   GET  /device/wait_task?device=<..>  (shared mode)
#   GET  /device/wait_task?tag=<..>     (per-tag mode)
#   POST /api/predict                   (sensor payload + tag)

import os
import time
import json
import random
import argparse
import requests
from datetime import datetime

DEFAULT_SERVER = "http://127.0.0.1:5000"
SHARED_DEVICE_MODE = True
SHARED_DEVICE_ID = "esp32_1"


def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def headers_common(api_key):
    return {"X-API-KEY": api_key}

def headers_json(api_key):
    return {"X-API-KEY": api_key, "Content-Type": "application/json"}

def post_ping(server, api_key, mode, device_id=None, tag=None, timeout=10):
    url = f"{server}/device/ping"
    if mode == "shared":
        body = {"device_id": device_id}
    else:
        body = {"tag": tag}
    try:
        r = requests.post(url, json=body, headers=headers_json(api_key), timeout=timeout)
        log(f"PING  -> {r.status_code} {r.text.strip()}")
        return r.status_code == 200
    except Exception as e:
        log(f"PING  !! {e}")
        return False

def wait_task(server, api_key, mode, device_id=None, tag=None, timeout=65):
    if mode == "shared":
        url = f"{server}/device/wait_task?device={device_id}"
    else:
        url = f"{server}/device/wait_task?tag={tag}"
    try:
        r = requests.get(url, headers=headers_common(api_key), timeout=timeout)
        txt = r.text.strip()
        log(f"WAIT  -> {r.status_code} {txt[:200] + ('...' if len(txt)>200 else '')}")
        if r.status_code != 200:
            return None
        # Expected: {"task": "vitals"|"milk" | null, "tag": "<cow_tag>"?}
        try:
            data = r.json()
        except Exception:
            # Some servers may return empty on timeout end; treat as no-task
            return None
        task = data.get("task")
        task_tag = data.get("tag") if mode == "shared" else tag
        if task is None:
            return None
        return {"task": str(task), "tag": task_tag}
    except requests.Timeout:
        # long-poll timed out with no task (normal)
        log("WAIT  .. long-poll timeout (no task)")
        return None
    except Exception as e:
        log(f"WAIT  !! {e}")
        return None

def post_predict(server, api_key, payload, timeout=15):
    url = f"{server}/api/predict"
    try:
        r = requests.post(url, json=payload, headers=headers_json(api_key), timeout=timeout)
        log(f"PRED  -> {r.status_code} {r.text.strip()}")
        return r.status_code == 200
    except Exception as e:
        log(f"PRED  !! {e}")
        return False

# ------- Simulated sensor data (tweak ranges to taste) -------
def gen_vitals():
    return {
        "temperature": round(random.uniform(36.9, 39.2), 2),
        "heart_rate": int(random.uniform(58, 105)),
    }

def gen_milk(include_fat=False):
    d = {
        "ph": round(random.uniform(6.4, 7.3), 2),
        "turbidity": round(random.uniform(0.15, 1.9), 3),
        "conductivity": round(random.uniform(3.9, 7.8), 2),
    }
    if include_fat:
        d["fat_content"] = round(random.uniform(2.5, 6.5), 2)
    return d
# -------------------------------------------------------------

def run_loop(
    server: str,
    api_key: str,
    mode: str,
    device_id: str,
    tag: str,
    loop_delay: float,
    include_fat: bool,
    once: bool,
):
    log(f"Simulator starting: server={server}, mode={mode}, device_id={device_id}, tag={tag}")
    post_ping(server, api_key, mode, device_id, tag)

    while True:
        taskinfo = wait_task(server, api_key, mode, device_id, tag)
        if taskinfo:
            t = taskinfo["task"]
            task_tag = taskinfo["tag"]
            log(f"TASK  == {t} for tag={task_tag}")

            if t == "vitals":
                payload = {"tag": task_tag}
                payload.update(gen_vitals())
                post_predict(server, api_key, payload)

            elif t == "milk":
                payload = {"tag": task_tag}
                payload.update(gen_milk(include_fat=include_fat))
                post_predict(server, api_key, payload)

            else:
                log(f"TASK  ?? unknown task '{t}'")

            # keep presence fresh
            post_ping(server, api_key, mode, device_id, tag)

            if once:
                log("ONE-SHOT done.")
                return
        else:
            # No task this poll; keep-alive ping every loop
            post_ping(server, api_key, mode, device_id, tag)

        time.sleep(loop_delay)

def main():
    ap = argparse.ArgumentParser(description="Simulate ESP32 shared/per-tag device for Smart Farm server")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="Flask server base URL")
    ap.add_argument("--apikey", default=os.environ.get("API_SECRET_KEY", "123"), help="API key (X-API-KEY)")
    ap.add_argument("--mode", choices=["shared","tag"], default="shared", help="Device mode")
    ap.add_argument("--device", default="esp32_1", help="Shared device id when --mode=shared")
    ap.add_argument("--tag", default="COW_TAG_1", help="Cow tag (used directly in tag mode; provided by server in shared mode)")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between loops (seconds)")
    ap.add_argument("--include-fat", action="store_true", help="Include fat_content in milk payloads")
    ap.add_argument("--once", action="store_true", help="Exit after fulfilling one task")
    args = ap.parse_args()

    run_loop(
        server=args.server.rstrip("/"),
        api_key=args.apikey,
        mode=args.mode,
        device_id=args.device,
        tag=args.tag,
        loop_delay=args.delay,
        include_fat=args.include_fat,
        once=args.once,
    )


def send_ping(server, tag):
    url = f"{server}/device/ping"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": os.environ.get("API_SECRET_KEY", "T3stKey_92#Flask!")
    }
    payload = {"tag": tag}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=6)
        print("[PING]", r.status_code, r.text)
    except Exception as e:
        print("Ping failed:", e)


if __name__ == "__main__":
    main()
