# migrate_db.py
# Safe sqlite migration script: ensure devices & device_tasks tables exist
# and add missing columns if necessary.
import sqlite3
from sqlite3 import Connection

DB = "farm.db"

def table_exists(conn: Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def column_exists(conn: Connection, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return column in cols

def create_devices_table(conn: Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag_id TEXT UNIQUE NOT NULL,
        cow_id INTEGER,
        api_key TEXT,
        last_seen DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

def create_device_tasks_table(conn: Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS device_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_tag TEXT NOT NULL,
        task TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        delivered_at DATETIME,
        result TEXT
    );
    """)
    conn.commit()

def add_missing_columns(conn: Connection):
    # Add last_seen to devices if missing
    if not column_exists(conn, "devices", "last_seen"):
        print("Adding column devices.last_seen ...")
        conn.execute("ALTER TABLE devices ADD COLUMN last_seen DATETIME")
    else:
        print("Column devices.last_seen already exists.")

    # Ensure device_tasks has 'result' and 'delivered_at' (older versions may lack)
    dt_cols = [c for c in ["delivered_at", "result"] if not column_exists(conn, "device_tasks", c)]
    for c in dt_cols:
        print(f"Adding column device_tasks.{c} ...")
        if c == "delivered_at":
            conn.execute("ALTER TABLE device_tasks ADD COLUMN delivered_at DATETIME")
        else:
            conn.execute("ALTER TABLE device_tasks ADD COLUMN result TEXT")
    if not dt_cols:
        print("device_tasks has required columns.")

    conn.commit()

def main():
    conn = sqlite3.connect(DB)
    try:
        # Ensure tables exist
        if not table_exists(conn, "devices"):
            print("devices table does not exist. Creating it.")
            create_devices_table(conn)
        else:
            print("devices table exists.")

        if not table_exists(conn, "device_tasks"):
            print("device_tasks table does not exist. Creating it.")
            create_device_tasks_table(conn)
        else:
            print("device_tasks table exists.")

        add_missing_columns(conn)
        print("Migration completed successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
