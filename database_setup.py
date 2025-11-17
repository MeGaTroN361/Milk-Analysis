# database_setup.py
# Creates/patches the Smart Farm SQLite database.
# Safe to run multiple times (idempotent). Adds missing columns if needed.

import os
import sqlite3
from sqlite3 import Connection

# ---- DB path: keep in sync with app.py / config.py ----
DB_PATH = os.environ.get("DB_PATH", "farm.db")

def connect() -> Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ---- DDL (with all columns your app uses) ----
DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT,
    password_hash TEXT,                         -- REQUIRED by app.py
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_COW_DETAILS = """
CREATE TABLE IF NOT EXISTS cow_details (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id        TEXT UNIQUE NOT NULL,
    age           INTEGER,
    breed         TEXT,
    owner_user_id INTEGER,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_user_id) REFERENCES users(user_id) ON DELETE SET NULL
);
"""

DDL_MILK_RECORDS = """
CREATE TABLE IF NOT EXISTS milk_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cow_id        INTEGER NOT NULL,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    temperature   REAL,
    ph            REAL,
    turbidity     REAL,
    conductivity  REAL,
    fat_content   REAL,
    heart_rate    REAL,
    predicted_fat REAL,
    health_status TEXT,
    source        TEXT,                         -- REQUIRED by app.py
    FOREIGN KEY (cow_id) REFERENCES cow_details(id) ON DELETE CASCADE
);
"""

DDL_DEVICES = """
CREATE TABLE IF NOT EXISTS devices (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id  TEXT UNIQUE NOT NULL,
    cow_id  INTEGER,
    FOREIGN KEY (cow_id) REFERENCES cow_details(id) ON DELETE CASCADE
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
    "CREATE INDEX IF NOT EXISTS idx_cow_details_tag ON cow_details(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_records_cow_ts ON milk_records(cow_id, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_devices_tag ON devices(tag_id);",
]

# ---- Lightweight migration helpers ----
def table_columns(conn: Connection, table: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cur.fetchall()}

def add_column_if_missing(conn: Connection, table: str, coldef: str, name: str):
    cols = table_columns(conn, table)
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef};")

def create_tables(conn: Connection):
    conn.executescript(DDL_USERS)
    conn.executescript(DDL_COW_DETAILS)
    conn.executescript(DDL_MILK_RECORDS)
    conn.executescript(DDL_DEVICES)
    for ddl in DDL_INDEXES:
        conn.execute(ddl)
    conn.commit()

def migrate_columns(conn: Connection):
    # users: ensure email, password_hash, created_at
    add_column_if_missing(conn, "users", "email TEXT", "email")
    add_column_if_missing(conn, "users", "password_hash TEXT", "password_hash")
    add_column_if_missing(conn, "users", "created_at DATETIME", "created_at")

    # cow_details: ensure owner_user_id
    add_column_if_missing(conn, "cow_details", "owner_user_id INTEGER", "owner_user_id")

    # milk_records: ensure predicted_fat, health_status, source
    add_column_if_missing(conn, "milk_records", "predicted_fat REAL", "predicted_fat")
    add_column_if_missing(conn, "milk_records", "health_status TEXT", "health_status")
    add_column_if_missing(conn, "milk_records", "source TEXT", "source")

    conn.commit()

def seed_demo(conn: Connection):
    # Seed demo user if none
    cur = conn.execute("SELECT COUNT(*) FROM users;")
    if (cur.fetchone() or [0])[0] == 0:
        # Create a hashed password for demo123
        try:
            from werkzeug.security import generate_password_hash
            pw_hash = generate_password_hash("demo123")
        except Exception:
            import hashlib
            pw_hash = hashlib.sha256("demo123".encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("demo", "demo@example.com", pw_hash)
        )
        conn.commit()

    # Seed a couple cows if none
    cur = conn.execute("SELECT COUNT(*) FROM cow_details;")
    if (cur.fetchone() or [0])[0] == 0:
        cows = [
            ("COW001", 4, "Holstein Friesian"),
            ("COW002", 5, "Jersey"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO cow_details (tag_id, age, breed) VALUES (?,?,?)",
            cows
        )
        conn.commit()

def main():
    print(f"Using DB: {DB_PATH}")
    conn = connect()
    try:
        create_tables(conn)
        migrate_columns(conn)
        seed_demo(conn)
        print("Database setup complete.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
