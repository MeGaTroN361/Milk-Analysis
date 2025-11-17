# populate_rich_data.py
"""
Interactive demo population script (updated to match app schema).

- Uses DB_PATH env var or 'smart_farm.db' by default (same as app.py).
- Ensures required tables/columns exist (idempotent migrations).
- Lets you pick an existing username (creates 'demo' with password 'demo123' if none).
- Creates cows and populates milk_records with realistic values.
- Marks inserted rows with source='seed'.
"""

import os
import sys
import sqlite3
import random
import datetime
from sqlite3 import Connection

DB_PATH = os.environ.get("DB_PATH", "farm.db")

DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT,
    password_hash TEXT,
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
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
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
    source        TEXT,
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
    "CREATE INDEX IF NOT EXISTS idx_cow_tag ON cow_details(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_records_cow_ts ON milk_records(cow_id, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_devices_tag ON devices(tag_id);",
]

def connect() -> Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def table_columns(conn: Connection, table: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cur.fetchall()}

def add_column_if_missing(conn: Connection, table: str, coldef: str, name: str):
    cols = table_columns(conn, table)
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef};")

def ensure_schema(conn: Connection):
    conn.executescript(DDL_USERS)
    conn.executescript(DDL_COW_DETAILS)
    conn.executescript(DDL_MILK_RECORDS)
    conn.executescript(DDL_DEVICES)
    for ddl in DDL_INDEXES:
        conn.execute(ddl)
    conn.commit()
    # Migrations
    add_column_if_missing(conn, "users", "email TEXT", "email")
    add_column_if_missing(conn, "users", "password_hash TEXT", "password_hash")
    add_column_if_missing(conn, "users", "created_at DATETIME", "created_at")
    add_column_if_missing(conn, "cow_details", "owner_user_id INTEGER", "owner_user_id")
    add_column_if_missing(conn, "milk_records", "predicted_fat REAL", "predicted_fat")
    add_column_if_missing(conn, "milk_records", "health_status TEXT", "health_status")
    add_column_if_missing(conn, "milk_records", "source TEXT", "source")
    conn.commit()

def list_users(conn):
    return conn.execute("SELECT user_id, username FROM users ORDER BY username ASC;").fetchall()

def seed_demo_user_if_empty(conn):
    users = list_users(conn)
    if not users:
        # create demo user with hashed password 'demo123'
        try:
            from werkzeug.security import generate_password_hash
            pw_hash = generate_password_hash("demo123")
        except Exception:
            import hashlib
            pw_hash = hashlib.sha256("demo123".encode("utf-8")).hexdigest()
        conn.execute("INSERT OR IGNORE INTO users (username, email, password_hash) VALUES (?,?,?)",
                     ("demo", "demo@example.com", pw_hash))
        conn.commit()

def prompt_username(conn):
    seed_demo_user_if_empty(conn)
    users = list_users(conn)
    if not users:
        print("No users found. Please create a user first.")
        return None
    print("\nAvailable users:")
    for u in users:
        print(f" - {u[0]} : {u[1]}")
    while True:
        username = input("\nEnter username to assign cows to: ").strip()
        if not username:
            print("Username is required. Try again.")
            continue
        row = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,)).fetchone()
        if row:
            return row[0], username
        print("Username not found. Try again.")

def prompt_cow_tags(default_n=5):
    txt = input(f"Enter comma-separated cow tag IDs (or press Enter to auto-generate cow1..cow{default_n}): ").strip()
    if txt == "":
        tags = [f"cow{i}" for i in range(1, default_n + 1)]
        print(f"Auto-generated tags: {', '.join(tags)}")
        return tags
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    seen, tags = set(), []
    for p in parts:
        if p in seen:
            print(f"Note: duplicate tag '{p}' removed.")
            continue
        seen.add(p)
        tags.append(p)
    if not tags:
        print("No valid tags provided, aborting.")
        return None
    return tags

def prompt_int(prompt_text, default):
    raw = input(f"{prompt_text} [{default}]: ").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        print("Invalid integer, using default.")
        return int(default)

def random_timestamps(count: int, days_span: int):
    now = datetime.datetime.now()
    earliest = now - datetime.timedelta(days=days_span)
    stamps = []
    for i in range(count):
        frac = 1.0 if count == 1 else i / max(1, count - 1)
        base_ts = earliest + (now - earliest) * frac
        jitter_seconds = random.randint(-12 * 3600, 12 * 3600)
        jittered = base_ts + datetime.timedelta(seconds=jitter_seconds)
        if jittered < earliest:
            jittered = earliest + datetime.timedelta(seconds=random.randint(0, 3600))
        if jittered > now:
            jittered = now - datetime.timedelta(seconds=random.randint(0, 3600))
        stamps.append(jittered)
    random.shuffle(stamps)
    return stamps

def ensure_cow(conn, raw_tag: str, owner_user_id: int, username_prefix: str):
    final_tag = f"{username_prefix}_{raw_tag}"
    row = conn.execute("SELECT id FROM cow_details WHERE tag_id = ?", (final_tag,)).fetchone()
    if row:
        cow_id = row[0]
        print(f"Reusing cow '{final_tag}' (id={cow_id}).")
        own = conn.execute("SELECT owner_user_id FROM cow_details WHERE id = ?", (cow_id,)).fetchone()
        if own and own[0] is None and owner_user_id is not None:
            conn.execute("UPDATE cow_details SET owner_user_id=? WHERE id=?", (owner_user_id, cow_id))
            conn.commit()
        return cow_id, final_tag
    else:
        age = random.randint(2, 8)
        breed = random.choice(["Holstein", "Jersey", "Brown Swiss", "Crossbreed"])
        conn.execute(
            "INSERT INTO cow_details (tag_id, age, breed, owner_user_id) VALUES (?, ?, ?, ?)",
            (final_tag, age, breed, owner_user_id)
        )
        conn.commit()
        cow_id = conn.execute("SELECT id FROM cow_details WHERE tag_id=?", (final_tag,)).fetchone()[0]
        print(f"Created cow '{final_tag}' (id={cow_id}, age={age}, breed={breed}).")
        return cow_id, final_tag

def clear_records_for_cow(conn, cow_id: int):
    conn.execute("DELETE FROM milk_records WHERE cow_id = ?", (cow_id,))
    conn.commit()

def insert_records_for_cow(conn, cow_id: int, records_per_cow: int, days_span: int) -> int:
    stamps = random_timestamps(records_per_cow, days_span)
    if not stamps:
        return 0

    inserted = 0
    for ts in stamps:
        temperature  = round(random.uniform(36.5, 39.5), 2)
        ph           = round(random.uniform(6.2, 7.4), 2)
        turbidity    = round(random.uniform(0.0, 2.0), 2)
        conductivity = round(random.uniform(3.5, 8.0), 2)
        fat_content  = round(random.uniform(2.0, 6.5), 2)
        heart_rate   = random.randint(45, 110)

        status = "healthy"
        if conductivity > 6.0 or turbidity > 1.0 or temperature > 39.0:
            status = "udder_infection"
        elif fat_content > 5.0 and temperature < 38.0:
            status = "ketosis_risk"
        elif conductivity > 5.5 or turbidity > 0.9:
            status = "mastitis_risk"

        conn.execute(
            """
            INSERT INTO milk_records
                (cow_id, timestamp, temperature, ph, turbidity, conductivity,
                 fat_content, heart_rate, health_status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'seed')
            """,
            (cow_id, ts.strftime("%Y-%m-%d %H:%M:%S"),
             temperature, ph, turbidity, conductivity,
             fat_content, heart_rate, status)
        )
        inserted += 1
    conn.commit()
    return inserted

def seed_devices_from_cows(conn):
    conn.execute("""
        INSERT OR IGNORE INTO devices (tag_id, cow_id)
        SELECT tag_id, id FROM cow_details WHERE tag_id IS NOT NULL;
    """)
    conn.commit()

def main():
    print(f"Using DB: {DB_PATH}")
    conn = connect()
    try:
        ensure_schema(conn)

        # Make sure there's at least one user
        seed_demo_user_if_empty(conn)

        # Choose user
        users = conn.execute("SELECT user_id, username FROM users ORDER BY username ASC;").fetchall()
        print("\nAvailable users:")
        for u in users:
            print(f" - {u[0]} : {u[1]}")
        username = input("\nEnter username to assign cows to (default: demo): ").strip() or "demo"
        row = conn.execute("SELECT user_id FROM users WHERE username=?", (username,)).fetchone()
        if not row:
            print("Username not found. Exiting.")
            return
        owner_user_id = row[0]

        # Tags
        default_n = 5
        raw = input(f"Enter comma-separated cow tag IDs (or press Enter for cow1..cow{default_n}): ").strip()
        if raw:
            tags = [t.strip() for t in raw.split(",") if t.strip()]
        else:
            tags = [f"cow{i}" for i in range(1, default_n + 1)]
            print(f"Auto-generated tags: {', '.join(tags)}")

        # How many records & span
        try:
            records_per_cow = int(input("Records per cow [20]: ").strip() or "20")
        except ValueError:
            records_per_cow = 20
        try:
            days_span = int(input("Days span to distribute over [30]: ").strip() or "30")
        except ValueError:
            days_span = 30

        do_overwrite = input("If a cow exists, overwrite its existing records? (y/N): ").strip().lower() == "y"

        total_inserted = 0
        for tag in tags:
            cow_id, final_tag = ensure_cow(conn, tag, owner_user_id, username)
            if do_overwrite:
                clear_records_for_cow(conn, cow_id)
                print(f"Cleared existing records for {final_tag} (id={cow_id}).")
            inserted = insert_records_for_cow(conn, cow_id, records_per_cow, days_span)
            print(f"Inserted {inserted} records for '{final_tag}' (id={cow_id}).")
            total_inserted += inserted

        seed_devices_from_cows(conn)
        print(f"\nDone. Processed {len(tags)} cows and inserted {total_inserted} records.")
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
