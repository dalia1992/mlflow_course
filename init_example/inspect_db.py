import sqlite3

conn = sqlite3.connect("mlflow.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cur.fetchall()]
print("Tables:", tables)

for table in ["experiments", "runs", "params", "metrics", "tags"]:
    if table not in tables:
        continue
    print(f"\n--- {table} ---")
    cur.execute(f"SELECT * FROM {table} LIMIT 20")
    cols = [d[0] for d in cur.description]
    print(cols)
    for row in cur.fetchall():
        print(row)

conn.close()
