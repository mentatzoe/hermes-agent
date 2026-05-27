import sqlite3
import re
conn = sqlite3.connect("/Users/zmll/.hermes/state.db")
c = conn.cursor()

c.execute("SELECT id, session_id, role, content FROM messages WHERE session_id IN ('20260527_003540_23db7e', '20260526_002150_0cec00', '20260527_003412_21b169d3') ORDER BY id DESC LIMIT 10")
for row in c.fetchall():
    print(row)

